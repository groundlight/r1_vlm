from r1_vlm.environments.simple_vision_env import SimpleVisionEnv
from r1_vlm.datasets.aok_vqa.aok_vqa_mc_r1 import AOK_VQA_MC_R1_PATH
from verifiers.parsers import XMLParser
from datasets import Dataset
from r1_vlm.datasets.utils import preprocess_r1_dataset
from trl.trainer.grpo_trainer import RewardFunc
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
import re

from r1_vlm.datasets.aok_vqa.aok_vqa_mc_r1 import create_r1_aok_vqa_mc_dataset


class AOKVQASimpleEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset: str = AOK_VQA_MC_R1_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer"])

    def get_dataset(self) -> tuple[Dataset, Dataset, Dataset]:
        dataset = create_r1_aok_vqa_mc_dataset()

        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        train_dataset = preprocess_r1_dataset(train_dataset)
        val_dataset = preprocess_r1_dataset(val_dataset)
        test_dataset = preprocess_r1_dataset(test_dataset)

        return train_dataset, val_dataset, test_dataset

    def check_format(text: str) -> float:
        """
        Checks if the format is correct for a single message.
        """
        # Find and start from the first <think> tag (removes the bootstrap prompt, if it exists)
        think_start = text.find("<think>")
        if think_start != -1:
            text = text[think_start:]

        try:
            # Check if the format is correct
            answer_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

            answer_match = re.search(answer_regex, text, re.DOTALL)

            if answer_match is not None and len(answer_match.groups()) == 2:
                return 1.0
            return 0.0
        except Exception as e:
            print(f"Error in check_format: {e}")
            return 0.0

    @staticmethod
    def preprocess_for_reward(*, prompts, completions, completions_messages):
        """
        Preprocessing for rewards

        Returns: dict with keys:
        merged_completions_messages: list[dict] - a list of the completion messages (after merging)
        texts: list[str] - a list of the text from the completion messages
        """
        merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
            prompts_messages=prompts, completions_messages=completions_messages
        )
        texts = []

        for completion_message in merged_completion_conversations:
            text = completion_message[0]["content"][0]["text"]
            texts.append(text)

        data = {
            "merged_completions_messages": merged_completion_conversations,
            "texts": texts,
        }

        return data

    def get_rubric(self) -> list[RewardFunc]:
        def format_reward_func(prompts, completions, completions_messages, **kwargs):
            """
            Provides a reward if the model's output is formatted correctly - a <think> </think> section and a <answer> </answer> section.
            """

            preprocessed_data = AOKVQASimpleEnv.preprocess_for_reward(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
            )

            texts = preprocessed_data["texts"]

            rewards = [AOKVQASimpleEnv.check_format(text) for text in texts]

            return rewards

        def correctness_reward_func(
            prompts, completions, completions_messages, **kwargs
        ):
            """
            Provides a reward if the model's output is correct.
            """
            preprocessed_data = AOKVQASimpleEnv.preprocess_for_reward(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
            )

            texts = preprocessed_data["texts"]

            # extract the answer from the completion messages
            answers = []
            for text in texts:
                parsed = self.parser.parse(text)
                if hasattr(parsed, "answer"):
                    answers.append(parsed.answer)
                else:
                    answers.append(None)

            # strip whitespace from answers
            for answer in answers:
                if isinstance(answer, str):
                    answer = answer.strip()

            correct_answers = kwargs["multiple_choice_answer"]

            if len(correct_answers) != len(answers):
                raise ValueError(
                    f"The number of correct answers ({len(correct_answers)}) does not match the number of answers ({len(answers)})"
                )

            rewards = []
            for answer, correct_answer in zip(answers, correct_answers):
                if answer == correct_answer:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)

            return rewards

        return [format_reward_func, correctness_reward_func]
