import json

from datasets import Dataset, concatenate_datasets

from r1_vlm.datasets.aok_vqa.aok_vqa_mc_r1 import (
    AOK_VQA_MC_R1_PATH,
    create_r1_aok_vqa_mc_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.environments.simple_vision_env import SimpleVisionEnv
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser


class AOKVQASimpleEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset: str = AOK_VQA_MC_R1_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer"])
        self._fields = [("think", ["think"]), ("answer", ["answer"])]

    def parse(self, text: str, strip: bool = True):
        return self.parser.parse(text, strip=strip)

    def get_dataset(self) -> tuple[Dataset, Dataset, Dataset]:
        dataset = create_r1_aok_vqa_mc_dataset()

        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        
        
        

        train_dataset = preprocess_r1_dataset(train_dataset)
        val_dataset = preprocess_r1_dataset(val_dataset)
        test_dataset = preprocess_r1_dataset(test_dataset)
        
        # sort so the difficult examples are at the top of the stack. We want to use these!
        train_dataset = train_dataset.sort("difficult_direct_answer", reverse=True)
        
        # start with some easy examples first, then move to train on the difficult examples
        num_easy = 100
        easiest = train_dataset[-num_easy:]
        train_dataset = train_dataset[:-num_easy]
        train_dataset = concatenate_datasets([easiest, train_dataset])
        
        for example in train_dataset:
            print(example["difficult_direct_answer"])
            
        return train_dataset, val_dataset, test_dataset

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
            """Soft reward function that checks if each step follows the expected format."""
            def check_format(trajectory):
                model_messages = [msg for msg in trajectory if msg["role"] == "assistant"]
                if not model_messages:
                    return 0.0
                format_scores = []
                for msg in model_messages:
                    content = msg["content"]
                    if isinstance(content, list):
                        text_content = " ".join(part.get("text", "") for part in content)
                    else:
                        text_content = content
                    parsed = self.parse(text_content)
                    parsed_no_strip = self.parse(text_content, strip=False)

                    has_any_field = False
                    total_fields = 0
                    expected_field_count = len(self._fields)
                    present_field_sets = set()
                    has_correct_spacing = True

                    for i, (canonical, alternatives) in enumerate(self._fields):
                        field_set_present = False
                        for alt in alternatives:
                            if hasattr(parsed, alt) and getattr(parsed, alt) is not None:
                                has_any_field = True
                                total_fields += 1
                                field_set_present = True
                                if not (hasattr(parsed_no_strip, alt) and getattr(parsed_no_strip, alt) is not None):
                                    has_correct_spacing = False
                            elif text_content.count(f"<{alt}>") > 0 or text_content.count(f"</{alt}>") > 0:
                                total_fields += 1
                                field_set_present = True
                        if field_set_present:
                            present_field_sets.add(i)

                    format_score = 0.0
                    starts_with_any_field = any(text_content.strip().startswith(f"<{alt}>") for alt in self._fields[0][1])
                    ends_with_any_field = any(text_content.strip().endswith(f"</{alt}>") for alt in self._fields[-1][1])

                    if has_any_field:
                        field_set_ratio = len(present_field_sets) / expected_field_count
                        format_score += 0.4 * field_set_ratio
                    if has_correct_spacing:
                        format_score += 0.2
                    if starts_with_any_field:
                        format_score += 0.2
                    if ends_with_any_field:
                        format_score += 0.2

                    format_scores.append(format_score)
                if not format_scores:
                    return 0.0
                return sum(format_scores) / len(format_scores)
            
            preprocessed_data = AOKVQASimpleEnv.preprocess_for_reward(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
            )

            merged_completions_messages = preprocessed_data["merged_completions_messages"]
            
            return [check_format(m) for m in merged_completions_messages]

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

        def record_data_func(prompts, completions, completions_messages, **kwargs):
            """
            Records data to disk for analysis, returns 0.0 reward for all examples.
            """

            # schema:
            # the prompts
            # the completions
            # if the model's answer was correct

            preprocessed_data = AOKVQASimpleEnv.preprocess_for_reward(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
            )

            texts = preprocessed_data["texts"]

            thinking_texts = []
            answer_texts = []
            for text in texts:
                parsed = self.parser.parse(text)
                if hasattr(parsed, "think"):
                    thinking_texts.append(parsed.think)
                else:
                    thinking_texts.append(None)
                if hasattr(parsed, "answer"):
                    answer_texts.append(parsed.answer)
                else:
                    answer_texts.append(None)

            correctness_rewards = correctness_reward_func(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
                **kwargs,
            )

            question_ids = kwargs["question_id"]
            questions = kwargs["question"]
            choices = kwargs["choices"]
            difficult_direct_answer = kwargs["difficult_direct_answer"]
            correct_choice_idx = kwargs["correct_choice_idx"]
            direct_answers = kwargs["direct_answers"]
            rationales = kwargs["rationales"]

            # now we can assemble an example for each completion and write it to disk
            for (
                question_id,
                question,
                choices,
                difficult_direct_answer,
                correct_choice_idx,
                direct_answers,
                rationales,
                completion,
                correctness_reward,
            ) in zip(
                question_ids,
                questions,
                choices,
                difficult_direct_answer,
                correct_choice_idx,
                direct_answers,
                rationales,
                texts,
                correctness_rewards,
            ):
                example = {
                    "question_id": question_id,
                    "question": question,
                    "choices": choices,
                    "difficult_direct_answer": difficult_direct_answer,
                    "correct_choice_idx": correct_choice_idx,
                    "direct_answers": direct_answers,
                    "rationales": rationales,
                    "completion": completion,
                    "correctness_reward": correctness_reward,
                }

                # save the example to disk
                with open(
                    "/millcreek/home/sunil/r1_vlm_bumbershoot0/r1_vlm/src/r1_vlm/environments/simple_aokvqa_env/aokvqa_examples.jsonl",
                    "a",
                ) as f:
                    f.write(json.dumps(example) + "\n")

            # No reward for this function
            return [0.0 for _ in range(len(completions))]

        return [format_reward_func, correctness_reward_func]

if __name__ == "__main__":
    env = AOKVQASimpleEnv()
    train_dataset, val_dataset, test_dataset = env.get_dataset()
    print(train_dataset)
