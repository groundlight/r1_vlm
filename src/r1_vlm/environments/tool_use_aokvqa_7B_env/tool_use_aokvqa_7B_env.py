from typing import Any, Callable

from datasets import Dataset, concatenate_datasets
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.aok_vqa.aok_vqa_mc_tool_use_7B_r1 import (
    create_r1_aok_vqa_tool_use_7B_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.environments.reward_schedules import create_linear_decay_schedule
from r1_vlm.environments.tool_vision_env import ToolVisionEnv
from r1_vlm.tools.zoom import zoom


class AOKVQAToolEnv(ToolVisionEnv):
    def __init__(
        self,
        processing_class: AutoProcessor,
        dataset_name: str = "Groundlight/real-iad-toy-brick-tool-use-r1",
        tools: list[Callable] = [zoom],
        max_steps: int = 3,
    ):
        super().__init__(
            processing_class=processing_class,
            tools=tools,
            max_steps=max_steps,
        )

        self.dataset_name = dataset_name
        self.parser = XMLParser(fields=["think", "answer", "tool"])
        self._fields = [
            ("think", ["think"]),
            ("answer", ["answer"]),
            ("tool", ["tool"]),
        ]

    def parse(self, text: str, strip: bool = True):
        return self.parser.parse(text, strip=strip)

    def get_dataset(self) -> tuple[Dataset, Dataset, Dataset]:
        dataset = create_r1_aok_vqa_tool_use_7B_dataset()

        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        # handle tool use system prompt injection
        train_dataset = self.inject_system_prompt(train_dataset)
        val_dataset = self.inject_system_prompt(val_dataset)
        test_dataset = self.inject_system_prompt(test_dataset)

        train_dataset = preprocess_r1_dataset(train_dataset)
        val_dataset = preprocess_r1_dataset(val_dataset)
        test_dataset = preprocess_r1_dataset(test_dataset)

        # reorganize the train dataset to frontload harder data.
        original_len = len(train_dataset)

        # sort so the difficult examples are at the top of the stack. We want to use these!
        train_dataset = train_dataset.sort("difficult_direct_answer", reverse=True)

        # start with some easy examples first, then move to train on the difficult examples
        num_easy = 100
        total_len = len(train_dataset)
        easiest = train_dataset.select(range(total_len - num_easy, total_len))
        rest = train_dataset.select(range(total_len - num_easy))
        train_dataset = concatenate_datasets([easiest, rest])

        assert len(train_dataset) == original_len

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def preprocess_for_reward(*, prompts, completions, completions_messages):
        """
        Preprocessing for rewards

        Returns: dict with keys
        merged_completion_conversations: list[dict] - a list of the completion conversations (after merging)
        """
        merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
            prompts_messages=prompts, completions_messages=completions_messages
        )

        return {
            "merged_completion_conversations": merged_completion_conversations,
        }

    @staticmethod
    def get_assistant_messages(conversation: list[dict[str, Any]]) -> list[str]:
        """
        Returns the assistant messages from the completion messages as a list of strings.
        """
        assistant_messages = [
            message["content"][0]["text"]
            for message in conversation
            if message["role"] == "assistant"
        ]
        return assistant_messages

    def get_reward_weights(self) -> list[float]:
        reward_functions = self.get_rubric()
        reward_weights = []
        for reward_function in reward_functions:
            if reward_function.__name__ == "format_reward_func":
                # consistent small reward for formatting properly
                schedule = 0.2
                reward_weights.append(schedule)
            elif reward_function.__name__ == "tool_execution_reward_func":
                # having proper formatting will be rewarded more heavily to start, and taper off
                schedule = create_linear_decay_schedule(
                    start_val=1.0, end_val=0.1, n_steps=1500
                )
                reward_weights.append(schedule)

            elif reward_function.__name__ == "correct_answer_reward_func":
                # consistent high reward for getting the answer right
                schedule = 1.0
                reward_weights.append(schedule)
            else:
                raise ValueError(
                    f"Unknown reward function: {reward_function.__name__} encountered in get_reward_weights"
                )

        assert len(reward_weights) == len(reward_functions), (
            f"reward_weights and reward_functions should be the same length, but got {len(reward_weights)} and {len(reward_functions)}"
        )
        return reward_weights

    def get_rubric(self) -> list[RewardFunc]:
        def format_reward_func(prompts, completions, completions_messages, **kwargs):
            """Soft reward function that checks if each step follows the expected format."""

            def check_format(trajectory):
                model_messages = [
                    msg for msg in trajectory if msg["role"] == "assistant"
                ]
                if not model_messages:
                    return 0.0
                format_scores = []
                for msg in model_messages:
                    content = msg["content"]
                    if isinstance(content, list):
                        text_content = " ".join(
                            part.get("text", "") for part in content
                        )
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
                            if (
                                hasattr(parsed, alt)
                                and getattr(parsed, alt) is not None
                            ):
                                has_any_field = True
                                total_fields += 1
                                field_set_present = True
                                if not (
                                    hasattr(parsed_no_strip, alt)
                                    and getattr(parsed_no_strip, alt) is not None
                                ):
                                    has_correct_spacing = False
                            elif (
                                text_content.count(f"<{alt}>") > 0
                                or text_content.count(f"</{alt}>") > 0
                            ):
                                total_fields += 1
                                field_set_present = True
                        if field_set_present:
                            present_field_sets.add(i)

                    format_score = 0.0
                    starts_with_any_field = any(
                        text_content.strip().startswith(f"<{alt}>")
                        for alt in self._fields[0][1]
                    )
                    ends_with_any_field = any(
                        text_content.strip().endswith(f"</{alt}>")
                        for alt in self._fields[-1][1]
                    )

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

            preprocessed_data = AOKVQAToolEnv.preprocess_for_reward(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
            )

            merged_completion_conversations = preprocessed_data[
                "merged_completion_conversations"
            ]

            return [check_format(m) for m in merged_completion_conversations]

        def tool_execution_reward_func(
            prompts, completions, completions_messages, **kwargs
        ) -> list[float]:
            """
            Reward function that checks if tools were executed successfully.
            Returns a reward based on the ratio of successful tool executions to total attempts.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
                prompts_messages=prompts, completions_messages=completions_messages
            )

            def check_execution(conversation):
                tool_attempts = 0
                successful_executions = 0

                for i, message in enumerate(conversation):
                    if message["role"] == "assistant":
                        parsed = self.llm_parser.parse(message["content"][0]["text"])
                        if hasattr(parsed, "tool") and parsed.tool is not None:
                            tool_attempts += 1
                            if (
                                i + 1 < len(conversation)
                                and conversation[i + 1]["role"] == "user"
                            ):
                                response = conversation[i + 1]["content"][0]["text"]
                                if "Error:" not in response:
                                    successful_executions += 1

                return (
                    0.0 if tool_attempts == 0 else successful_executions / tool_attempts
                )

            rewards = [
                check_execution(conv) for conv in merged_completion_conversations
            ]
            return rewards

        def correct_answer_reward_func(
            prompts, completions, completions_messages, **kwargs
        ) -> list[float]:
            """
            Provides a reward if the model's answer is correct.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
                prompts_messages=prompts, completions_messages=completions_messages
            )

            # select the last message in each completion (completions are conversations)
            texts = [
                c[-1]["content"][0]["text"] for c in merged_completion_conversations
            ]

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

        return [
            format_reward_func,
            tool_execution_reward_func,
            correct_answer_reward_func,
        ]
