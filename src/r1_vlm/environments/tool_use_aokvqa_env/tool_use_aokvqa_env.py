import re
from typing import Any, Callable

from datasets import Dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.aok_vqa.aok_vqa_mc_tool_use_r1 import (
    create_r1_aok_vqa_tool_use_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.environments.tool_vision_env import ToolArgParser, ToolVisionEnv
from r1_vlm.tools.object_detection import (
    ObjectDetectionTool,
    detect_objects,
    parse_detect_objects_args,
    set_object_detection_tool,
)
from r1_vlm.tools.tool_prompts import SINGLE_TOOL_PROMPT_TEMPLATE
from r1_vlm.tools.zoom import parse_zoom_args, zoom

# This is a global variable that is used to store the object detection tool. It is accessed by the detect_objects function.
od_tool = ObjectDetectionTool()
set_object_detection_tool(od_tool)


class AOKVQAToolEnv(ToolVisionEnv):
    def __init__(
        self,
        processing_class: AutoProcessor,
        dataset_name: str = "Groundlight/real-iad-toy-brick-tool-use-r1",
        tools_with_parsers: list[tuple[Callable, ToolArgParser]] = [
            (detect_objects, parse_detect_objects_args),
            (zoom, parse_zoom_args),
        ],
        max_steps: int = 3,
        tool_prompt_template: str = SINGLE_TOOL_PROMPT_TEMPLATE,
    ):
        super().__init__(
            processing_class=processing_class,
            tools_with_parsers=tools_with_parsers,
            max_steps=max_steps,
            tool_prompt_template=tool_prompt_template,
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
        dataset = create_r1_aok_vqa_tool_use_dataset()

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

        # shuffle the train dataset
        train_dataset = train_dataset.shuffle()

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
                schedule = 1.0
                reward_weights.append(schedule)
            elif reward_function.__name__ == "tool_execution_reward_func":
                # no reward for tool execution
                schedule = 0.0
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
            """Soft reward function that checks if each step follows the expected format.
            Expected format: <think>...</think> followed by either <tool>...</tool> or <answer>...</answer>"""

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
                        text_content = "".join(part.get("text", "") for part in content)
                    else:
                        text_content = content
                    parsed = self.parse(text_content)
                    parsed_no_strip = self.parse(text_content, strip=False)

                    # Check specific field presence
                    has_think = (
                        hasattr(parsed, "think")
                        and getattr(parsed, "think") is not None
                    )
                    has_tool = (
                        hasattr(parsed, "tool") and getattr(parsed, "tool") is not None
                    )
                    has_answer = (
                        hasattr(parsed, "answer")
                        and getattr(parsed, "answer") is not None
                    )

                    # Check correct spacing (no extra whitespace within tags)
                    # This requires checking non-stripped parsing results for *present* fields
                    has_correct_spacing = True
                    if has_think and not (
                        hasattr(parsed_no_strip, "think")
                        and getattr(parsed_no_strip, "think") is not None
                    ):
                        has_correct_spacing = False
                    if has_tool and not (
                        hasattr(parsed_no_strip, "tool")
                        and getattr(parsed_no_strip, "tool") is not None
                    ):
                        has_correct_spacing = False
                    if has_answer and not (
                        hasattr(parsed_no_strip, "answer")
                        and getattr(parsed_no_strip, "answer") is not None
                    ):
                        has_correct_spacing = False

                    # Check structural validity: think + (tool XOR answer)
                    is_valid_structure = has_think and (
                        (has_tool and not has_answer) or (not has_tool and has_answer)
                    )

                    # Check start and end tags based on expected structure
                    text_stripped = text_content.strip()
                    starts_with_think = text_stripped.startswith("<think>")
                    ends_with_tool = text_stripped.endswith("</tool>")
                    ends_with_answer = text_stripped.endswith("</answer>")

                    # Valid end requires ending with the tag that is present (tool or answer)
                    valid_end = (has_tool and ends_with_tool) or (
                        has_answer and ends_with_answer
                    )

                    # Calculate score
                    format_score = 0.0
                    if is_valid_structure:
                        format_score += 0.4  # Core structure reward
                    if has_correct_spacing:
                        format_score += 0.2
                    if (
                        starts_with_think
                    ):  # Should always start with think if structure is valid
                        format_score += 0.2
                    if valid_end:  # Should end with tool/answer if structure is valid
                        format_score += 0.2

                    format_scores.append(format_score)
                if not format_scores:
                    return 0.0
                # Return the average score over all assistant messages in the trajectory
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

        def check_execution(conversation):
            """
            Returns the ratio of successful tool executions to total attempts.
            """
            tool_attempts = 0
            successful_executions = 0

            for i, message in enumerate(conversation):
                if message["role"] == "assistant":
                    parsed = self.parser.parse(message["content"][0]["text"])
                    if hasattr(parsed, "tool") and parsed.tool is not None:
                        tool_attempts += 1
                        if (
                            i + 1 < len(conversation)
                            and conversation[i + 1]["role"] == "user"
                        ):
                            response = conversation[i + 1]["content"][0]["text"]
                            if "Error:" not in response:
                                successful_executions += 1

            return 0.0 if tool_attempts == 0 else successful_executions / tool_attempts

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

            rewards: list[float] = [
                check_execution(conv) for conv in merged_completion_conversations
            ]

            return rewards

        def check_correctness(conversation, correct_answer) -> bool:
            text = conversation[-1]["content"][0]["text"]

            parsed = self.parser.parse(text)
            if hasattr(parsed, "answer"):
                answer = parsed.answer
            else:
                answer = None

            return answer == correct_answer

        def correct_answer_reward_func(
            prompts, completions, completions_messages, **kwargs
        ) -> list[float]:
            """
            Provides a reward if the model's answer is correct. Gates on the model either not using tools or using tools correctly.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
                prompts_messages=prompts, completions_messages=completions_messages
            )

            correct_answers = kwargs["multiple_choice_answer"]

            correctness_results: list[bool] = [
                check_correctness(conv, correct_answer)
                for conv, correct_answer in zip(
                    merged_completion_conversations, correct_answers
                )
            ]

            return [1.0 if result else 0.0 for result in correctness_results]

        return [
            format_reward_func,
            tool_execution_reward_func,
            correct_answer_reward_func,
        ]

    def log_metrics(self, conversations, completions_text, completion_messages):
        # 1. compute how many completions attempt to use any tool
        # 2. for each tool, compute how many completions attempt to use it

        completions_with_tool_use = 0
        completions_with_zoom_use = 0
        completions_with_detect_objects_use = 0

        for completion in completions_text:
            tool_use_regex = r"<tool>(.*?)</tool>"
            zoom_use_string = "name: zoom"
            detect_objects_use_string = "name: detect_objects"

            tool_matches = re.findall(tool_use_regex, completion, re.DOTALL)
            if tool_matches:
                completions_with_tool_use += 1
                for tool_content in tool_matches:
                    if zoom_use_string in tool_content:
                        completions_with_zoom_use += 1
                    if detect_objects_use_string in tool_content:
                        completions_with_detect_objects_use += 1

        print(
            f"There are {len(completions_text)} completions, {completions_with_tool_use} of which attempt to use a tool, {completions_with_zoom_use} of which attempt to use zoom, and {completions_with_detect_objects_use} of which attempt to use detect_objects"
        )

        num_completions = len(completions_text)
        tool_use_proportion = completions_with_tool_use / num_completions
        zoom_use_proportion = completions_with_zoom_use / num_completions
        detect_objects_use_proportion = (
            completions_with_detect_objects_use / num_completions
        )

        return {
            "tool_use_proportion": tool_use_proportion,
            "zoom_use_proportion": zoom_use_proportion,
            "detect_objects_use_proportion": detect_objects_use_proportion,
        }


if __name__ == "__main__":
    env = AOKVQAToolEnv(processing_class=None)
    train_dataset, val_dataset, test_dataset = env.get_dataset()
    import ipdb

    ipdb.set_trace()
    print("hi")
