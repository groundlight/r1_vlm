import re
from typing import Any, List

from datasets import Dataset, concatenate_datasets, load_dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.simple_vision_env import SimpleVisionEnv


class DigitRecognitionEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset: str = "sunildkumar/digit-recognition-r1",
        system_prompt: str = "",
        **kwargs,  # passed to the superclass
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer"])

    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset_name)

        # select all three splits
        digits_1 = dataset["digits_1"]
        digits_2 = dataset["digits_2"]
        digits_3 = dataset["digits_3"]

        # concatenate the three splits
        dataset = concatenate_datasets([digits_1, digits_2, digits_3])

        # handle image injection
        dataset = preprocess_r1_dataset(dataset)

        return dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def _recognition_correctness_reward_func(completions, **kwargs) -> List[float]:
            """Reward function for recognition task"""

            # verify that the task is recognition for all instances. If it isn't assumptions downstream fail
            tasks = kwargs["task"]
            if not all(t == "recognition" for t in tasks):
                raise ValueError("All instances must be recognition tasks")

            # the answer in this case is the label, which is a list of list of ints, e.g. [[0, 5], [0, 5], [2, 8], [2, 8]]
            answers = kwargs["label"]

            # parse the last message in each completion (completions are conversations) for data between <answer> and </answer> tags
            # only selects the data between the first answer tags if there are multiple valid sets per completion.
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]

            def parse_list(r: str) -> List[int]:
                try:
                    # Remove brackets and split by commas
                    nums = [int(x.strip()) for x in r.strip("[]").split(",")]
                    return nums
                except:
                    return []

            parsed_responses = [parse_list(r) for r in responses]
            return [
                1.0 if r == sorted(a) else 0.0
                for r, a in zip(parsed_responses, answers)
            ]

        def _addition_correctness_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function for addition task
            """

            answers = kwargs["total"]
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]

            def check_answer(response, answer):
                try:
                    response = int(response)
                    answer = int(answer)

                    if response == answer:
                        return 1.0
                    else:
                        return 0.0

                except Exception as e:
                    print(f"Error in _addition_correctness_reward_func: {e}")
                    return 0.0

            rewards = [check_answer(r, a) for r, a in zip(responses, answers)]
            return rewards

        def correctness_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks if the predicted digits match the true labels"""

            tasks = kwargs["task"]
            if not (task == tasks[0] for task in tasks):
                raise ValueError(
                    "All tasks must be same type, invalidates an assumption of the code"
                )

            task = tasks[0]

            if task == "recognition":
                return _recognition_correctness_reward_func(completions, **kwargs)
            elif task == "addition":
                return _addition_correctness_reward_func(completions, **kwargs)
            else:
                raise ValueError(f"Invalid task: {task}")

        def format_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks for proper XML formatting

            Must have think and answer fields
            """

            def check_format(text: str) -> float:
                # remove the bootstrap prompt from the text if it appears at the start
                text = text.removeprefix("Let me solve this step by step.\n")

                try:
                    # Check if the format is correct
                    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                    match = re.search(regex, text, re.DOTALL)

                    if match is None or len(match.groups()) != 2:
                        return 0.0
                    else:
                        return 1.0
                except Exception as e:
                    print(f"Error in format_reward_func: {e}")
                    return 0.0

            rewards = [check_format(c[0]["content"]) for c in completions]
            return rewards

        return [correctness_reward_func, format_reward_func]
