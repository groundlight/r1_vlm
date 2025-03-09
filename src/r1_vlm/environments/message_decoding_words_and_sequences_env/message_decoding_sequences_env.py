import random
import re
import string
from typing import Any, List

import Levenshtein
from datasets import Dataset, concatenate_datasets, load_dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.environments.simple_vision_env import SimpleVisionEnv
from r1_vlm.datasets.utils import preprocess_r1_dataset


class MessageDecodingEnv(SimpleVisionEnv):
    def __init__(
        self,
        dataset: str = "sunildkumar/message-decoding-words-and-sequences-r1",
        system_prompt: str = "",
        **kwargs,  # passed to the superclass
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer", "chars"])

    def get_dataset(self) -> tuple[Dataset, Dataset]:
        dataset = load_dataset(self.dataset_name)["train"]

        # handle image injection
        dataset = preprocess_r1_dataset(dataset)
        
        # split into train and test
        splits = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = splits["train"]
        test_dataset = splits["test"]

        return train_dataset, test_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def chars_intermediate_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks if the <chars> section is correct. Reward is proportional to 1 - edit distance.
            """
            responses = [self.parser.parse(c[0]["content"]).chars for c in completions]
            true_chars = kwargs["decoded_message"]
            coded_messages = kwargs["coded_message"]

            # convert true chars to the format we expect in <chars>
            # e.g. "cat dog" -> "c a t _ d o g" or "cat" -> "c a t"
            def format_chars(text: str) -> str:
                words = text.split()
                spaced_words = [" ".join(word) for word in words]
                return " _ ".join(spaced_words)

            formatted_true_chars = [format_chars(msg) for msg in true_chars]

            def check_chars(response, answer, coded_message):
                if response is None:
                    return 0.0
                try:
                    response = response.strip()
                    answer = answer.strip()

                    edit_distance_answer = Levenshtein.distance(response, answer)
                    edit_distance_coded_message = Levenshtein.distance(
                        response, coded_message
                    )

                    # no reward if the chars data is more similar to the coded message than the answer
                    if edit_distance_coded_message < edit_distance_answer:
                        return 0.0

                    reward = 1 - edit_distance_answer / max(len(answer), len(response))

                    reward = min(max(0.0, reward), 1.0)

                    return reward

                except Exception:
                    return 0.0

            rewards = [
                check_chars(r, t, c)
                for r, t, c in zip(responses, formatted_true_chars, coded_messages)
            ]
            return rewards

        def correctness_intermediate_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that provides a soft reward for getting the <answer> section partially correct.
            Gated on getting the <chars> section correct and only using chars in the decoder (plus space).
            """
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]
            true_decoded_messages = kwargs["decoded_message"]

            def format_chars(text: str) -> str:
                words = text.split()
                spaced_words = [" ".join(word) for word in words]
                return " _ ".join(spaced_words)

            # the answer needs to be closer to the correct solution than the spaced out version
            formatted_true_decoded_messages = [
                format_chars(msg) for msg in true_decoded_messages
            ]

            def check_answer_chars(response):
                """
                Returns True if the response only contains characters in the decoder. False otherwise.
                """
                valid_characters = set(string.ascii_lowercase + " ")
                chars_in_response = set(response)
                return chars_in_response.issubset(valid_characters)

            def check_answer(response, answer, formatted_answer):
                if response is None:
                    return 0.0

                try:
                    response = response.strip()
                    answer = answer.strip()

                    # the model's answer must be closer to the correct answer than the answer separated with spaces and
                    # underscores
                    edit_distance_response_answer = Levenshtein.distance(
                        response, answer
                    )
                    edit_distance_formatted_answer_answer = Levenshtein.distance(
                        formatted_answer, answer
                    )

                    # if the answer with spaces and underscores is more similar to the correct answer than the model's answer,
                    # then no partial credit
                    if (
                        edit_distance_formatted_answer_answer
                        <= edit_distance_response_answer
                    ):
                        return 0.0

                    # if the response contains invalid characters, no reward
                    if not check_answer_chars(response):
                        return 0.0

                    # otherwise compute reward
                    reward = 1 - edit_distance_response_answer / max(
                        len(answer), len(response)
                    )

                    reward = min(max(0.0, reward), 1.0)

                    return reward

                except Exception:
                    return 0.0

            rewards = [
                check_answer(r, t, f)
                for r, t, f in zip(
                    responses, true_decoded_messages, formatted_true_decoded_messages
                )
            ]

            # gate the reward on getting the <chars> section correct
            chars_intermediate_reward = chars_intermediate_reward_func(
                completions, **kwargs
            )

            weights = [1.0 if c == 1.0 else 0.0 for c in chars_intermediate_reward]

            return [r * w for r, w in zip(rewards, weights)]

        def correctness_reward_func(completions, **kwargs) -> List[float]:
            """
            1.0 if exactly correct, otherwise 0.0. Conditioned on getting the <chars> section correct.
            """
            # parse the predicted decoded message from each completion
            responses = [self.parser.parse(c[0]["content"]).answer for c in completions]
            true_decoded_messages = kwargs["decoded_message"]

            def check_answer(response, answer):
                # the parser returns None if the answer is not found
                if response is None:
                    return 0.0

                try:
                    response = response.strip()
                    answer = answer.strip()
                except Exception as e:
                    print(f"Error in check_answer for correctness: {e}")
                    return 0.0

                if response == answer:
                    return 1.0
                else:
                    return 0.0

            # check which completions got the <chars> section correct - boolean mask
            chars_intermediate_reward = chars_intermediate_reward_func(
                completions, **kwargs
            )
            chars_correct = [r == 1.0 for r in chars_intermediate_reward]

            answers_correct = [
                check_answer(r, t) for r, t in zip(responses, true_decoded_messages)
            ]

            # achieves the answer reward IFF <chars> correct as well.
            rewards = []
            for char_correct, answer_correct in zip(chars_correct, answers_correct):
                if char_correct:
                    rewards.append(answer_correct)
                else:
                    rewards.append(0.0)

            return rewards

        def check_format(text: str) -> float:
            """
            Helper function to check if the format is correct

            Desired format:
            <think>
            Some initial thinking
            <chars> c a t </chars>
            More thinking after
            </think>
            <answer> cat </answer>
            """
            # remove the bootstrap prompt from the text if it appears at the start
            text = text.removeprefix("Let me solve this step by step.\n")

            try:
                # Strict regex that ensures:
                # 1. Only one <think> section
                # 2. Exactly one <chars> section within think
                # 3. One <answer> section at the end
                regex = r"^<think>(?:(?!<think>).)*?<chars>([\s\S]*?)<\/chars>(?:(?!<think>|<chars>).)*?<\/think>\n<answer>([\s\S]*?)<\/answer>$"

                regex_match = re.search(regex, text, re.DOTALL)

                if regex_match is None or len(regex_match.groups()) != 2:
                    return 0.0
                else:
                    return 1.0
            except Exception:
                return 0.0

        def format_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks for proper XML formatting

            Must have think and answer fields
            """

            rewards = [check_format(c[0]["content"]) for c in completions]
            return rewards

        # removed thinking_reward as it was actively hurting performance
        return [
            chars_intermediate_reward_func,
            correctness_reward_func,
            format_reward_func,
            correctness_intermediate_reward_func,
        ]
