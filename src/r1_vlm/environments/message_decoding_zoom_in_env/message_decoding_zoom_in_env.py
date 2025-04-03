import re
import string
import Levenshtein
from statistics import mean
from typing import Any, Callable

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.tool_vision_env import ToolVisionEnv
from r1_vlm.tools.message_decoding_zoom_in_tool import zoom_in


class MessageDecodingZoomInEnv(ToolVisionEnv):
    def __init__(self,
                 processing_class: AutoProcessor,
                 dataset_name: str = "Groundlight/message-decoding-words-and-sequences-zoom-in-r1",
                #  tool that simulates zoom-in by reconstructing the zoomed-in image from the full coordinates of texts
                 tools: list[Callable] = [zoom_in],
                 max_steps: int = 10,
                 **kwargs,
                 ):

        super().__init__(
            tools=tools,
            processing_class=processing_class,
            max_steps=max_steps,
        )
        
        self.dataset_name = dataset_name

    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset_name)
        dataset = dataset["train"]
        dataset = self.inject_system_prompt(dataset)
        dataset = preprocess_r1_dataset(dataset)
        return dataset
    
    def get_assistant_messages(self, conversation: list[dict[str, Any]]) -> list[str]:
        '''
        Returns the assistant messages from the completion messages as a list of strings.
        '''
        assistant_messages = [message["content"][0]["text"] for message in conversation if message["role"] == "assistant"]
        return assistant_messages
    
    def get_rubric(self) -> list[RewardFunc]:
        
        def format_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            '''
            Returns the average compliance over all model messages in the completion.
            
            prompts: list of messages that make up the original prompt
            completions: list of completion strings (not used, but required by the interface)
            completions_messages: list of messages in the completion
            '''
            
            merged_completion_conversations = self.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
            rewards = []
            for conversation in merged_completion_conversations:
                assistant_messages = self.get_assistant_messages(conversation)
                
                format_correct = [check_format(message) for message in assistant_messages]
                # the last message should be an answer but not tool call to receive a reward
                if len(format_correct) > 0:
                    format_correct[-1] = check_format(assistant_messages[-1], answer_only=True)
                format_correct = mean(format_correct)
                rewards.append(format_correct)
                
            return rewards

        def check_format(text: str, answer_only: bool = False) -> float:
            '''
            Checks if the format is correct for a single message.
            '''
            # Find and start from the first <think> tag (removes the bootstrap prompt, if it exists)
            think_start = text.find("<think>")
            if think_start != -1:
                text = text[think_start:]

            try:
                # Check if the format is correct
                answer_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
                tool_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<tool>([\s\S]*?)<\/tool>$"

                answer_match = re.search(answer_regex, text, re.DOTALL)
                tool_match = re.search(tool_regex, text, re.DOTALL)

                if answer_only:
                    if answer_match is not None and len(answer_match.groups()) == 2:
                        return 1.0
                    else:
                        return 0.0
                else:
                    if (answer_match is not None and len(answer_match.groups()) == 2) or \
                    (tool_match is not None and len(tool_match.groups()) == 2):
                        return 1.0
                    else:
                        return 0.0
            except Exception as e:
                print(f"Error in check_format: {e}")
                return 0.0

        def correctness_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            """Reward function that checks if the predicted answer matches the true answer. Only checks the last message in the completion."""
            # parse the predicted decoded message from each completion

            merged_completion_conversations = self.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            # parse the last message in each completion (completions are conversations) for data between <answer> and </answer> tags
            responses = [self.llm_parser.parse(c[-1]["content"][0]["text"]).answer for c in merged_completion_conversations]
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
            
            answers_correct = [check_answer(r, t) for r, t in zip(responses, true_decoded_messages)]
            return answers_correct

        def tool_execution_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            """
            Reward function that checks if tools were executed successfully.
            Returns a reward based on the ratio of successful tool executions to total attempts.
            """
            merged_completion_conversations = self.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
            def check_execution(conversation):
                tool_attempts = 0
                successful_executions = 0
                
                for i, message in enumerate(conversation):
                    if message["role"] == "assistant":
                        parsed = self.llm_parser.parse(message["content"][0]["text"])
                        if hasattr(parsed, "tool") and parsed.tool is not None:
                            tool_attempts += 1
                            if i + 1 < len(conversation) and conversation[i + 1]["role"] == "user":
                                response = conversation[i + 1]["content"][0]["text"]
                                if "Error:" not in response:
                                    successful_executions += 1
                
                return 0.0 if tool_attempts == 0 else successful_executions / tool_attempts
            
            rewards = [check_execution(conv) for conv in merged_completion_conversations]
            return rewards

        def chars_intermediate_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            """
            Reward function that checks if the <chars> section is correct. Reward is proportional to 1 - edit distance.
            """
            merged_completion_conversations = self.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            responses = [self.llm_parser.parse(c[-1]["content"][0]["text"]).chars for c in merged_completion_conversations]
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

        def correctness_intermediate_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            """
            Reward function that provides a soft reward for getting the <answer> section partially correct.
            Gated on getting the <chars> section correct and only using chars in the decoder (plus space).
            """
            merged_completion_conversations = self.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            responses = [self.llm_parser.parse(c[-1]["content"][0]["text"]).answer for c in merged_completion_conversations]
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
                prompts, completions, completions_messages, **kwargs
            )

            weights = [1.0 if c == 1.0 else 0.0 for c in chars_intermediate_reward]

            return [r * w for r, w in zip(rewards, weights)]
        
        return [format_reward_func, correctness_reward_func, tool_execution_reward_func, chars_intermediate_reward_func, correctness_intermediate_reward_func]
