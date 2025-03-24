import re
from statistics import mean
from typing import Any, List

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.double_check_env import DoubleCheckVisionEnv


class DigitsDoubleCheckEnv(DoubleCheckVisionEnv):
    def __init__(
        self,
        dataset_name: str = "sunildkumar/digits-doublecheck-r1",
        mask_env_response: bool = True,
        processing_class: AutoProcessor = None,
    ):
        super().__init__(
            mask_env_response=mask_env_response,
            processing_class=processing_class,
        )
        self.dataset_name = dataset_name
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

    
    def get_rubric(self, **kwargs: Any) -> list[RewardFunc]:
        '''
        The correctness rewards only apply to the last message in the completion, not the intermediate completion message from the model.
        '''
        
        def _recognition_correctness_reward_func(completion_conversations, **kwargs) -> list[float]:
                """Reward function for recognition task"""

                # verify that the task is recognition for all instances. If it isn't assumptions downstream fail
                tasks = kwargs["task"]
                if not all(t == "recognition" for t in tasks):
                    raise ValueError("All instances must be recognition tasks")

                # the answer in this case is the label, which is a list of list of ints, e.g. [[0, 5], [0, 5], [2, 8], [2, 8]]
                answers = kwargs["label"]

                # parse the last message in each completion (completions are conversations) for data between <answer> and </answer> tags
                # only selects the data between the first answer tags in the message if there are multiple valid answer sets in it.
                responses = [self.parser.parse(c[-1]["content"][0]["text"]).answer for c in completion_conversations]

                def parse_list(r: str) -> List[int]:
                    try:
                        # Remove brackets and split by commas
                        nums = [int(x.strip()) for x in r.strip("[]").split(",")]
                        return nums
                    except: # noqa: E722
                        return []

                parsed_responses = [parse_list(r) for r in responses]
                return [
                    1.0 if r == sorted(a) else 0.0
                    for r, a in zip(parsed_responses, answers)
                ]
    
        def _addition_correctness_reward_func(completion_conversations, **kwargs) -> List[float]:
            """
            Reward function for addition task
            """
            
            # verify that the task is recognition for all instances. If it isn't assumptions downstream fail
            tasks = kwargs["task"]
            if not all(t == "addition" for t in tasks):
                raise ValueError("All instances must be addition tasks")
            
            # the answer is a list of ints, e.g. [1, 2, 3, 4]
            answers = kwargs["total"]
            
            # parse the last message in each completion (completions are conversations) for data between <answer> and </answer> tags
            # only selects the data between the first answer tags in the message if there are multiple valid answer sets in it.
            responses = [self.parser.parse(c[-1]["content"][0]["text"]).answer for c in completion_conversations]

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

        def correctness_reward_func(prompts, completions, completions_messages, **kwargs) -> List[float]:
            """Reward function that checks if the predicted digits match the true labels. Only checks the last message in the completion."""

            tasks = kwargs["task"]
            if not (task == tasks[0] for task in tasks):
                raise ValueError(
                    "All tasks must be same type, invalidates an assumption of the code"
                )

            task = tasks[0]
            
            merged_completion_conversations = self.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)

            if task == "recognition":
                return _recognition_correctness_reward_func(completion_conversations=merged_completion_conversations, **kwargs)
            elif task == "addition":
                return _addition_correctness_reward_func(completion_conversations=merged_completion_conversations, **kwargs)
            else:
                raise ValueError(f"Invalid task: {task}")
            
        
        def format_reward_func(prompts, completions, completions_messages, **kwargs) -> List[float]:
            '''
            Returns the average compliance over all model messages in the completion.
            
            prompts: list of messages that make up the original prompt
            completions: list of completion strings (not used, but required by the interface)
            completions_messages: list of messages in the completion
            '''
            
            # all messages that are part of the completion
            merged_completion_conversations = self.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
            # average the format correctness over all model messages per completion
            rewards = []
            for conversation in merged_completion_conversations:
                model_messages = []
                for element in conversation:
                    if element["role"] == "assistant":
                        for message in element["content"]:
                            if message["type"] == "text":
                                model_messages.append(message["text"])
                            else:
                                raise ValueError(f"Invalid message type: {message['type']} from model. The model should only return text messages.")
                
                if len(model_messages) != 2:
                    raise ValueError(f"There should be two model messages in the completion, before and after the environment response. Found {len(model_messages)}. {model_messages=}")
                
                # check the format of each model message
                format_correct = [check_format(message) for message in model_messages]
                
                format_correct = mean(format_correct)
                rewards.append(format_correct)
                
            return rewards
                
            
        def check_format(text: str) -> float:
            '''
            Checks if the format is correct for a single message.
            '''
            # Find and start from the first <think> tag (removes the bootstrap prompt, if it exists)
            think_start = text.find("<think>")
            if think_start != -1:
                text = text[think_start:]

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
        
        return [correctness_reward_func, format_reward_func]
        