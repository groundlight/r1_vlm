import re
from statistics import mean
from typing import Any, Callable

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.environments.tool_vision_env import ToolVisionEnv
from r1_vlm.tools.digits_answer_tool import get_answer


class DigitsToolUseEnv(ToolVisionEnv):
    '''
    This env tests the ability of the model to use tools, in this case a tool that "cheats" by looking up the correct answer given the input image.
    This is meant as a sanity check/development environment to ensure that the tool calling works properly, not a proper experiment.
    '''
    
    def __init__(
        self,
        processing_class: AutoProcessor,
        dataset_name: str = "sunildkumar/digit-recognition-tool-use-r1",
        # tool that directly gets the answer from the dataset
        tools: list[Callable] = [get_answer],
        max_steps: int = 10,
        ):
        
        super().__init__(
            tools=tools,
            processing_class=processing_class,
            max_steps=max_steps,
        )
        
        self.dataset_name = dataset_name
    
    def get_dataset(self) -> Dataset:
        
        dataset = load_dataset(self.dataset_name)

        # select all three splits
        digits_1 = dataset["digits_1"]
        
        digits_2 = dataset["digits_2"]
        digits_3 = dataset["digits_3"]

        # concatenate the three splits
        dataset = concatenate_datasets([digits_1, digits_2, digits_3])
        
        # handle system prompt injection
        dataset = self.inject_system_prompt(dataset)

        # handle image injection
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
            
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
            rewards = []
            for conversation in merged_completion_conversations:
                assistant_messages = self.get_assistant_messages(conversation)
                
                format_correct = [check_format(message) for message in assistant_messages]
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
                answer_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
                tool_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<tool>([\s\S]*?)<\/tool>$"

                answer_match = re.search(answer_regex, text, re.DOTALL)
                tool_match = re.search(tool_regex, text, re.DOTALL)

                if (answer_match is not None and len(answer_match.groups()) == 2) or \
                   (tool_match is not None and len(tool_match.groups()) == 2):
                    return 1.0
                return 0.0
            except Exception as e:
                print(f"Error in check_format: {e}")
                return 0.0
        
        def correctness_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            """Reward function that checks if the predicted answer matches the true answer. Only checks the last message in the completion."""

            tasks = kwargs["task"]
            if not (task == tasks[0] for task in tasks):
                raise ValueError(
                    "All tasks must be same type, invalidates an assumption of the code"
                )

            task = tasks[0]
            
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)

            if task == "recognition":
                return _recognition_correctness_reward_func(completion_conversations=merged_completion_conversations, **kwargs)
            elif task == "addition":
                return _addition_correctness_reward_func(completion_conversations=merged_completion_conversations, **kwargs)
            else:
                raise ValueError(f"Invalid task: {task}")
        
        
        
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
                responses = [self.llm_parser.parse(c[-1]["content"][0]["text"]).answer for c in completion_conversations]

                def parse_list(r: str) -> list[int]:
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
        
        def _addition_correctness_reward_func(completion_conversations, **kwargs) -> list[float]:
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
            responses = [self.llm_parser.parse(c[-1]["content"][0]["text"]).answer for c in completion_conversations]

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
        
        def tool_execution_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            """
            Reward function that checks if tools were executed successfully.
            Returns a reward based on the ratio of successful tool executions to total attempts.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
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
                                if not response.startswith("Error:"):
                                    successful_executions += 1
                
                return 0.0 if tool_attempts == 0 else successful_executions / tool_attempts
            
            rewards = [check_execution(conv) for conv in merged_completion_conversations]
            return rewards
        
        return [format_reward_func, correctness_reward_func, tool_execution_reward_func]
    