import re
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

        def correctness_reward_func(completions, **kwargs) -> list[float]:
            """Reward function that checks if the predicted answer matches the true answer. Only checks the last message in the completion."""
            # parse the predicted decoded message from each completion
            responses = [self.llm_parser.parse(c[0]["content"]).answer for c in completions]
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
        
        return [format_reward_func, correctness_reward_func]
