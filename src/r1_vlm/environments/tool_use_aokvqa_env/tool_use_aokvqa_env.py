from typing import Any, Callable

from datasets import Dataset, concatenate_datasets
from transformers import AutoProcessor

from r1_vlm.datasets.aok_vqa.aok_vqa_mc_tool_use_r1 import (
    create_r1_aok_vqa_mc_tool_use_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.environments.tool_vision_env import ToolVisionEnv
from r1_vlm.tools.object_detection import detect_objects
from r1_vlm.tools.zoom import zoom
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser


class AOKVQAToolEnv(ToolVisionEnv):
    
    def __init__(self,
                 processing_class: AutoProcessor,
                 dataset_name: str = "Groundlight/real-iad-toy-brick-tool-use-r1",
                 tools: list[Callable] = [zoom, detect_objects],
                 max_steps: int = 8,
                 ):
        
        super().__init__(
            processing_class=processing_class,
            tools=tools,
            max_steps=max_steps,
        )
        
        self.dataset_name = dataset_name    
        self.parser = XMLParser(fields=["think", "answer"])
        self._fields = [("think", ["think"]), ("answer", ["answer"])]

    
    
    
    def get_dataset(self) -> tuple[Dataset, Dataset, Dataset]:
        dataset = create_r1_aok_vqa_mc_tool_use_dataset()

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
        '''
        Returns the assistant messages from the completion messages as a list of strings.
        '''
        assistant_messages = [message["content"][0]["text"] for message in conversation if message["role"] == "assistant"]
        return assistant_messages
    
    
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
            
            preprocessed_data = AOKVQAToolEnv.preprocess_for_reward(
                prompts=prompts,
                completions=completions,
                completions_messages=completions_messages,
            )

            merged_completion_conversations = preprocessed_data["merged_completion_conversations"]
            
            return [check_format(m) for m in merged_completion_conversations]
        
        return [format_reward_func]


