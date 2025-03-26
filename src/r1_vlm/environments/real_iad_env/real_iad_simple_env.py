import re
from typing import Any, List

from datasets import Dataset, load_dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.simple_vision_env import SimpleVisionEnv


class RealIADSimpleEnv(SimpleVisionEnv):
    '''
    Baseline environment for the Real IAD dataset (no tool use).
    '''
    def __init__(self, dataset: str = "Groundlight/real-iad-toy-brick-r1", system_prompt: str = "", **kwargs):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.dataset_name = dataset
        self.parser = XMLParser(fields=["think", "answer"])
        self.answer_parser = XMLParser(fields=["label", "box"])
        
    def get_dataset(self) -> Dataset:
        dataset = load_dataset(self.dataset_name)["train"]
        # handle image injection
        dataset = preprocess_r1_dataset(dataset)
        return dataset
    
    
    def _parse_answer(self, answer_string: str) -> dict[str, str | List[float] | None] | None:
        '''
        Attempts to parse the data between <answer> </answer> tags for this particular task. 
        
        Expected format:
        
        1. No anomaly
        <label> ok </label>
        
        2. Anomaly
        <label> [anomaly_class] </label> <box> [x1, y1, x2, y2] </box>
        
        Args:
            answer_string (str): The string to parse. It should include the data between <answer> </answer> tags but not the tags themselves.
            
        Returns: None if not possible to parse, otherwise returns a dictionary:
            {
                "label": str,
                "box": List[float] | None
            }
        '''
        
        try:
            label = self.answer_parser.parse(answer_string)["label"]
            box = self.answer_parser.parse(answer_string)["box"]
            
            if box:
                box = box.strip('[]').split(',')
                box = [float(x.strip()) for x in box]
            
            return {"label": label, "box": box}
            
        except:
            return None
    
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
        
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def classification_reward_func():
            '''
            Provides a reward if the model's classification is correct.
            '''
            pass
        
        def bounding_box_reward_func():
            '''
            Provides a reward based on the model's proposed bounding box.
            
            If the GT is None, the model gets full reward IFF no bounding box is provided.
            
            If the GT is not None, the model gets reward equal to the IOU between the proposed and ground truth bounding boxes.
            '''
            pass
        
        
        def format_reward_func():
            '''
            Provides a reward if the model's output is formatted correctly - a <think> </think> section and a <answer> </answer> section.
            '''
            
            
        
    
    
    
    