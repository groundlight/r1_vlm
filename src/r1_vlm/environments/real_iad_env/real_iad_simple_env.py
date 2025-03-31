import re
from typing import Any, List

from datasets import Dataset, load_dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
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
        
        # resizes images from 1024x1024 to 400x400
        def resize_images(example):
            image = example["image"]
            example["image"] = image.resize((400, 400)) 
            return example
            
        dataset = dataset.map(resize_images)
        
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
        
    @staticmethod
    def check_answer_format(text: str) -> float:
        '''
        Given a response, check if the answer is formatted correctly.
        
        Valid formats:
        1. <answer><label>ok</label></answer>
        2. <answer><label>{defect}</label><box>[x1,y1,x2,y2]</box></answer>
        '''
        try:
            # Extract answer content first
            answer_pattern = r'<answer>([\s\S]*?)</answer>'
            answer_match = re.search(answer_pattern, text)
            if not answer_match:
                return 0.0
            
            answer_content = answer_match.group(1)
            
            # Pattern for "ok" case: just a label tag with "ok"
            ok_pattern = r'^[\s\S]*<label>[\s\n]*ok[\s\n]*</label>[\s\S]*$'
            
            # Pattern for defect case: label tag with content followed by box tag with coordinates
            defect_pattern = r'^[\s\S]*<label>([^<]+)</label>[\s\S]*<box>\s*\[[\s\d\.,]+\]\s*</box>[\s\S]*$'
            
            if re.match(ok_pattern, answer_content) or re.match(defect_pattern, answer_content):
                return 1.0
            
            return 0.0
        
        except Exception:
            return 0.0
        
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def classification_reward_func(prompts, completions, completions_messages, **kwargs):
            '''
            Provides a reward if the model's classification is correct.
            '''
            pass
        
        def bounding_box_reward_func(prompts, completions, completions_messages, **kwargs):
            '''
            Provides a reward based on the model's proposed bounding box.
            
            If the GT is None, the model gets full reward IFF no bounding box is provided.
            
            If the GT is not None, the model gets reward equal to the IOU between the proposed and ground truth bounding boxes.
            '''
            pass
        
        
        def format_reward_func(prompts, completions, completions_messages, **kwargs):
            '''
            Provides a reward if the model's output is formatted correctly - a <think> </think> section and a <answer> </answer> section.
            '''
            
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            texts = []
            
            for completion_message in merged_completion_conversations:
                text = completion_message[0]['content'][0]['text']
                texts.append(text)
            
            rewards = [RealIADSimpleEnv.check_format(text) for text in texts]
            
            
            return rewards
            
        
        def answer_format_reward_func(prompts, completions, completions_messages, **kwargs):
            '''
            The answer format is non trivial for this task. We give a reward if the model's answer is formatted exactly correct.
            '''
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
            texts = []
            for completion_message in merged_completion_conversations:
                text = completion_message[0]['content'][0]['text']
                texts.append(text)
                
            rewards = [RealIADSimpleEnv.check_answer_format(text) for text in texts]
   
            return rewards
        
        
        return [format_reward_func, answer_format_reward_func]
        #return [classification_reward_func, bounding_box_reward_func, format_reward_func, answer_format_reward_func]
            
            
        
    
    
    
    