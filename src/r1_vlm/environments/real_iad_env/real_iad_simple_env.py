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
        
        # we need to class balance the dataset, there are way more ok samples than any other class, which causes the model to overfit to the ok class when training
        
        # Count samples per class
        label_counts = dataset.select_columns(['label']).to_pandas()['label'].value_counts()
    
        # Calculate average anomaly count
        anomaly_classes = ['missing parts', 'pit', 'scratch', 'contamination']
        anomaly_counts = [label_counts.get(cls, 0) for cls in anomaly_classes]
        avg_anomaly_count = sum(anomaly_counts) / len(anomaly_classes)
        
        
        # Filter ok samples to match average anomaly count
        ok_indices = [i for i, label in enumerate(dataset['label']) if label == 'ok']
        keep_ok_indices = ok_indices[:int(avg_anomaly_count)]
        non_ok_indices = [i for i, label in enumerate(dataset['label']) if label != 'ok']
        
        # Combine indices and filter dataset
        keep_indices = keep_ok_indices + non_ok_indices
        dataset = dataset.select(keep_indices)
        
        print(f"After balancing: {dataset.select_columns(['label']).to_pandas()['label'].value_counts()}")
        
        # Handle image injection and resizing
        dataset = preprocess_r1_dataset(dataset, image_size=(800, 800))
        return dataset
    
    
    def _parse_answer(self, completion_message: str) -> dict[str, str | List[float] | None] | None:
        '''
        Given a completion message, attempts to parse the data between <answer> </answer> tags and extract the label and box.
        
        Expected format:
        1. No anomaly
        <label> ok </label>
        
        2. Anomaly
        <label> [anomaly_class] </label> <box> [x1, y1, x2, y2] </box>
        
        Args:
            completion_message (str): The completion message to parse. It should include the data between <answer> </answer> tags but not the tags themselves.
            
        Returns: a dictionary with the following keys.
            {
                "label": str | None,
                "box": List[float] | None
            }
        '''
        try:
            # get the data within the <answer> </answer> tags
            answer_pattern = r'<answer>([\s\S]*?)</answer>'
            answer_match = re.search(answer_pattern, completion_message)
            
            # if no answer match, no label or box
            if not answer_match:
                return {"label": None, "box": None}
            
            answer_string = answer_match.group(1)

            label = self.answer_parser.parse(answer_string).label
            box = self.answer_parser.parse(answer_string).box
            
            return {"label": label, "box": box}
            
        except:
            return {"label": None, "box": None}
    
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

            answer_match = re.search(answer_regex, text, re.DOTALL)

            if (answer_match is not None and len(answer_match.groups()) == 2):
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
        
        
    @staticmethod
    def _box_reward_func_helper(proposed_box: list[float], true_box: list[float]) -> float:
        '''
        Helper function for the bounding box reward function. Score is the sum of a valid box reward and a IOU reward.
                
        Args:
            proposed_box (list[float]): The proposed bounding box represented as [x1, y1, x2, y2]
            true_box (list[float]): The ground truth bounding box represented as [x1, y1, x2, y2]
            
        Returns:
            float: The sum of a valid box reward and a IOU reward - total reward is between 0 and 1.
        '''
        
        # unpack the boxes. The proposed box should have 4 values in it. If it doesn't, no reward (to protect against unpacking errors.)
        if len(proposed_box) != 4:
            return 0.0
        
        proposed_box_x1, proposed_box_y1, proposed_box_x2, proposed_box_y2 = proposed_box
        
        if not isinstance(proposed_box_x1, float) or not isinstance(proposed_box_y1, float) or not isinstance(proposed_box_x2, float) or not isinstance(proposed_box_y2, float):
            return 0.0
        
        
        try:
            true_box_x1, true_box_y1, true_box_x2, true_box_y2 = true_box
        except Exception as e:
            raise ValueError(f"Invalid ground truth box: {true_box=}") from e
        
        # check that the gt box is valid, otherwise something is very wrong
        if not (true_box_x1 < true_box_x2 and true_box_y1 < true_box_y2) or not all(0<=x<=1 for x in true_box):
            raise ValueError(f"Invalid ground truth box: {true_box=}")
        
        # check that the proposed box is valid, if it isn't no reward
        if not (proposed_box_x1 < proposed_box_x2 and proposed_box_y1 < proposed_box_y2) or not all(0<=x<=1 for x in proposed_box):
            return 0.0
        
        # get a small reward for returning a valid box
        valid_box_reward = 0.1
        
        # the rest of the reward is based on the IOU
        # calculate intersection coordinates
        x_left = max(proposed_box_x1, true_box_x1)
        y_top = max(proposed_box_y1, true_box_y1)
        x_right = min(proposed_box_x2, true_box_x2)
        y_bottom = min(proposed_box_y2, true_box_y2)

        # calculate areas
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        proposed_box_area = (proposed_box_x2 - proposed_box_x1) * (proposed_box_y2 - proposed_box_y1)
        true_box_area = (true_box_x2 - true_box_x1) * (true_box_y2 - true_box_y1)
        union_area = proposed_box_area + true_box_area - intersection_area

        # calculate IOU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return valid_box_reward + 0.9 * iou  # Scale remaining 0.9 reward by IOU (so max reward is 1.0 rather than 1.1)
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        def classification_reward_func(prompts, completions, completions_messages, **kwargs):
            '''
            Provides a reward if the model's classification is correct.
            '''
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
            texts = []
            for completion_message in merged_completion_conversations:
                text = completion_message[0]['content'][0]['text']
                texts.append(text)
            
            # get the answers from the completion messages
            answers = [self._parse_answer(text) for text in texts]
            true_labels = kwargs["label"]
            
            rewards = []
            for answer, true_label in zip(answers, true_labels):
                if answer["label"] == true_label:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            
            return rewards


        
        
        def bounding_box_reward_func(prompts, completions, completions_messages, **kwargs):
            '''
            Provides a reward based on the model's proposed bounding box.
            
            If the GT is None, the model gets full reward IFF no bounding box is provided.
            
            If the GT is not None, the model gets reward equal to the IOU between the proposed and ground truth bounding boxes.
            '''
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
            
            texts = []
            for completion_message in merged_completion_conversations:
                text = completion_message[0]['content'][0]['text']
                texts.append(text)
            
            answers = [self._parse_answer(text) for text in texts]
            proposed_boxes: list[str] = [answer['box'] for answer in answers]
            
            # convert the proposed box from a string to a list of floats
            proposed_boxes_floats = [] 
            for box in proposed_boxes:
                if box is not None:
                    try:
                        box = eval(box, {}, {})
                    except:
                        box = None
                    proposed_boxes_floats.append(box)
                else:
                    proposed_boxes_floats.append(None)
                
                
            
            true_boxes = kwargs["bounding_box"]
            
            rewards = []
            for proposed_box, true_box in zip(proposed_boxes_floats, true_boxes):
                
                # handle case where there is no GT box
                if true_box is None:
                    if proposed_box is None:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                        
                # handle case where there is a GT box
                else:
                    if proposed_box is None:
                        rewards.append(0.0)
                    else:
                        try:
                            # compute the IOU between the proposed and true boxes
                            iou_reward = RealIADSimpleEnv._box_reward_func_helper(proposed_box, true_box)
                            rewards.append(iou_reward)
                        except Exception as e:
                            print(f"Error in bounding_box_reward_func: {e}")
                            rewards.append(0.0)
                        
            return rewards
        
        
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
        
        
        return [format_reward_func, answer_format_reward_func, classification_reward_func, bounding_box_reward_func]
   
            
            
        
    
    
    
    