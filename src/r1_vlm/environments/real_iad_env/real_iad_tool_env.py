import re
from statistics import mean
from typing import Any, Callable

from datasets import Dataset, load_dataset
from transformers import AutoProcessor
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.parsers import XMLParser

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.multistep_vision_env import MultistepVisionEnv
from r1_vlm.environments.tool_vision_env import ToolVisionEnv
from r1_vlm.tools.display_bounding_box import display_bounding_box
from r1_vlm.tools.zoom import zoom


class RealIadToolEnv(ToolVisionEnv):
    '''
    This env tests the ability of the model to solve the toy brick real iad task using tools.
    '''
    
    def __init__(self,
                 processing_class: AutoProcessor,
                 dataset_name: str = "Groundlight/real-iad-toy-brick-tool-use-r1",
                 tools: list[Callable] = [zoom, display_bounding_box],
                 max_steps: int = 3,
                 ):
        
        super().__init__(
            processing_class=processing_class,
            tools=tools,
            max_steps=max_steps,
        )
        
        self.dataset_name = dataset_name
        # we will resize the input images to this resolution prior to training
        # 1024x1024 is the native resolution of the images
        # TODO: increase this to 1024x1024 on more GPUs
        self.image_size = (300, 300)
        
        
        self.parser = XMLParser(fields=["think", "answer"])
        self.answer_parser = XMLParser(fields=["label", "box"])
        
        
    def get_dataset(self) -> tuple[Dataset, Dataset]:
        """
        Loads the dataset and preprocesses it.
        
        Training split is balanced to have an equal number of samples per anomaly class.
        
        Both splits are converted to pixel coordinates, images are resized to self.image_size, and injected into the conversation.
        
        Returns a tuple of the training and test datasets: (train_dataset, test_dataset)
        """
        dataset = load_dataset(self.dataset_name)

        # Get both train and test splits
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Only balance the training split
        # Count samples per class in training
        label_counts = (
            train_dataset.select_columns(["label"]).to_pandas()["label"].value_counts()
        )

        # Calculate average anomaly count
        anomaly_classes = ["missing parts", "pit", "scratch", "contamination"]
        anomaly_counts = [label_counts.get(cls, 0) for cls in anomaly_classes]
        avg_anomaly_count = sum(anomaly_counts) / len(anomaly_classes)

        # Filter ok samples to match average anomaly count in training only
        ok_indices = [
            i for i, label in enumerate(train_dataset["label"]) if label == "ok"
        ]
        keep_ok_indices = ok_indices[: int(avg_anomaly_count)]
        non_ok_indices = [
            i for i, label in enumerate(train_dataset["label"]) if label != "ok"
        ]

        # Combine indices and filter training dataset
        keep_indices = keep_ok_indices + non_ok_indices
        train_dataset = train_dataset.select(keep_indices)

        print(
            f"After balancing training split: {train_dataset.select_columns(['label']).to_pandas()['label'].value_counts()}"
        )

        # Convert normalized bounding boxes to pixel coordinates for both splits
        def convert_bbox(example):
            bbox = example["bounding_box"]
            if bbox is None:
                return {"bounding_box": None}

            width, height = self.image_size
            # Convert from normalized [0,1] to pixel coordinates and round to nearest int
            x1, y1, x2, y2 = bbox
            pixel_bbox = [
                round(x1 * width),  # x1
                round(y1 * height),  # y1
                round(x2 * width),  # x2
                round(y2 * height),  # y2
            ]
            return {"bounding_box": pixel_bbox}

        train_dataset = train_dataset.map(convert_bbox)
        test_dataset = test_dataset.map(convert_bbox)
        
        # handle system prompt injection
        train_dataset = self.inject_system_prompt(train_dataset)
        test_dataset = self.inject_system_prompt(test_dataset)

        # Handle image injection and resizing for both splits
        train_dataset = preprocess_r1_dataset(train_dataset, image_size=self.image_size)
        test_dataset = preprocess_r1_dataset(test_dataset, image_size=self.image_size)

        # Return a DatasetDict containing both splits
        return train_dataset, test_dataset
        
    def get_assistant_messages(self, conversation: list[dict[str, Any]]) -> list[str]:
        '''
        Returns the assistant messages from the completion messages as a list of strings.
        '''
        assistant_messages = [message["content"][0]["text"] for message in conversation if message["role"] == "assistant"]
        return assistant_messages
    
    
    def _parse_answer(
            self, completion_message: str
        ) -> dict[str, str | list[float] | None] | None:
            """
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
            """
            try:
                # get the data within the <answer> </answer> tags
                answer_pattern = r"<answer>([\s\S]*?)</answer>"
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
    
    @staticmethod
    def check_answer_format(text: str) -> float:
        """
        Given a response, check if the answer is formatted correctly.

        Valid formats:
        1. <answer><label>ok</label></answer>
        2. <answer><label>{defect}</label><box>[x1,y1,x2,y2]</box></answer>
        """
        try:
            # Extract answer content first
            answer_pattern = r"<answer>([\s\S]*?)</answer>"
            answer_match = re.search(answer_pattern, text)
            if not answer_match:
                return 0.0

            answer_content = answer_match.group(1)

            # Pattern for "ok" case: just a label tag with "ok"
            ok_pattern = r"^[\s\S]*<label>[\s\n]*ok[\s\n]*</label>[\s\S]*$"

            # Pattern for defect case: label tag with content followed by box tag with coordinates
            defect_pattern = r"^[\s\S]*<label>([^<]+)</label>[\s\S]*<box>\s*\[[\s\d\.,]+\]\s*</box>[\s\S]*$"

            if re.match(ok_pattern, answer_content) or re.match(
                defect_pattern, answer_content
            ):
                return 1.0

            return 0.0

        except Exception:
            return 0.0
            
    
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
        
        
        
        def classification_reward_func(prompts, completions, completions_messages, **kwargs) -> list[float]:
            '''
            Provides a reward if the model's classification is correct.
            '''
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(prompts_messages=prompts, completions_messages=completions_messages)
                
            # select the last message in each completion (completions are conversations) 
            texts = [c[-1]["content"][0]["text"] for c in merged_completion_conversations]
            
            answers = [self._parse_answer(text) for text in texts]
            
            true_labels = kwargs["label"]
            
            rewards = []
            
            for answer, true_label in zip(answers, true_labels):
                if answer["label"] == true_label:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            
            return rewards
        
        
        
        def bounding_box_reward_func(
            prompts, completions, completions_messages, **kwargs
        ):
            """
            Provides a reward based on the model's proposed bounding box.

            If the GT is None, the model gets full reward IFF no bounding box is provided.

            If the GT is not None, the model gets reward equal to the IOU between the proposed and ground truth bounding boxes.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
                prompts_messages=prompts, completions_messages=completions_messages
            )

            texts = [c[-1]["content"][0]["text"] for c in merged_completion_conversations]

            answers = [self._parse_answer(text) for text in texts]
            proposed_boxes: list[str] = [answer["box"] for answer in answers]

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
                            iou_reward = RealIadToolEnv._box_reward_func_helper(
                                proposed_box, true_box, self.image_size
                            )
                            rewards.append(iou_reward)
                        except Exception as e:
                            print(f"Error: {e} in bounding_box_reward_func")
                            rewards.append(0.0)

            return rewards

        
        def answer_format_reward_func(
            prompts, completions, completions_messages, **kwargs
        ):
            """
            The answer format is non trivial for this task. We give a reward if the model's answer is formatted exactly correct.
            """
            merged_completion_conversations = MultistepVisionEnv.preprocess_messages(
                prompts_messages=prompts, completions_messages=completions_messages
            )

            # select the last message in each completion (completions are conversations) 
            texts = [c[-1]["content"][0]["text"] for c in merged_completion_conversations]

            rewards = [RealIadToolEnv.check_answer_format(text) for text in texts]

            return rewards

        return [format_reward_func, answer_format_reward_func, tool_execution_reward_func, classification_reward_func, bounding_box_reward_func]