from typing import Callable

from datasets import Dataset, load_dataset
from transformers import AutoProcessor
from verifiers.parsers import XMLParser

from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.tool_vision_env import ToolVisionEnv
from r1_vlm.tools.zoom import zoom_in


class RealIadToolEnv(ToolVisionEnv):
    '''
    This env tests the ability of the model to solve the toy brick real iad task using tools.
    '''
    
    def __init__(self,
                 processing_class: AutoProcessor,
                 dataset_name: str = "Groundlight/real-iad-toy-brick-tool-use-r1",
                 tools: list[Callable] = [zoom_in],
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
        self.image_size = (400, 400)
        
        
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
        
    
    
        
        