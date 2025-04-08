# The Real IAD dataset is not public, so we cannot share it. However, you may email the authors for access. See instructions here: https://huggingface.co/datasets/Real-IAD/Real-IAD.
import os
import random

import cv2
import numpy as np
from datasets import Dataset
from PIL import Image

# I'm not sure where these abbreviations come from (maybe foreign language?), but you can find it here: https://realiad4ad.github.io/Real-IAD/.
classes_mapping = {
    "OK": "ok",
    "AK": "pit",
    "BX": "deformation",
    "CH": "abrasion",
    "HS": "scratch",
    "PS": "damage",
    "QS": "missing parts",
    "YW": "foreign objects",
    "ZW": "contamination",
}

# The toy brick subset only contains these anomaly classes.
valid_anomaly_classes = ["AK", "HS", "QS", "ZW"]


def validate_input_dataset(real_iad_path):
    """
    Validates that the appropriate data is present in the Real IAD dataset.
    """

    assert os.path.exists(real_iad_path), "Real IAD dataset not found."

    toy_brick_path = os.path.join(real_iad_path, "toy_brick")
    assert os.path.exists(toy_brick_path), "Toy brick subset not found."

    # there should be 2 splits, OK and NG (Not Good)
    assert os.path.exists(os.path.join(toy_brick_path, "OK")), "OK split not found."
    assert os.path.exists(os.path.join(toy_brick_path, "NG")), "NG split not found."

    # Within the NG split, the data is split by anomaly class. We expect the following anomaly classes for the toy brick subset.
    for anomaly_class in valid_anomaly_classes:
        assert os.path.exists(os.path.join(toy_brick_path, "NG", anomaly_class)), (
            f"{anomaly_class} split not found."
        )


def collect_data_for_class(*, real_iad_path, dataset_class, classes_mapping):
    """
    For a given class in the dataset, returns a list of dicts:
        {
            "image_path": str,
            "anomaly_class": str,
            "bounding_box": list[int] | None # [x1, y1, x2, y2], normalized to [0, 1] if anomaly is present otherwise None
        }

    Args:
        real_iad_path (str): Path to the Real IAD dataset.
        dataset_class (str): The class to collect data for.
        classes_mapping (dict): A mapping of the dataset class to a human-readable label.
    Returns:
        list[dict]: A list of dicts containing the image paths, anomaly classes and bounding boxes.
    """

    examples = []

    # construct the path to where examples are stored
    if dataset_class == "OK":
        examples_path = os.path.join(real_iad_path, "toy_brick", "OK")
    else:
        # verify that the dataset class exists
        assert dataset_class in classes_mapping, (
            f"{dataset_class} not found in classes mapping."
        )
        examples_path = os.path.join(real_iad_path, "toy_brick", "NG", dataset_class)

    # each example is a folder
    example_paths = os.listdir(examples_path)

    for example_path in example_paths:
        # Join with the base path to get the full path
        full_example_path = os.path.join(examples_path, example_path)
        examples.extend(
            create_examples(example_path=full_example_path, dataset_class=dataset_class)
        )

    return examples


def create_examples(*, example_path, dataset_class):
    """
    Returns a list of dicts:
        {
            "image_path": str,
            "anomaly_class": str,
            "bounding_box": list[int] | None # [x1, y1, x2, y2], normalized to [0, 1] if anomaly is present otherwise None
        }
    """
    # get all the images in the example folder
    files = os.listdir(example_path)
    # separate images and masks
    images = [f for f in files if f.endswith(".jpg")]
    masks = [f for f in files if f.endswith(".png")]

    # create pairs of images and their corresponding masks
    image_mask_pairs = []
    for image in images:
        if dataset_class == "OK":
            # No masks for OK examples
            image_mask_pairs.append((image, None))
            continue

        # replace .jpg with .png to find corresponding mask
        mask = image.replace(".jpg", ".png")
        if mask in masks:
            image_mask_pairs.append((image, mask))

    # if the dataset class is OK, all views are valid
    if dataset_class == "OK":
        result = []
        for image_path, _ in image_mask_pairs:
            result.append(
                {
                    "image_path": os.path.join(example_path, image_path),
                    "anomaly_class": dataset_class,
                    "bounding_box": None,
                }
            )
        return result

    else:
        result = []
        for image_path, mask_path in image_mask_pairs:
            # Create full paths
            full_mask_path = os.path.join(example_path, mask_path)
            is_valid_view_result = is_valid_view(mask_path=full_mask_path)
            if is_valid_view_result["is_valid"]:
                result.append(
                    {
                        "image_path": os.path.join(example_path, image_path),
                        "anomaly_class": dataset_class,
                        "bounding_box": is_valid_view_result["bounding_box"],
                    }
                )
        return result

    # otherwise, we need to find the valid views and their bounding boxes. A view is valid if the mask if non empty


def is_valid_view(*, mask_path):
    """
    Returns if a view of an example with an anomaly is valid. If so, it also returns the smallest bounding box that contains the anomaly in normalized coordinates.

    Returns:
        dict: {
            "is_valid": bool,
            "bounding_box": list[int] | None # [x1, y1, x2, y2], normalized to [0, 1] if anomaly is present otherwise None
        }
    """
    # load the mask
    mask = cv2.imread(mask_path)

    # Convert the 2d mask
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a binary mask
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

    mask_sum = np.sum(binary_mask)
    # anomaly is not present
    if mask_sum == 0:
        return {"is_valid": False, "bounding_box": None}

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # If no contours are found, return invalid
    if not contours:
        return {"is_valid": False, "bounding_box": None}

    # Find the bounding box that contains all contours
    x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Normalize the bounding box coordinates
    height, width = binary_mask.shape
    bounding_box = [x_min / width, y_min / height, x_max / width, y_max / height]

    return {"is_valid": True, "bounding_box": bounding_box}


def save_to_hf(*, examples):

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Group examples by class
    examples_by_class = {}
    for example in examples:
        anomaly_class = example["anomaly_class"]
        if anomaly_class not in examples_by_class:
            examples_by_class[anomaly_class] = []
        examples_by_class[anomaly_class].append(example)
    
    # Create train/test splits for each class
    train_examples = []
    test_examples = []
    
    for class_name, class_examples in examples_by_class.items():
        # Shuffle examples
        random.shuffle(class_examples)
        
        # Calculate split index (80% train)
        split_idx = int(len(class_examples) * 0.8)
        
        # Split the examples
        train_examples.extend(class_examples[:split_idx])
        test_examples.extend(class_examples[split_idx:])
        
        print(f"Class {class_name}: {len(class_examples)} total, "
              f"{len(class_examples[:split_idx])} train, "
              f"{len(class_examples[split_idx:])} test")
    
    # Function to convert examples to dataset format
    def create_dataset_dict(examples_list):
        data = {
            "image": [],
            "anomaly_class": [],
            "bounding_box": [],
            "label": []
        }
        
        # Process examples in batches to avoid too many open files
        BATCH_SIZE = 100 
        
        for i in range(0, len(examples_list), BATCH_SIZE):
            batch = examples_list[i:i + BATCH_SIZE]
            
            for example in batch:
                try:
                    with Image.open(example["image_path"]) as img:
                        # Convert to RGB to ensure consistency
                        image = img.convert('RGB')
                        data["image"].append(image)
                        
                        anomaly_class = example["anomaly_class"]
                        data["anomaly_class"].append(anomaly_class)
                        data["label"].append(classes_mapping[anomaly_class])
                        data["bounding_box"].append(example["bounding_box"])
                except Exception as e:
                    print(f"Error processing {example['image_path']}: {e}")
                    continue
            
            print(f"Processed {min(i + BATCH_SIZE, len(examples_list))}/{len(examples_list)} examples")
            
        return Dataset.from_dict(data)
    
    # Create train and test datasets
    train_dataset = create_dataset_dict(train_examples)
    test_dataset = create_dataset_dict(test_examples)
    
    # Create DatasetDict with both splits
    from datasets import DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # saving as a private dataset to the Groundlight account (paid), which gets us the dataset viewer. Private datasets on free accounts don't get the viewer.
    dataset_dict.push_to_hub(
        "Groundlight/real-iad-toy-brick",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
        # Private as we don't have permission to distribute the image data. Other uses will have to download the data from the original source.
        private=True,
    )


if __name__ == "__main__":
    print("Starting toy brick dataset creation...")
    real_iad_path = "/millcreek/home/sunil/dvc-datasets/external-benchmarks/NON-COMMERCIAL-ONLY/Real-IAD"
    validate_input_dataset(real_iad_path)

    examples = []
    for dataset_class in ["OK"] + valid_anomaly_classes:
        class_examples = collect_data_for_class(
            real_iad_path=real_iad_path,
            dataset_class=dataset_class,
            classes_mapping=classes_mapping,
        )
        print(f"Collected {len(class_examples)} examples for class {dataset_class}")
        examples.extend(class_examples)

    save_to_hf(examples=examples)
