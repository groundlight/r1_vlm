import os

# Make sure to import necessary types for Features
from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
)
from tqdm import tqdm

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER

# generates the R1 messages for the toy brick task

def generate_r1_messages(example):
    image = example["image"]
    anomaly_class = example["anomaly_class"]
    bounding_box = example["bounding_box"]
    label = example["label"]
    
    system_prompt = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer. Show your work in <think> </think> tags and return the answer in <answer> </answer> tags."
    
    instruction = """
    This is an image of a wooden block. This block might have one of the following defects:
    1. missing parts
    2. pit
    3. scratch
    4. contamination
    
    Please classify the block into one of these categories: [missing parts, pit, scratch, contamination, ok].

    If the block is not ok, please determine the smallest bounding box that contains the entire defect. Express this bounding box as [x_min, y_min, x_max, y_max].
    
    How to express your answer (include <label> and <box> tags as appropriate):
    1. If the block is ok, <answer> <label> ok </label> </answer>.
    2. If the block has a defect, <answer> <label> <defect_type_here> </label> <box> [x_min, y_min, x_max, y_max] </box> </answer>.
    """

    r1_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PLACEHOLDER}, 
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "\n<think> Let me solve this step by step."}
            ],
        },
    ]

    return {
        "messages": r1_messages,
        "image": image,
        "anomaly_class": anomaly_class,
        "bounding_box": bounding_box,
        "label": label,
    }


def create_r1_toy_brick_dataset(max_examples_per_split=None):
    print("Loading original dataset...")
    dataset = load_dataset(
        "Groundlight/real-iad-toy-brick", 
        token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    print("Original dataset loaded.")

    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        
        # Get the dataset for this split
        split_dataset = dataset[split]
        
        # If max_examples is set, limit the dataset
        if max_examples_per_split is not None:
            split_dataset = split_dataset.select(range(min(max_examples_per_split, len(split_dataset))))
        
        # Process each example
        for example in tqdm(split_dataset, desc=f"Processing {split} examples"):
            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

        processed_datasets[split] = Dataset.from_list(examples)
        print(f"Finished processing {split} split.")

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    # Set this to an integer (e.g., 10) to limit examples per split for testing.
    # Set to None to process all examples.
    MAX_EXAMPLES_PER_SPLIT = None


    print(f"Processing dataset. Max examples per split: {MAX_EXAMPLES_PER_SPLIT or 'All'}")
    dataset = create_r1_toy_brick_dataset(max_examples_per_split=MAX_EXAMPLES_PER_SPLIT)


    print("\nUploading dataset to Hugging Face Hub (this may take several minutes)...")
    dataset.push_to_hub(
        "Groundlight/real-iad-toy-brick-r1",
        private=True,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
    print("Upload complete!")
