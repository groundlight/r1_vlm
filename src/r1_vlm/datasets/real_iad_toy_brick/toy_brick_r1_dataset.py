import os

from datasets import Dataset, DatasetDict, load_dataset
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
    
    If the block is not ok, please determine the smallest bounding box that contains the entire defect. Express this bounding box using normalized coordinates (as floats from 0 to 1) [x_min, y_min, x_max, y_max].
    
    How to express your answer (include <label> and <box> tags as appropriate):
    1. If the block is ok, <answer> <label> ok </label> </answer>. 
    2. If the block has a scratch, <answer> <label> scratch </label> <box> [x_min, y_min, x_max, y_max] </box> </answer>.
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
        }
    ]
    
    return {
        "messages": r1_messages,
        "image": image,
        "anomaly_class": anomaly_class,
        "bounding_box": bounding_box,
        "label": label,
    }
    
    
    

def create_r1_toy_brick_dataset():
    dataset = load_dataset("Groundlight/real-iad-toy-brick", token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            processed_example = generate_r1_messages(example)
            examples.append(processed_example)
            
        processed_datasets[split] = Dataset.from_list(examples)
        
    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    dataset = create_r1_toy_brick_dataset()
    
    # private as it includes the RealIAD image data, which we cannot distribute. However toy_brick_dataset.py explains how to get permission to access it.
    dataset.push_to_hub(
        "Groundlight/real-iad-toy-brick-r1",
        private=True,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )
   