import os

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER

load_dotenv(dotenv_path=find_dotenv())

# generates the R1 messages for the digit recognition task

def generate_r1_messages(example, task):
    label = example["label"]
    image = example["image"]
    
    total = sum(label)
    
    if task == 'recognition':
        instruction = "What digits are in this image? I need the digits in a list sorted from lowest to highest."
        ending = "You are very bad at determining this directly from the image. Instead, please use the tools available to you to solve this problem."
        
        instruction = f"{instruction} {ending}"

        r1_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "REPLACED WITH TOOLS SYSTEM PROMPT",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<image_name> input_image </image_name>"},
                    {"type": "image", "image": IMAGE_PLACEHOLDER},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<think> Let me solve this step by step.\n"}
                ],
            },
        ]
        
        return {
            "messages": r1_messages,
            "task": task,
            "label": label,
            "total": total,
            "image": image,
        }
    elif task == 'addition':
        instruction = "What is the sum of the digits in this image?"
        ending = "You are very bad at determining this directly from the image. Instead, please use the tools available to you to solve this problem."
 
        instruction = f"{instruction} {ending}"
        
        
        r1_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "REPLACED WITH TOOLS SYSTEM PROMPT",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<image_name> input_image </image_name>"},
                    {"type": "image", "image": IMAGE_PLACEHOLDER},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<think> Let me solve this step by step.\n"}
                ],
            },
        ]
        
        
        return {
            "messages": r1_messages,
            "task": task,
            "label": label,
            "total": total,
            "image": image,
        }
        
    else:
        raise ValueError(f"Task {task} not supported")
        
    


def create_r1_counting_mnist_dataset():
    dataset = load_dataset("sunildkumar/digit-recognition")
    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            for task in ['recognition', 'addition']:
                processed_example = generate_r1_messages(example, task)
                examples.append(processed_example)
                
        processed_datasets[split] = Dataset.from_list(examples)
        
    return DatasetDict(processed_datasets)

        
if __name__ == "__main__":
    dataset = create_r1_counting_mnist_dataset()
    dataset.push_to_hub(
        "sunildkumar/digit-recognition-tool-use-r1", token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
