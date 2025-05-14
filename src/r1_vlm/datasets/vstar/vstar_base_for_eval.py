
import os

from PIL import Image
from tqdm import tqdm
from typing import Any
from datasets import load_dataset, Dataset

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER
from r1_vlm.datasets.text_vqa.text_vqa_tool_use_r1 import resize_image



def generate_simple_vstar_messages(example: dict, benchmark_directory: str, max_size: int | None = None) -> dict[str, Any]:
    question = example["text"]
    image_path = os.path.join(benchmark_directory, example["image"])
    image = Image.open(image_path)
    # resize the image's width to 1024
    if max_size is not None:
        image = resize_image(image, max_size=max_size)

    r1_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PLACEHOLDER},
                {"type": "text", "text": question},
            ],
        },
    ]

    return {
        "messages": r1_messages,
        "image": image,
        "category": example["category"],
        "label": example["label"],
        "question_id": example["question_id"],
    }



def create_r1_vstar_simple_dataset(benchmark_directory: str, max_size: int | None = None) -> Dataset:
    dataset = load_dataset("craigwu/vstar_bench", split="test")
    processed_datasets = []
    for example in tqdm(dataset, desc="Processing V*-bench dataset"):
        processed_datasets.append(generate_simple_vstar_messages(example, benchmark_directory, max_size))

    return Dataset.from_list(processed_datasets)

