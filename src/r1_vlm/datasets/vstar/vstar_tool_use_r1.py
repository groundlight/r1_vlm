import os
from tqdm import tqdm
from PIL import Image
from typing import Any
from datasets import load_dataset, Dataset

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER
from r1_vlm.datasets.text_vqa.text_vqa_tool_use_r1 import resize_image

def generate_vstar_tool_use_messages(example: dict, benchmark_directory: str, max_size: int | None = None) -> dict[str, Any]:
    question = example["text"]
    image_path = os.path.join(benchmark_directory, example["image"])
    image = Image.open(image_path)
    # resize the image's width to 1024
    if max_size is not None:
        image = resize_image(image, max_size=max_size)
    image_size = image.size

    system_prompt = "REPLACED WITH TOOLS SYSTEM PROMPT"

    instruction = f"""
    The image size is {image_size}.
    Please thoroughly think through the question and refine your answer while thinking. You should try to collect the visual evidence you need to support your answer. Then, provide your answer. The answer (which you will provide in the <answer> </answer> tags) should be a single word or phrase directly answering the question.
    Question: {question}
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
                {"type": "text", "text": "<image_name> input_image </image_name>"},
                {"type": "image", "image": IMAGE_PLACEHOLDER},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n<think> Let me think step by step.",
                }
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

def create_r1_vstar_tool_use_dataset(benchmark_directory: str, max_size: int | None = None) -> Dataset:
    dataset = load_dataset("craigwu/vstar_bench", split="test")
    processed_datasets = []
    for example in tqdm(dataset, desc="Processing V*-bench dataset"):
        processed_datasets.append(generate_vstar_tool_use_messages(example, benchmark_directory, max_size))

    return Dataset.from_list(processed_datasets)



if __name__ == "__main__":
    dataset = create_r1_vstar_tool_use_dataset(
        benchmark_directory="/millcreek/data/vstar_bench/",
        max_size=1024,
    )
    print(dataset[0])
