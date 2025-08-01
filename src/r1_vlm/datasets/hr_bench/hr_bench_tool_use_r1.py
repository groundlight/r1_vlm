import os
import string
import pandas as pd
from tqdm import tqdm
from PIL import Image
from typing import Any
from datasets import load_dataset, Dataset

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER
from r1_vlm.datasets.text_vqa.text_vqa_tool_use_r1 import resize_image
from r1_vlm.datasets.hr_bench.hr_bench_base_for_eval import b64_decode_image

def build_tool_use_prompt_and_image(line, max_size: int | None = None):
    image = b64_decode_image(line['image'])
    # resize the image's width if necessary
    if max_size is not None:
        image = resize_image(image, max_size=max_size)
    image_size = image.size
    question = line['question']
    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    options_prompt = 'Options:\n'
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'
    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    prompt = ''
    if hint is not None:
        prompt += f'Hint: {hint}\n'
    prompt += f'Question: {question}\n'
    if len(options):
        prompt += options_prompt
        prompt += 'Please select the correct answer from the options above. \n'

    system_prompt = "REPLACED WITH TOOLS SYSTEM PROMPT"

    instruction = f"""
    The image size is {image_size}.
    Please thoroughly think through the question and refine your answer while thinking. You should try to collect the visual evidence you need to support your answer. Then, provide your answer. The answer (which you will provide in the <answer> </answer> tags) should be a single word or phrase directly answering the question.
    Question: {prompt}
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

    return r1_messages, image

def generate_hr_bench_tool_use_messages(example: dict, max_size: int | None = None) -> dict[str, Any]:
    msgs, image = build_tool_use_prompt_and_image(example, max_size)
    return {
        "messages": msgs,
        "image": image,
        "index": example["index"],
        "category": example["category"],
        "answer": example["answer"],
    }

def create_r1_hr_bench_tool_use_dataset(split: str, max_size: int | None = None) -> Dataset:
    if split not in ["4k", "8k"]:
        raise ValueError(f"Invalid split: {split}. Must be either 4k or 8k.")
    dataset = load_dataset("DreamMr/HR-Bench", split=f"hrbench_{split}")
    processed_datasets = []
    for example in tqdm(dataset, desc="Processing HR-bench dataset"):
        processed_datasets.append(generate_hr_bench_tool_use_messages(example, max_size))

    return Dataset.from_list(processed_datasets)



if __name__ == "__main__":
    dataset = create_r1_hr_bench_tool_use_dataset(
        split="4k",
        max_size=1024,
    )
    print(dataset[0])
