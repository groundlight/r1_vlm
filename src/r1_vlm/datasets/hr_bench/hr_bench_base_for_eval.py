import io
import string
import base64

import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Any
from datasets import load_dataset, Dataset

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER
from r1_vlm.datasets.text_vqa.text_vqa_tool_use_r1 import resize_image

def b64_decode_image(image_b64: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    return image

def build_prompt_and_image(line):

    image = b64_decode_image(line['image'])

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

    msgs = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                }
            ],
        }
    ]
    msgs.append({
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PLACEHOLDER},
            {"type": "text", "text": prompt},
        ],
    })

    return msgs, image


def generate_simple_hr_bench_messages(example: dict, max_size: int | None = None) -> dict[str, Any]:
    msgs, image = build_prompt_and_image(example)
    # resize the image's width to 1024
    if max_size is not None:
        image = resize_image(image, max_size=max_size)

    return {
        "messages": msgs,
        "image": image,
        "index": example["index"],
        "category": example["category"],
        "answer": example["answer"],
    }



def create_r1_hr_bench_simple_dataset(split: str, max_size: int | None = None) -> Dataset:
    if split not in ["4k", "8k"]:
        raise ValueError(f"Invalid split: {split}. Must be either 4k or 8k.")
    dataset = load_dataset("DreamMr/HR-Bench", split=f"hrbench_{split}")
    processed_datasets = []
    for example in tqdm(dataset, desc="Processing HR-bench dataset"):
        processed_datasets.append(generate_simple_hr_bench_messages(example, max_size))

    return Dataset.from_list(processed_datasets)

