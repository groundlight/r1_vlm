import os
import re
import json
import argparse

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from datasets import Dataset, DatasetDict, load_dataset

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER
from r1_vlm.datasets.text_vqa.text_vqa_base_for_eval import (
    create_text_vqa_base_for_eval_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.tool_use_text_vqa_env.tool_use_text_vqa_env import (
    normalize_answer,
)

# evaluating the base model on the train set.


def evaluate_result(result: dict):
    gt_answers = result["gt_answers"]
    generated_text = result["generated_text"]
    normalized_model_answer, normalized_correct_answers = normalize_answer(
        model_answer=generated_text, correct_answers=gt_answers
    )

    total_matches = sum(
        [
            1
            for answer in normalized_correct_answers
            if answer == normalized_model_answer
        ]
    )
    vqa_score = total_matches / 3.0
    vqa_score = min(1, vqa_score)

    result["vqa_score"] = vqa_score

    return result


def evaluate(results: list[dict]):
    total_score = 0
    total_count = 0
    for result in results:
        total_count += 1

        total_score += result["vqa_score"]

    return total_score / total_count

def resize_image(image, max_size=1024):
    # if the longer side is greater than 1024, resize it so the longer side is 1024
    if image.size[0] > max_size or image.size[1] > max_size:
        longer_side = max(image.size[0], image.size[1])
        image = image.resize(
            (
                int(max_size * image.size[0] / longer_side),
                int(max_size * image.size[1] / longer_side),
            )
        )

    return image

def process_textcot_stage1_example(example):
    # unpack the example
    image_id = example["image_id"]
    question_id = example["question_id"]
    question = example["question"]
    # NOTE: We resize the image here if it is too large
    image = resize_image(example["image"], max_size=1024)
    answers = example["answers"]

    instruction = f"Describe the scene in the image in one sentence."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMAGE_PLACEHOLDER,
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]

    return {
        "messages": messages,
        "image": image,
        "raw_image": example["image"],
        "image_id": image_id,
        "question_id": question_id,
        "question": question,
        "answers": answers,
    }

def process_textcot_stage2_example(example):
    # unpack the example
    image_id = example["image_id"]
    question_id = example["question_id"]
    question = example["question"]
    # NOTE: We resize the image here if it is too large
    image = resize_image(example["image"], max_size=1024)
    answers = example["answers"]

    instruction = f"{question}\nThe image size is {image.size}. According to the information in the image and the question, \ndetail the bounding box coordinates of the answer in the scene in JSON format: [x1, y1, x2, y2]. All coordinates should be integers representing the pixel values of the bounding box. "

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMAGE_PLACEHOLDER,
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]

    return {
        "messages": messages,
        "image": image,
        "raw_image": example["image"],
        "image_id": image_id,
        "question_id": question_id,
        "question": question,
        "answers": answers,
    }


def create_text_vqa_textcot_base_for_eval_dataset(
    splits_to_process: list[str] | None = None,
    max_examples_per_split: int | None = None,
    stage: int = 1,
    idx_to_start: int = 0,
):
    dataset = load_dataset("lmms-lab/textvqa")
    process_fn = process_textcot_stage1_example if stage == 1 else process_textcot_stage2_example

    valid_splits = ["train", "validation", "test"]
    if splits_to_process is None:
        splits_to_process = valid_splits
    else:
        for split in splits_to_process:
            if split not in valid_splits:
                raise ValueError(f"Invalid split: {split}")

    processed_datasets = {}
    for split in splits_to_process:
        print(f"Processing {split} split...")
        examples = []
        selected_dataset = dataset[split].select(range(idx_to_start, len(dataset[split]))) if idx_to_start > 0 else dataset[split]
        for example in tqdm(selected_dataset, desc=f"Processing {split} examples"):
            processed_example = process_fn(example)
            examples.append(processed_example)

            if (
                max_examples_per_split is not None
                and len(examples) >= max_examples_per_split
            ):
                break

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


def process_textcot_stage3_example(example, result1, result2):
    # unpack the example
    context = result1['generated_text']
    bounding_box = result2['bounding_box']
    instruction = f"This is the context of the scene: {context}\nUse the image and text information as context and answer the following question:\n{example['question']}\nAnswer the question using a single word or phrase."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMAGE_PLACEHOLDER,
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]
    # zoom in to the bounding box
    zoomed_in_image = example["image"]
    # check if the raw image is downsized: if so, we apply the zoom to the raw image
    if example["raw_image"].size != example["image"].size:
        scale = example["raw_image"].size[0] / example["image"].size[0]
        scaled_bounding_box = [int(coord * scale) for coord in bounding_box]
        zoomed_in_image = example["raw_image"].crop(scaled_bounding_box)
    else:
        zoomed_in_image = example["image"].crop(bounding_box)

    example["messages"] = messages
    example["image"] = zoomed_in_image
    return example

def create_text_vqa_textcot_stage3_base_for_eval_dataset(
    dataset: Dataset,
    prev_stage1_results: list[dict],
    prev_stage2_results: list[dict],
) -> Dataset:
    if len(dataset) != len(prev_stage1_results):
        raise ValueError(f"The length of the dataset and the previous stage results must be the same. \nDataset length: {len(dataset)}\nPrevious stage results length: {len(prev_stage1_results)}")

    if len(dataset) != len(prev_stage2_results):
        raise ValueError(f"The length of the dataset and the previous stage results must be the same. \nDataset length: {len(dataset)}\nPrevious stage results length: {len(prev_stage2_results)}")

    examples = []
    for example, result1, result2 in zip(dataset, prev_stage1_results, prev_stage2_results):
        bounding_box = result2["bounding_box"]
        if bounding_box is None:
            continue
        processed_example = process_textcot_stage3_example(example, result1, result2)
        examples.append(processed_example)

    return Dataset.from_list(examples)


def inference(llm, processor, sampling_params, dataset):
    results = []
    for example in tqdm(dataset):
        if example["question_id"] in [result["question_id"] for result in results]:
            continue

        messages = example["messages"]
        for message in messages:
            content = message["content"]
            message["content"] = [
                {k: v for k, v in item.items() if v is not None} for item in content
            ]

        text = processor.apply_chat_template(
            messages,
            continue_final_message=False,
            tokenize=False,
            add_generation_prompt=True,
        )

        vllm_inputs = []

        vllm_image_inputs, _ = process_vision_info(messages)
        mm_data = {"image": vllm_image_inputs}
        vllm_input = {"prompt": text, "multi_modal_data": mm_data}
        vllm_inputs.append(vllm_input)

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        result = {
            "question_id": example["question_id"],
            "gt_answers": example["answers"],
            "generated_text": generated_text,
        }
        # result = evaluate_result(result)
        results.append(result)

    return results

def save_jsonl(results, file_path):
    with open(file_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--results_dir_path", type=str, default="/millcreek/home/bowen/Projects/r1_vlm/results")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_examples_per_split", type=int, default=None)
    parser.add_argument("--do_base_eval", action="store_true")
    parser.add_argument("--idx_to_start", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.results_dir_path, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 1, "video": 0},
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=1.0,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=128,
        stop_token_ids=[],
    )

    stage_1_dataset = create_text_vqa_textcot_base_for_eval_dataset(splits_to_process=[args.split], stage=1, max_examples_per_split=args.max_examples_per_split, idx_to_start=args.idx_to_start)
    stage_2_dataset = create_text_vqa_textcot_base_for_eval_dataset(splits_to_process=[args.split], stage=2, max_examples_per_split=args.max_examples_per_split, idx_to_start=args.idx_to_start)


    # stage 1
    dataset = preprocess_r1_dataset(stage_1_dataset[args.split])
    stage_1_results = inference(llm, processor, sampling_params, dataset)
    save_jsonl(stage_1_results, os.path.join(args.results_dir_path, "stage_1_results.jsonl"))
    # stage 2
    dataset = preprocess_r1_dataset(stage_2_dataset[args.split])
    stage_2_results = inference(llm, processor, sampling_params, dataset)
    save_jsonl(stage_2_results, os.path.join(args.results_dir_path, "stage_2_results.jsonl"))
    # extract the bounding box from the generated text
    bounding_box_pattern = r"\[(\d+), (\d+), (\d+), (\d+)\]"
    for result in stage_2_results:
        generated_text = result["generated_text"]
        # extract the bounding box from the generated text
        bounding_box = re.search(bounding_box_pattern, generated_text)
        if bounding_box:
            bounding_box = [int(coord) for coord in bounding_box.groups()]
            result["bounding_box"] = bounding_box
        else:
            result["bounding_box"] = None
        
    # stage 3
    stage_3_dataset = create_text_vqa_textcot_stage3_base_for_eval_dataset(stage_1_dataset[args.split], stage_1_results, stage_2_results)
    dataset = preprocess_r1_dataset(stage_3_dataset)
    stage_3_results = inference(llm, processor, sampling_params, dataset)
    for result in stage_3_results:
        result = evaluate_result(result)

    print("Textcot avg score: ", evaluate(stage_3_results))
    save_jsonl(stage_3_results, os.path.join(args.results_dir_path, "stage_3_results.jsonl"))

    if args.do_base_eval:
        dataset = create_text_vqa_base_for_eval_dataset(splits_to_process=[args.split])
        dataset = preprocess_r1_dataset(dataset[args.split])
        results = inference(llm, processor, sampling_params, dataset)
        for result in results:
            result = evaluate_result(result)

        print("Base avg score: ", evaluate(results))
        save_jsonl(results, os.path.join(args.results_dir_path, "base_results.jsonl"))