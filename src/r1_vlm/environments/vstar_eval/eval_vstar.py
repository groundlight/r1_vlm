import os
import json
import argparse
from PIL import Image
from typing import Any
from transformers import AutoProcessor
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from verifiers.parsers import XMLParser
from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER
from r1_vlm.environments.vstar_eval.vstar_eval_env import VStarToolEnv

from tqdm import tqdm

# evaluating a VLM on V* Bench: craigwu/vstar_bench on HF
# the evaluation is done with the multi-choice setting, where the answe could only be one of the options: A, B, C, or D

def generate_simple_vstar_messages(example: dict, benchmark_directory: str) -> dict[str, Any]:
    question = example["text"]
    image_path = os.path.join(benchmark_directory, example["image"])
    image = Image.open(image_path)
    # resize the image's width to 1024
    # image = image.resize((1024, int(image.height * 1024 / image.width)))

    r1_messages: list[dict[str, Any]] = [
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
                {"type": "image", "image": image},
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

def generate_completions(args: argparse.Namespace):
    vlm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": args.limit_image_per_prompt, "video": 0},
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    env = VStarToolEnv(processing_class=processor, benchmark_directory=args.benchmark_directory)
    dataset = env.get_dataset(max_size=args.max_size)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    batch_size = args.batch_size
    batches = []

    for example in dataset:
        if len(batches) == 0:
            batches.append([example])
        elif len(batches[-1]) < batch_size:
            batches[-1].append(example)
        else:
            batches.append([example])

    results = []
    for batch in tqdm(batches):
        conversations, texts, processed_batch, vllm_inputs = env.prepare_data(
            inputs=batch, processing_class=processor
        )

        completion_ids = env.generate(
            conversations=conversations,
            vlm_inputs=vllm_inputs,
            vlm=vlm,
            sampling_params=sampling_params,
        )

        generated_texts = processor.batch_decode(
            completion_ids["ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for example, model_answer in zip(batch, generated_texts):
            correct_answers = example["label"]

            result = {
                "question_id": example["question_id"],
                "gt_answers": correct_answers,
                "model_answer": model_answer,
                "correct": correct_answers == model_answer,
                "category": example["category"],
            }
            results.append(result)
            with open(args.output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--benchmark_directory", type=str, default="/millcreek/data/vstar_bench")
    parser.add_argument("--limit_image_per_prompt", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="vstar_results.jsonl")
    args = parser.parse_args()

    results = generate_completions(args)
    
    # parse the results
    parser = XMLParser(fields=["think", "answer"])
    parsed_answers = [parser.parse(result["model_answer"]).answer for result in results]
    parsed_answers = [''.join([x for x in v if x.isupper()]) for v in parsed_answers]
    gt_answers = [v['gt_answers'] for v in results]
    accuracy = len([1 for pred, gt in zip(parsed_answers, gt_answers) if pred == gt]) / len(parsed_answers)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # calculate relative_position and direct_attributes subset accuracy
    relative_position_correct = len([1 for pred, gt, v in zip(parsed_answers, gt_answers, results) if pred == gt and v['category'] == 'relative_position']) / len([v for v in results if v['category'] == 'relative_position'])
    print(f"Relative Position Accuracy: {relative_position_correct * 100:.2f}%")
    direct_attributes_correct = len([1 for pred, gt, v in zip(parsed_answers, gt_answers, results) if pred == gt and v['category'] == 'direct_attributes']) / len([v for v in results if v['category'] == 'direct_attributes'])
    print(f"Direct Attribute Accuracy: {direct_attributes_correct * 100:.2f}%")
