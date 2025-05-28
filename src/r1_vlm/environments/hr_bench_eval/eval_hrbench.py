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
from r1_vlm.environments.hr_bench_eval.hr_bench_eval_env import HRBenchToolEnv

from tqdm import tqdm

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
    env = HRBenchToolEnv(processing_class=processor, split=args.split)
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

    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            results = [json.loads(line) for line in f]
    else:
        results = []

    for batch in tqdm(batches):
        for example in batch:
            if example["index"] in [result["index"] for result in results]:
                batch.remove(example)
                print(f"Skipping example {example['index']} because it already exists")

        if len(batch) == 0:
            continue

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
            correct_answers = example["answer"]
            result = {
                "index": example["index"],
                "gt_answers": correct_answers,
                "model_answer": model_answer,
                "correct": correct_answers == model_answer.upper(),
                "category": example["category"],
            }
            results.append(result)
            with open(args.output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--split", type=str, default="4k")
    parser.add_argument("--limit_image_per_prompt", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="hrbench_results.jsonl")
    args = parser.parse_args()

    results = generate_completions(args)