import json
import os
import re
from copy import deepcopy

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from r1_vlm.environments.tool_use_text_vqa_env.tool_use_text_vqa_env import (
    TextVQAToolEnv,
    normalize_answer,
)


def extract_answer(generation: str):
    """Extracts the text between the first <answer> and </answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", generation, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def generate_completions(
    checkpoint_path: str, file_path: str, dataset: Dataset, env, processor
):
    """
    Generate completions given a checkpoint and a file path to save the generations
    """
    if os.path.exists(file_path):
        raise ValueError(f"File {file_path} already exists")

    vlm = LLM(
        model=checkpoint_path,
        gpu_memory_utilization=1.0,
        dtype="bfloat16",
        tensor_parallel_size=2,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 2, "video": 0},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
    )

    batch_size = 6
    batches = []

    for example in dataset:
        if len(batches) == 0:
            batches.append([example])
        elif len(batches[-1]) < batch_size:
            batches[-1].append(example)
        else:
            batches.append([example])

    generations = []
    for batch in tqdm(batches, desc="Generating completions"):
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
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        for example, generation in zip(batch, generated_texts):
            data = {
                "question_id": example["question_id"],
                "question": example["question"],
                "answers": example["answers"],
                "generation": generation,
                "model_answer": extract_answer(generation),
            }
            generations.append(data)

    with open(file_path, "w") as f:
        json.dump(generations, f, indent=2)


def evaluate(generations_dict: dict, dataset: Dataset):
    with open(generations_dict, "r") as f:
        generations = json.load(f)

    generations_dict = {}
    for generation in generations:
        if generation["question_id"] in generations_dict:
            raise ValueError(f"Duplicate question_id: {generation['question_id']}")
        generations_dict[generation["question_id"]] = generation

    total = 0
    correct = 0
    in_option_set = 0

    for example in dataset:
        question_id = example["question_id"]

        if question_id not in generations_dict:
            raise ValueError(f"Question_id not found in generations: {question_id}")

        model_answer = generations_dict[question_id]["model_answer"]
        correct_answers = example["answers"]

        # normalize the data according to the eval script
        model_answer, correct_answers = normalize_answer(model_answer, correct_answers)

        num_matches = sum(
            1 for correct_answer in correct_answers if correct_answer == model_answer
        )

        score = min(1, num_matches / 3)

        total += 1
        correct += score

    results = {
        "accuracy": correct / total,
    }

    print(f"Accuracy: {results['accuracy']}")

    return results


if __name__ == "__main__":
    checkpoints_folder = "/millcreek/home/sunil/r1_vlm/vlm-r1-text-vqa-tool-use-constant-tool-reward-may3-3B"

    checkpoint_paths = [
        os.path.join(checkpoints_folder, f)
        for f in os.listdir(checkpoints_folder)
        if os.path.isdir(os.path.join(checkpoints_folder, f))
    ]

    checkpoints_to_eval = ["400"]

    checkpoint_paths = [
        path
        for path in checkpoint_paths
        if any(num in path for num in checkpoints_to_eval)
    ]

    processor = AutoProcessor.from_pretrained(checkpoint_paths[0], padding_side="left")
    env = TextVQAToolEnv(processing_class=processor)
    datasets = env.get_dataset(splits=["validation"])
    val_dataset = datasets["validation"]

    results_dict = {}

    # we'll save evaluations to the same folder as the checkpoints
    for checkpoint_path in checkpoint_paths:
        file_path = os.path.join(
            checkpoints_folder, f"{checkpoint_path}_generations.json"
        )
        if not os.path.exists(file_path):
            generate_completions(
                checkpoint_path, file_path, deepcopy(val_dataset), env, processor
            )
        else:
            print(f"Skipping {checkpoint_path} because it already exists")

        results = evaluate(file_path, deepcopy(val_dataset))
        results_dict[checkpoint_path] = results

    print(results_dict)
