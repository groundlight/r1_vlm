import json
import os
import re

from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from r1_vlm.environments.tool_use_text_vqa_env.tool_use_text_vqa_env import (
    TextVQAToolEnv,
    normalize_answer,
)

# evaluating the trained model on the validation set.


def extract_answer(generation: str):
    """Extracts the text between the first <answer> and </answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", generation, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def evaluate_result(result: dict):
    gt_answers = result["gt_answers"]
    model_answer = result["model_answer"]

    if model_answer is None:
        result["vqa_score"] = 0
        return result

    normalized_model_answer, normalized_correct_answers = normalize_answer(
        model_answer=model_answer, correct_answers=gt_answers
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


if __name__ == "__main__":
    MODEL_PATH = "/data/r1_vlm_checkpoints/vlm-r1-text-vqa-0_5-VQA-score-with-structured-output-small-tool-reward-may13-restart-with-structured-output-3B/checkpoint-500"
    results_file_path = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_text_vqa_env/eval_zoom_vqa_less_than_0_5_with_small_tool_reward_step650_validation_results_single_shot.jsonl"

    vlm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=1.0,
        dtype="bfloat16",
        tensor_parallel_size=4,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 2, "video": 0},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=2048,
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH, padding_side="left")
    env = TextVQAToolEnv(processing_class=processor)
    datasets = env.get_dataset(splits=["validation"])
    dataset = datasets["validation"]

    batch_size = 8

    results = []
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as f:
            results = [json.loads(line) for line in f]

    question_ids_to_skip = [result["question_id"] for result in results]

    batches = []
    for example in dataset:
        if example["question_id"] in question_ids_to_skip:
            continue

        if len(batches) == 0:
            batches.append([example])
        elif len(batches[-1]) < batch_size:
            batches[-1].append(example)
        else:
            batches.append([example])

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
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        for example, generation in zip(batch, generated_texts):
            model_answer = extract_answer(generation)
            correct_answers = example["answers"]

            result = {
                "question_id": example["question_id"],
                "gt_answers": example["answers"],
                "model_answer": model_answer,
            }
            result = evaluate_result(result)

            with open(results_file_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            results.append(result)

        current_score = evaluate(results)
        print(f"Current score: {current_score}")
