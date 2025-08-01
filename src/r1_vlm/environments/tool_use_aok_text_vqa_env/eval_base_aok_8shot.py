import json
import os
from copy import deepcopy

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from r1_vlm.datasets.aok_vqa.aok_vqa_base_for_eval import (
    create_aok_vqa_base_mc_for_eval_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset


def evaluate_result(result: dict):
    gt_answer = result["gt_answer"]
    generated_text = result["generated_text"]

    score = 1.0 if gt_answer == generated_text else 0.0

    result["score"] = score

    return result


def evaluate(results: list[dict]):
    total_score = 0
    total_count = 0
    for result in results:
        total_count += 1

        total_score += result["score"]

    return total_score / total_count


if __name__ == "__main__":
    MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
    results_file_path = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/tool_use_aok_text_vqa_env/eval_on_aok_train_results_8shot.jsonl"

    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 1, "video": 0},
        tensor_parallel_size=1,
        gpu_memory_utilization=1.0,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        # max_tokens=10,
        stop_token_ids=[],
    )

    dataset = create_aok_vqa_base_mc_for_eval_dataset(splits_to_process=["train"])
    dataset = preprocess_r1_dataset(dataset)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    results = []
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as f:
            results = [json.loads(line) for line in f]

    for example in tqdm(dataset["train"]):
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

        # do 8 shot inference
        for _ in range(8):
            vllm_inputs.append(deepcopy(vllm_input))

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            result = {
                "question_id": example["question_id"],
                "gt_answer": example["multiple_choice_answer_letter"],
                "generated_text": generated_text,
            }
            result = evaluate_result(result)
            results.append(result)
            with open(results_file_path, "a") as f:
                f.write(json.dumps(result) + "\n")

        current_score = evaluate(results)
        print(f"Current score: {current_score}")
