import json
import os

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from r1_vlm.datasets.text_vqa.text_vqa_base_for_eval import (
    create_text_vqa_base_for_eval_dataset,
)
from r1_vlm.datasets.utils import preprocess_r1_dataset
from r1_vlm.environments.tool_use_text_vqa_env.tool_use_text_vqa_env import (
    normalize_answer,
)


def evalulate(results: list[dict]):
    total_score = 0
    total_count = 0
    for result in results:
        total_count += 1

        gt_answers = result["gt_answers"]
        generated_text = result["generated_text"]
        normalized_model_answer, normalized_correct_answers = normalize_answer(
            model_answer=generated_text, correct_answers=gt_answers
        )

        # VQA score
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

        total_score += vqa_score

    return total_score / total_count


if __name__ == "__main__":
    MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 1, "video": 0},
        tensor_parallel_size=2,
        gpu_memory_utilization=1.0,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        # max_tokens=10,
        stop_token_ids=[],
    )

    dataset = create_text_vqa_base_for_eval_dataset(
        splits_to_process=["validation"], max_examples_per_split=1000
    )
    dataset = preprocess_r1_dataset(dataset)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    results = []
    if os.path.exists("results.jsonl"):
        with open("results.jsonl", "r") as f:
            results = [json.loads(line) for line in f]

    for example in tqdm(dataset["validation"]):
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
        results.append(result)

        current_score = evalulate(results)
        print(f"Current score: {current_score}")

    with open("results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
