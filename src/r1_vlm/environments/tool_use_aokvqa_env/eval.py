import json
import os
import re

from imgcat import imgcat
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from r1_vlm.environments.tool_use_aokvqa_env.tool_use_aokvqa_env import AOKVQAToolEnv


def extract_answer(generation: str):
    """Extracts the text between the first <answer> and </answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", generation, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def main():
    checkpoint = (
        "/millcreek/home/sunil/r1_vlm_bumbershoot2/r1_vlm/checkpoint-850-better-zoom"
    )
    processor = AutoProcessor.from_pretrained(checkpoint, padding_side="left")
    vf_env = AOKVQAToolEnv(processing_class=processor)
    train_dataset, val_dataset, test_dataset = vf_env.get_dataset()

    if not os.path.exists("generations.json"):
        vlm = LLM(
            model=checkpoint,
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

        for example in val_dataset:
            if len(batches) == 0:
                batches.append([example])
            elif len(batches[-1]) < batch_size:
                batches[-1].append(example)
            else:
                batches.append([example])

        generations = []
        for batch in tqdm(batches, desc="Generating completions"):
            conversations, texts, processed_batch, vllm_inputs = vf_env.prepare_data(
                inputs=batch, processing_class=processor
            )

            completion_ids = vf_env.generate(
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

            print(generated_texts)

            for example, generation in zip(batch, generated_texts):
                data = {
                    "question_id": example["question_id"],
                    "question": example["question"],
                    "options": example["choices"],
                    "rationales": example["rationales"],
                    "gt_answer": example["multiple_choice_answer"],
                    "generation": generation,
                    "model_answer": extract_answer(generation),
                }
                generations.append(data)

        # Save the generations list as a JSON array to a file
        with open("generations.json", "w") as f:
            json.dump(generations, f, indent=2)  # Use indent for readability (optional)

    else:
        with open("generations.json", "r") as f:
            generations = json.load(f)

    generations_dict = {}
    for generation in generations:
        if generation["question_id"] in generations_dict:
            raise ValueError(f"Duplicate question_id: {generation['question_id']}")
        generations_dict[generation["question_id"]] = generation

    total = 0
    correct = 0
    in_option_set = 0
    for example in val_dataset:
        question_id = example["question_id"]

        if question_id not in generations_dict:
            raise ValueError(f"Question_id not found in generations: {question_id}")

        model_answer = generations_dict[question_id]["model_answer"]
        gt_answer = example["multiple_choice_answer"]

        options_set = example["choices"]

        if model_answer in options_set:
            in_option_set += 1

        total += 1
        if model_answer == gt_answer:
            correct += 1

        else:
            print("--------------------------------")
            print("Incorrect answer:")
            print(f"Question: {example['question']}")
            print(f"Model answer: {model_answer}")
            print(f"GT answer: {gt_answer}")
            print(f"Options: {options_set}")
            print(f"Generation: {generations_dict[question_id]['generation']}")
            print(f"Reasoning: {example['rationales']}")
            imgcat(example["image"])
            print("--------------------------------")

    print(f"Accuracy: {correct / total}")
    print(f"In option set: {in_option_set / total}")


if __name__ == "__main__":
    main()
