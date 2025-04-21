import json
import os
import re
from unittest.mock import patch

from imgcat import imgcat
from tqdm import tqdm
from vllm import LLM, SamplingParams

from r1_vlm.environments.simple_aokvqa_env.simple_aokvqa_env import AOKVQASimpleEnv
from r1_vlm.environments.simple_aokvqa_env.simple_aokvqa_train import (
    load_model_and_processor,
)


def extract_answer(generation: str):
    """Extracts the text between the first <answer> and </answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", generation, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def evaluate(model_name_or_path:str):
   
    model, _, processor, _, _ = load_model_and_processor(model_name_or_path=model_name_or_path)
    model.eval()
    vf_env = AOKVQASimpleEnv(processing_class=processor)
    train_dataset, val_dataset, test_dataset = vf_env.get_dataset()
    
    if not os.path.exists("generations.json"):
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=1024,
        )

        world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
        profiling_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
            return_value=None,
        )
        with world_size_patch, profiling_patch:
            vlm = LLM(
                model=model.name_or_path,
                device="cuda:0",
                gpu_memory_utilization=1.0,
                dtype="bfloat16",
                enable_prefix_caching=True,
                limit_mm_per_prompt={"image": 1, "video": 0},
            )
            
        batch_size = 2
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
            inputs=batch, processing_class=processor)
            
            completion_ids = vf_env.generate(
                conversations=conversations,
                vlm_inputs=vllm_inputs,
                vlm=vlm,
                sampling_params=sampling_params,
            )
            
            generated_texts = processor.batch_decode(
                completion_ids["ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            
            for example, generation in zip(batch, generated_texts):
                data = {
                    "question_id": example["question_id"],
                    "generation": generation,
                }
                generations.append(data)
                    

        # Save the generations list as a JSON array to a file
        with open("generations.json", "w") as f:
            json.dump(generations, f, indent=2) # Use indent for readability (optional)
    
    else:
        print("Loading generations from file")
        with open("generations.json", "r") as f:
            generations = json.load(f)
            
    # get the answer from each generation 
    for generation in generations:
        generation["answer"] = extract_answer(generation["generation"])

    # convert to a dictionary for fast lookup
    generations_dict ={}
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
        
        model_answer = generations_dict[question_id]["answer"]
        gt_answer  = example["multiple_choice_answer"]
        
        options_set = example["choices"]
        
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
        if model_answer in options_set:
            in_option_set += 1
            
        total += 1
        
    print(f"Accuracy: {correct / total}")
    print(f"In option set: {in_option_set / total}")
        
    
    
    



if __name__ == "__main__":
    evaluate(model_name_or_path="/millcreek/home/sunil/r1_vlm/vlm-r1-simple-aokvqa-env-cliphigh-ssr-soft-format-reward/checkpoint-2600")