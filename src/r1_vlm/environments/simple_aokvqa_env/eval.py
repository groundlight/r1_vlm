from unittest.mock import patch

from vllm import LLM, SamplingParams

from r1_vlm.environments.simple_aokvqa_env.simple_aokvqa_env import AOKVQASimpleEnv
from r1_vlm.environments.simple_aokvqa_env.simple_aokvqa_train import (
    load_model_and_processor,
)


def evaluate():
    model, _, processor, _, _ = load_model_and_processor(model_name_or_path="/millcreek/home/sunil/r1_vlm/vlm-r1-simple-aokvqa-env/checkpoint-1800")
    model.eval()
    vf_env = AOKVQASimpleEnv(processing_class=processor)
    train_dataset, val_dataset, test_dataset = vf_env.get_dataset()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
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
    batches  =[]

    for example in val_dataset:
        if len(batches) == 0:
            batches.append([example])
        elif len(batches[-1]) < batch_size:
            batches[-1].append(example)
        else:
            batches.append([example])
    
    for batch in batches:
        conversations, texts, processed_batch, vllm_inputs = vf_env.prepare_data(
        inputs=batch, processing_class=processor
    )
        
    completion_ids = vf_env.generate(
        conversations=conversations,
        vlm_inputs=vllm_inputs,
        vlm=vlm,
        sampling_params=sampling_params,
    )
    print(completion_ids)
    # decode the ids to text
    # generated_texts = processor.batch_decode(
    #     completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    # )    


if __name__ == "__main__":
    evaluate()