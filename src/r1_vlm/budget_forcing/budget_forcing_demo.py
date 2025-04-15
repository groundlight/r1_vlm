
from unittest.mock import patch

from datasets import Dataset
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import ModelConfig
from vllm import LLM

from r1_vlm.budget_forcing.budget_forcing import (
    generate_completions_with_budget_forcing,
)
from r1_vlm.environments.real_iad_env.real_iad_simple_env import RealIADSimpleEnv


def setup_model_and_processor() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    '''
    Returns the model and processor.
    '''
    # this monkey patches the Qwen2.5-VL model to use the Liger Kernel on init. 
    apply_liger_kernel_to_qwen2_5_vl()

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    # model_name = "/millcreek/home/sunil/r1_vlm/vlm-r1-real-iad-simple-env/checkpoint-80"
    
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_config = ModelConfig(
        model_name_or_path=model_name,
        torch_dtype="bfloat16",
        use_peft=False,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_config.model_name_or_path,
        torch_dtype=model_config.torch_dtype,
        use_cache=False,
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="left"
    )
    
    return model, processor


def setup_env(processor: AutoProcessor) -> RealIADSimpleEnv:
    vf_env = RealIADSimpleEnv(processing_class=processor)
    return vf_env

def dataset_to_batches(dataset: Dataset, batch_size: int) -> list[list[dict]]:
    '''
    Convert a dataset to a list of batches of examples.
    ''' 
    batches = []
    
    for example in dataset:
        if len(batches) == 0:
            batches.append([example])
        elif len(batches[-1]) < batch_size:
            batches[-1].append(example)
        else:
            batches.append([example])
        
    return batches

def setup_vllm(model: Qwen2_5_VLForConditionalGeneration):
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None)
    
    with world_size_patch, profiling_patch:
        vlm = LLM(
            model=model.name_or_path,
            device="cuda:0",
            gpu_memory_utilization=1.0,
            dtype="bfloat16",
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )
    
    return vlm

if __name__ == "__main__":
    model, processor = setup_model_and_processor()
    env = setup_env(processor)
    _, test_dataset = env.get_dataset()
    
    # convert dataset to batches of examples
    batches = dataset_to_batches(test_dataset, 4)
    
    vlm = setup_vllm(model=model)
    
    for batch in tqdm(batches):
        conversations, texts, processed_batch, vllm_inputs = env.prepare_data(
            inputs=batch, processing_class=processor
        )
        
        completions = generate_completions_with_budget_forcing(vllm_inputs=vllm_inputs, vlm=vlm, processor=processor, num_ignore=5)
