import random
from copy import deepcopy
from unittest.mock import patch

from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vllm import LLM

from r1_vlm.budget_forcing.budget_forcing import (
    generate_completions_with_budget_forcing,
)
from r1_vlm.environments.real_iad_env.real_iad_simple_env import RealIADSimpleEnv
from trl import ModelConfig


def setup_model_and_processor(checkpoint_path: str) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    '''
    Returns the model and processor.
    '''
    # this monkey patches the Qwen2.5-VL model to use the Liger Kernel on init. 
    apply_liger_kernel_to_qwen2_5_vl()

    model_config = ModelConfig(
        model_name_or_path=checkpoint_path,
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
    vf_env = RealIADSimpleEnv(
    processing_class=processor,
    use_budget_forcing=True,
    image_size=(400, 400),
    max_thinking_tokens=1500,
    num_ignore=[1, 2],
    ignore_str=[
        # Focused on Self-Reflection/Re-evaluation:
        "Wait, let me review my previous steps carefully.",
        "Reflecting on my approach up to this point",
        "Let me re-evaluate my reasoning.",
        "I should double-check my work before proceeding.",
        "Analyzing my previous output for potential improvements, ",
        "Let me think step-by-step again.",
        "Wait, I should verify my understanding before moving forward.",

        # Focused on Considering Alternatives:
        "Hmm, is there a different way to approach this?",
        "Let's explore another possible solution.",
        "Let me think about other options or methods.",
        "What if I tried a different approach to this?",

        # Expressing Uncertainty/Need for More Thought:
        "Hold on, let me think this through more carefully.",
        "I need to reconsider my last thought process.",
        "Let me pause and ensure I haven't missed anything.",
        "Something doesn't seem quite right, let me re-examine.",
        "Before concluding, let me verify my logic."
],
)
    return vf_env


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
    model, processor = setup_model_and_processor(checkpoint_path="/millcreek/home/sunil/r1_vlm/vlm-r1-real-iad-simple-env-budget-forcing-longer-ignore-strings/checkpoint-100")
    env = setup_env(processor)

    reward_funcs_list = env.get_rubric()
    reward_funcs_dict = {
        func.__name__: func 
        for func in reward_funcs_list
    }
    
    # drop the functions we don't want to use
    reward_funcs_dict.pop("answer_format_reward_func")
    reward_funcs_dict.pop("format_reward_func")
    
    assert len(reward_funcs_dict) == 2, "We should only have two reward functions"
    
    train_dataset, _ = env.get_dataset()
    
    vlm = setup_vllm(model=model)
    
    for example in train_dataset:
        batch_size = 2
        
        # create a batch of the same example, deep copy the example
        example_batch = [deepcopy(example) for _ in range(batch_size)]
        
        conversations, texts, processed_batch, vllm_inputs = env.prepare_data(
            inputs=example_batch, processing_class=processor
        )
        
        num_ignore = random.choice(env.num_ignore)
        completions = generate_completions_with_budget_forcing(
                vllm_inputs=vllm_inputs,
                vlm=vlm,
                processor=processor,
                max_thinking_tokens=env.max_thinking_tokens,
                num_ignore=num_ignore,
                ignore_str=env.ignore_str,
                temperature=1.0,
                repetition_penalty=1.0,
            )
        completion_messages = []
        for completion in completions:
            text = completion.outputs[0].text
            messages = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                }
            ]
            completion_messages.append(messages)
        
        # maps the function name to a list of rewards, one for each example in the batch
        reward_dict = {}
        for reward_func_name, reward_func in reward_funcs_dict.items():
            # Create lists of labels and bounding boxes that match the batch size
            labels = [example["label"]] * batch_size
            bounding_boxes = [example["bounding_box"]] * batch_size
            
            reward = reward_func(conversations, texts, completion_messages, label=labels, bounding_box=bounding_boxes)
            reward_dict[reward_func_name] = reward
            
            
        # sum the reward for each example in the batch
        rewards = []
        for index in range(batch_size):
            reward = sum([reward_dict[function_name][index] for function_name in reward_funcs_dict.keys()])
            rewards.append(reward)
        
        print(f"Rewards: {rewards}")
        
    
