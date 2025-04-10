from unittest.mock import patch

from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams

from r1_vlm.environments.real_iad_env.real_iad_simple_env import RealIADSimpleEnv
from trl import ModelConfig


def post_process_generated_text(generated_text: str) -> str:
    # check if it has a </think> token. If so, remove it and everything after it.
    if "</think>" in generated_text:
        return generated_text.split("</think>")[0]
    # otherwise, check if it has an <answer> token. If so, remove it and everything after it.
    elif "<answer>" in generated_text:
        return generated_text.split("<answer>")[0]
    else:
        return generated_text


# this monkey patches the Qwen2.5-VL model to use the Liger Kernel on init. 
apply_liger_kernel_to_qwen2_5_vl()

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

vf_env = RealIADSimpleEnv(processing_class=processor)

_, test_dataset = vf_env.get_dataset()


conversations, texts, processed_batch, vllm_inputs = vf_env.prepare_data(
    inputs=[test_dataset[0]], processing_class=processor
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
# how many total tokens are we willing to think for
MAX_THINKING_TOKENS = 1024
# How many times we're willing to ignore the model trying to end thinking.
NUM_IGNORE = 1

sampling_params = SamplingParams(
    temperature=1.0,
    min_tokens=0,
    max_tokens=MAX_THINKING_TOKENS,
    skip_special_tokens=False,
    stop=[
        "</think>",
        "<answer>",
        "<|im_start|>",
        "<|im_end|>",
    ],
    # don't include the stop string in the output as we want to prevent it from returning this token and instead guide it to think more.
    include_stop_str_in_output=False,
)
completion_ids = vlm.generate(
    vllm_inputs,
    sampling_params=sampling_params,
)

# Extract token IDs from the RequestOutput object
token_ids = [output.outputs[0].token_ids for output in completion_ids]

generated_texts = processor.batch_decode(
    token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)

for index in range(len(generated_texts)):
    generated_texts[index] = post_process_generated_text(generated_texts[index])


print("The model decided to stop thinking at this point:")
print(generated_texts)

# append the generated text to the conversation as part of the prompt
vllm_inputs[0]["prompt"] += generated_texts[0]

print("Added the generated text to the vllm_inputs:")
print(vllm_inputs)

# this is technically an overestimate as we truncate the generated text at the first </think> token, but it is close enough.
tokens_used = len(completion_ids[0].outputs[0].token_ids)
remaining_tokens = MAX_THINKING_TOKENS - tokens_used
ignore_str = "Wait, let me reconsider my thinking"
# how many times we've ignored the model trying to end thinking.
num_ignores = 0

while remaining_tokens > 0 and num_ignores < NUM_IGNORE:
    vllm_inputs[0]["prompt"] += ignore_str

    print("Added the ignore string to the vllm_inputs:")
    print(vllm_inputs)
    print(f"Remaining tokens: {remaining_tokens}")
    sampling_params = SamplingParams(
        max_tokens=remaining_tokens,
        min_tokens=100,
        stop=[
            "</think>",
            "<answer>",
            "<|im_start|>",
            "<|im_end|>",
        ],
        skip_special_tokens=False,
        temperature=1.0,
    )

    completion_ids = vlm.generate(
        vllm_inputs,
        sampling_params=sampling_params,
    )

    token_ids = completion_ids[0].outputs[0].token_ids
    print("The model has generated the following tokens:")
    print(token_ids)

    generated_texts = processor.batch_decode(
        [token_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    generated_texts[0] = post_process_generated_text(generated_texts[0])
    print("The model has generated the following text:")
    print(generated_texts[0])

    # add the generated text to the conversation as part of the prompt
    vllm_inputs[0]["prompt"] += generated_texts[0]

    print("Added the generated text to the vllm_inputs:")
    print(vllm_inputs)

    # adjust the remaining tokens
    tokens_used = len(token_ids)
    remaining_tokens -= tokens_used
    # increment the number of ignores
    num_ignores += 1
    print()
    print()

print("The model has finished thinking and generated the following text:")
print(vllm_inputs[0]["prompt"])

# now we append the end think and start answer tokens to the prompt
vllm_inputs[0]["prompt"] += "</think>\n<answer>"

sampling_params = SamplingParams(
    max_tokens=100,
    min_tokens=1,
    stop=[
        "<|im_end|>",
    ],
)

completion_ids = vlm.generate(
    vllm_inputs,
    sampling_params=sampling_params,
)

token_ids = completion_ids[0].outputs[0].token_ids

generated_texts = processor.batch_decode(
    [token_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False
)

generated_texts[0] = post_process_generated_text(generated_texts[0])

print("The model has generated the following text:")
print(generated_texts[0])

vllm_inputs[0]["prompt"] += generated_texts[0]

print("The model has finished answering and generated the following text:")
print(vllm_inputs[0]["prompt"])

import ipdb

ipdb.set_trace()
