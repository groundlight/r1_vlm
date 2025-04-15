import os

from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

from r1_vlm.environments.real_iad_env.real_iad_simple_env import (
    RealIADSimpleEnv,
)

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "real-iad-simple-env-post-sft"

# Flag that determines if gradient checkpointing is used. If it is, we need to set use_cache to False.
gradient_checkpointing = True

checkpoint_path = "/millcreek/home/sunil/r1_vlm/src/r1_vlm/environments/real_iad_env/sft/output/v0-20250413-202349/checkpoint-100"
model_config = ModelConfig(
    model_name_or_path=checkpoint_path,
    torch_dtype="bfloat16",
    use_peft=False,
)

# this monkey patches the Qwen2.5-VL model to use the Liger Kernel on model init
apply_liger_kernel_to_qwen2_5_vl()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_config.model_name_or_path,
    torch_dtype=model_config.torch_dtype,
    use_cache=False,
)

# use cache if not gradient checkpointing
if gradient_checkpointing:
    model.config.use_cache = False
elif not gradient_checkpointing:
    model.config.use_cache = True
else:
    raise ValueError("Invalid gradient checkpointing value")


processor = AutoProcessor.from_pretrained(
    model_config.model_name_or_path, padding_side="left"
)

vf_env = RealIADSimpleEnv(
    processing_class=processor,
    use_budget_forcing=False,
    image_size=(400, 400),)
train_dataset, test_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()


training_args = GRPOConfig(
    model_init_kwargs=model_config,
    # save path on the runpod instance
    output_dir="vlm-r1-real-iad-simple-env-post-sft",
    learning_rate=1e-6,
    adam_beta2=0.98,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=1,
    save_steps=50,
    save_total_limit=100,
    num_train_epochs=10,
    per_device_train_batch_size=7,
    num_generations=21,
    gradient_accumulation_steps=4,
    gradient_checkpointing=gradient_checkpointing,
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=None,  # must be None for vllm + verifiers
    max_completion_length=4096,
    beta=0.001,
    temperature=1.0,
    sync_ref_model=True,
    ref_model_sync_steps=64,
    eval_strategy="no",
    log_completions=True,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.9,
    report_to="wandb",
    vllm_device="cuda:3",
)


trainer = QwenGRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=rubric,
    args=training_args,
    train_dataset=train_dataset,
    env=vf_env,
)

trainer.train()

# CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file src/r1_vlm/deepspeed_configs/multi_gpu_3only.yaml src/r1_vlm/environments/real_iad_env/simple_env_post_sft_train.py
