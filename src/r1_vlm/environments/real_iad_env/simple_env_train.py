import os

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from r1_vlm.environments.real_iad_env.real_iad_simple_env import (
    RealIADSimpleEnv,
)
from trl import GRPOConfig, ModelConfig
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "real-iad-simple-env"

# Flag that determines if gradient checkpointing is used. If it is, we need to set use_cache to False.
gradient_checkpointing = False


model_config = ModelConfig(
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    use_peft=False,
)

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

vf_env = RealIADSimpleEnv(processing_class=processor)
train_dataset, test_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()




training_args = GRPOConfig(
    model_init_kwargs=model_config,
    output_dir="vlm-r1-real-iad-simple-env-correctness-over-format-weighted",
    learning_rate=1e-6,
    adam_beta2=0.98,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    logging_steps=1,
    save_steps=100,
    save_total_limit=50,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    num_generations=3,
    # give higher weight to the correctness rewards over the format rewards, to hopefully encourage the model to learn the task over learning to format. 
    #[format_reward_func, answer_format_reward_func, classification_reward_func, bounding_box_reward_func]
    reward_weights = [1.0, 1.0, 5.0, 5.0],
    gradient_accumulation_steps=4,
    gradient_checkpointing=gradient_checkpointing,
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=None,  # must be None for vllm + verifiers
    max_completion_length=1024,
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

#CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file src/r1_vlm/deepspeed_configs/multi_gpu_3only_zero3.yaml src/r1_vlm/environments/real_iad_env/simple_env_train.py

