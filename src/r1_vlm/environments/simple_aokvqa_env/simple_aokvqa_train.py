import os
from trl import ModelConfig
from trl import GRPOConfig

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from r1_vlm.environments.aokvqa_env.simple_aokvqa_env import AOKVQASimpleEnv
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "simple-aokvqa-env"


def load_model_and_processor(
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    gradient_checkpointing: bool = True,
):
    model_config = ModelConfig(
        model_name_or_path=model_name_or_path,
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

    return model, processor, model_config, gradient_checkpointing


def train():
    model, processor, model_config, gradient_checkpointing = load_model_and_processor(
        gradient_checkpointing=True
    )

    vf_env = AOKVQASimpleEnv(processing_class=processor)

    train_dataset, val_dataset, test_dataset = vf_env.get_dataset()

    rubric = vf_env.get_rubric()

    training_args = GRPOConfig(
        model_init_kwargs=model_config,
        # save path on the runpod instance
        output_dir="/workspace/r1_vlm/vlm-r1-simple-aokvqa-env",
        learning_rate=1e-6,
        adam_beta2=0.98,
        lr_scheduler_type="cosine",
        warmup_steps=0,
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        num_generations=12,
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
        vllm_device="cuda:6",
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


if __name__ == "__main__":
    train()
