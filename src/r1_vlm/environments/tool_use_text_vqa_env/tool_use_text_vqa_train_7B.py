import os

import torch
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from peft import LoraConfig, TaskType
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig, ModelConfig
from trl.trainer.qwen_grpo_trainer import QwenGRPOTrainer

from r1_vlm.environments.tool_use_text_vqa_env.tool_use_text_vqa_env import (
    TextVQAToolEnv,
)

os.environ["WANDB_ENTITY"] = "groundlightai"
os.environ["WANDB_PROJECT"] = "tool-use-text-vqa-env"


def load_model_and_processor(
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    gradient_checkpointing: bool = True,
    use_peft: bool = False,
):
    model_config = ModelConfig(
        model_name_or_path=model_name_or_path,
        torch_dtype="bfloat16",
        use_peft=use_peft,
    )

    # this monkey patches the Qwen2.5-VL model to use the Liger Kernel on model init
    apply_liger_kernel_to_qwen2_5_vl()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_config.model_name_or_path,
        torch_dtype=model_config.torch_dtype,
        use_cache=False,
    )

    # the peft config is not applied here, as the trainer handles it
    if use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=find_target_linear_names(
                model,
                num_lora_modules=-1,
                lora_namespan_exclude=["lm_head", "embed_tokens"],
                verbose=False,
            ),
        )
    else:
        peft_config = None

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

    return model, peft_config, processor, model_config, gradient_checkpointing


def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True
):
    # source: https://github.com/2U1/Qwen2-VL-Finetune/blob/master/src/training/train.py
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def train():
    model, peft_config, processor, model_config, gradient_checkpointing = (
        load_model_and_processor(gradient_checkpointing=True, use_peft=False)
    )
    print("loaded model")

    vf_env = TextVQAToolEnv(processing_class=processor, max_steps=3)

    print("loaded env")

    # TODO: increase max examples per split
    datasets = vf_env.get_dataset(splits=["train"])
    train_dataset = datasets["train"]

    rubric = vf_env.get_rubric()

    reward_weights = vf_env.get_reward_weights()

    training_args = GRPOConfig(
        model_init_kwargs=model_config,
        # save path on the runpod instance
        output_dir="vlm-r1-text-vqa-guided-decoding-zoom-may6-7B",
        # increase learning rate for PEFT - 1e-4
        learning_rate=1e-4 if peft_config is not None else 1e-6,
        max_grad_norm=1.0,
        adam_beta2=0.98,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        save_total_limit=4,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_generations=6,
        gradient_accumulation_steps=4,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        # GRPO specific parameters
        max_prompt_length=None,  # must be None for vllm + verifiers
        max_completion_length=2048,
        # smaller KL regularization for PEFT than full finetuning
        beta=1e-5 if peft_config is not None else 0.0001,
        temperature=1.0,
        sync_ref_model=True,
        ref_model_sync_steps=64,
        eval_strategy="no",
        log_completions=True,
        use_vllm=True,
        vllm_gpu_memory_utilization=1.0,
        report_to="wandb",
        vllm_device="cuda:6",
        limit_image_per_prompt=2,
        # clipHigh strategy from DAPO paper
        epsilon_low=0.2,
        # TODO: reduce to 0.28?
        epsilon_high=0.35,
        # reward weights with schedules for some of the reward functions
        reward_weights=reward_weights,
    )

    trainer = QwenGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=rubric,
        args=training_args,
        train_dataset=train_dataset,
        env=vf_env,
        peft_config=peft_config,
        guided_regex=None,
    )

    trainer.train()


if __name__ == "__main__":
    train()

# CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --config_file src/r1_vlm/deepspeed_configs/multi_gpu_3only.yaml src/r1_vlm/environments/tool_use_text_vqa_env/tool_use_text_vqa_train.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 uv run accelerate launch --config_file src/r1_vlm/deepspeed_configs/multi_gpu_6only.yaml src/r1_vlm/environments/tool_use_text_vqa_env/tool_use_text_vqa_train_7B.py
