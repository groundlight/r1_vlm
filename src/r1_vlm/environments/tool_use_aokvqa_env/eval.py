from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from r1_vlm.environments.tool_use_aokvqa_env.tool_use_aokvqa_env import AOKVQAToolEnv


def main():
    checkpoint = "/millcreek/home/sunil/r1_vlm/vlm-r1-tool-use-aokvqa-env-better-tool-format/checkpoint-600"

    vlm = LLM(
        model=checkpoint,
        gpu_memory_utilization=1.0,
        dtype="bfloat16",
        tensor_parallel_size=4,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 2, "video": 0},
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
    )

    processor = AutoProcessor.from_pretrained(checkpoint, padding_side="left")
    vf_env = AOKVQAToolEnv(processing_class=processor)

    train_dataset, val_dataset, test_dataset = vf_env.get_dataset()

    batch_size = 5
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
                "generation": generation,
            }
            generations.append(data)


if __name__ == "__main__":
    main()
