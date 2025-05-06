from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from r1_vlm.datasets.text_vqa.text_vqa_base_for_eval import (
    create_text_vqa_base_for_eval_dataset,
)

print("Loading model...")
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 1, "video": 0},
)

sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=10,
    stop_token_ids=[],
)

dataset = create_text_vqa_base_for_eval_dataset(splits_to_process=["validation"])

processor = AutoProcessor.from_pretrained(MODEL_PATH)


for example in dataset["validation"]:
    messages = example["messages"]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(generated_text)
