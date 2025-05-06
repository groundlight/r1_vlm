from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from r1_vlm.datasets.text_vqa.text_vqa_tool_use_r1 import resize_image
from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER


def process_example(example):
    # unpack the example
    image_id = example["image_id"]
    question_id = example["question_id"]
    question = example["question"]
    # NOTE: We resize the image here if it is too large
    image = resize_image(example["image"])
    answers = example["answers"]

    instruction = f"{question}\nPlease try to answer the question with short words or phrases if possible."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": IMAGE_PLACEHOLDER,
                },
                {"type": "text", "text": instruction},
            ],
        }
    ]

    return {
        "messages": messages,
        "image": image,
        "image_id": image_id,
        "question_id": question_id,
        "question": question,
        "answers": answers,
    }


def create_text_vqa_base_for_eval_dataset(
    splits_to_process: list[str] | None = None,
    max_examples_per_split: int | None = None,
):
    dataset = load_dataset("lmms-lab/textvqa")

    valid_splits = ["train", "validation", "test"]
    if splits_to_process is None:
        splits_to_process = valid_splits
    else:
        for split in splits_to_process:
            if split not in valid_splits:
                raise ValueError(f"Invalid split: {split}")

    processed_datasets = {}
    for split in splits_to_process:
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            processed_example = process_example(example)
            examples.append(processed_example)

            if (
                max_examples_per_split is not None
                and len(examples) >= max_examples_per_split
            ):
                break

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    dataset = create_text_vqa_base_for_eval_dataset(splits_to_process=["train"])
    print(dataset["train"][0])
