from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER


def resize_image(image, max_size=1024):
    # if the longer side is greater than 1024, resize it so the longer side is 1024
    if image.size[0] > max_size or image.size[1] > max_size:
        longer_side = max(image.size[0], image.size[1])
        image = image.resize(
            (
                int(max_size * image.size[0] / longer_side),
                int(max_size * image.size[1] / longer_side),
            )
        )

    return image


def generate_r1_messages(example, max_size=1024):
    # unpack the example
    image_id = example["image_id"]
    question_id = example["question_id"]
    question = example["question"]
    # NOTE: We resize the image here if it is too large
    image = resize_image(example["image"], max_size=max_size)
    image_size = image.size
    answers = example["answers"]

    system_prompt = "REPLACED WITH TOOLS SYSTEM PROMPT"

    instruction = f"""
    You are answering a question from the TextVQA benchmark, where you need to read and reason about text in images to answer questions about them.

    Please thoroughly think through the question and refine your answer while thinking. Then, provide your answer. The answer (which you will provide in the <answer> </answer> tags) should be a single word or phrase directly answering the question.
    Question: {question}

    The image size is {image_size}.
    """

    r1_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<image_name> input_image </image_name>"},
                {"type": "image", "image": IMAGE_PLACEHOLDER},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n<think> Let me think step by step.",
                }
            ],
        },
    ]

    return {
        "messages": r1_messages,
        "image": image,
        "image_id": image_id,
        "question_id": question_id,
        "question": question,
        "answers": answers,
    }


def create_r1_text_vqa_tool_use_dataset(
    max_examples_per_split: int | None = None,
    splits_to_process: list[str] = None,
    max_size: int = 1024,
):
    """
    max_size - used to resize the input image to a smaller size to avoid OOM errors
    """

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
            processed_example = generate_r1_messages(example, max_size=max_size)
            examples.append(processed_example)

            if max_examples_per_split is not None:
                if len(examples) >= max_examples_per_split:
                    break

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    dataset = create_r1_text_vqa_tool_use_dataset(
        max_examples_per_split=10,
        splits_to_process=["train"],
        max_size=1024,
    )
    print(dataset["train"][0])
