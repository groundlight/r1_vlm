from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER


def generate_r1_messages(example):
    # unpack the example

    image = example["image"]
    question_id = example["question_id"]
    question = example["question"]
    choices = example["choices"]
    difficult_direct_answer = example["difficult_direct_answer"]

    # fields on train, val but not test
    correct_choice_idx = example.get("correct_choice_idx", None)
    direct_answers = example.get("direct_answers", None)
    rationales = example.get("rationales", None)

    # denormalize the multiple choice answer if it exists for ease of use down the line
    multiple_choice_answer = (
        choices[correct_choice_idx] if correct_choice_idx is not None else None
    )

    multiple_choice_answer_letter = (
        ["A", "B", "C", "D"][correct_choice_idx]
        if correct_choice_idx is not None
        else None
    )

    choices_str = ""
    for choice, letter in zip(choices, ["A", "B", "C", "D"]):
        choices_str += f"{letter}. {choice}\n"

    choices_str += "\nAnswer with the option's letter from the given choices directly. Do not provide an explanation or include any other text."

    instruction = f"""
    {question}
    {choices_str}
    """

    r1_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PLACEHOLDER},
                {"type": "text", "text": instruction},
            ],
        },
    ]

    return {
        "messages": r1_messages,
        "image": image,
        "question_id": question_id,
        "question": question,
        "choices": choices,
        "difficult_direct_answer": difficult_direct_answer,
        "correct_choice_idx": correct_choice_idx,
        "direct_answers": direct_answers,
        "rationales": rationales,
        "multiple_choice_answer": multiple_choice_answer,
        "multiple_choice_answer_letter": multiple_choice_answer_letter,
    }


def create_aok_vqa_base_mc_for_eval_dataset(
    splits_to_process: list[str] | None = None,
    max_examples_per_split: int | None = None,
):
    dataset = load_dataset("HuggingFaceM4/A-OKVQA")

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
            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

            if (
                max_examples_per_split is not None
                and len(examples) >= max_examples_per_split
            ):
                break

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    dataset = create_aok_vqa_base_mc_for_eval_dataset(
        splits_to_process=["train"], max_examples_per_split=100
    )
    print(dataset["train"][0])
