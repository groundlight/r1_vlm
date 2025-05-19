from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from tqdm import tqdm

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER


def generate_r1_messages(example):
    # unpack the example

    image: Image.Image = example["image"]
    question_id = example["question_id"]
    question = example["question"]
    choices = example["choices"]
    difficult_direct_answer = example["difficult_direct_answer"]

    # fields on train, val but not test
    correct_choice_idx = example.get("correct_choice_idx", None)
    direct_answers = example.get("direct_answers", None)
    rationales = example.get("rationales", None)

    image_size = image.size

    # denormalize the multiple choice answer if it exists for ease of use down the line
    multiple_choice_answer = (
        choices[correct_choice_idx] if correct_choice_idx is not None else None
    )

    multiple_choice_answer_letter = (
        ["A", "B", "C", "D"][correct_choice_idx]
        if correct_choice_idx is not None
        else None
    )

    system_prompt = "REPLACED WITH TOOLS SYSTEM PROMPT"

    choices_str = ""
    for choice, letter in zip(choices, ["A", "B", "C", "D"]):
        choices_str += f"{letter}. {choice}\n"

    instruction = f"""  
    The image size is {image_size}.
    Please thoroughly think through the question and refine your answer while thinking. You should try to collect the visual evidence you need to support your answer. Then, provide your answer.  The answer (which you will provide in the <answer> </answer> tags) should be the letter corresponding to the correct answer.

    Question: {question}
    {choices_str}
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
                    "text": "\n<think> ",
                }
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
        "dataset_name": "aok_vqa",
    }


def create_r1_aok_vqa_tool_use_dataset(
    max_examples_per_split=None,
    train_examples_to_include: list[str] | None = None,
    splits_to_process: list[str] = None,
):
    dataset = load_dataset("HuggingFaceM4/A-OKVQA")

    processed_datasets = {}
    for split in splits_to_process:
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            if (
                split == "train"
                and train_examples_to_include is not None
                and example["question_id"] not in train_examples_to_include
            ):
                continue

            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

            if max_examples_per_split is not None:
                if len(examples) >= max_examples_per_split:
                    break

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    dataset = create_r1_aok_vqa_tool_use_dataset(
        splits_to_process=["train"],
        max_examples_per_split=10,
    )
    print(dataset["train"][0])
