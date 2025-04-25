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

    system_prompt = "You are a helpful assistant. You first think and reason about the question, then provide the user with the answer from the list of possible answers. Show your work in <think> </think> tags and return the answer in <answer> </answer> tags. Please consider each option carefully and completely while reasoning before selecting the correct answer. You may change your mind while reasoning, and circle back to previous options if they are more correct."

    choices_str = "Possible answers: "
    for i, choice in enumerate(choices):
        if i == len(choices) - 1:
            choices_str += f"or {choice}."
        else:
            choices_str += f"{choice}, "

    instruction = f"""
    {question}

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
                {"type": "image", "image": IMAGE_PLACEHOLDER},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "\n<think>"}
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
    }


def create_r1_aok_vqa_mc_dataset(max_examples_per_split=None):
    dataset = load_dataset("HuggingFaceM4/A-OKVQA")

    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)


