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

    system_prompt = "REPLACED WITH TOOLS SYSTEM PROMPT"

    instruction = f"""  
    You are answering a visual-question-answering task from the A-OKVQA dataset.

    After thinking, when you are ready to provide your answer, output your answer in the following format:
    1. all lowercase ASCII
    2. omit “a”, “an”, “the” unless part of a proper name
    3. yes/no questions → exactly “yes” or “no”
    4. Generally, the answer is a short unambiguous noun phrase (≤ 3 words)
    
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
        "question_id": question_id,
        "question": question,
        "choices": choices,
        "difficult_direct_answer": difficult_direct_answer,
        "correct_choice_idx": correct_choice_idx,
        "direct_answers": direct_answers,
        "rationales": rationales,
    }


def create_r1_aok_vqa_da_tool_use_dataset(max_examples_per_split=None):
    dataset = load_dataset("HuggingFaceM4/A-OKVQA")

    processed_datasets = {}
    for split in dataset.keys():
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            # skip examples that have a difficult direct answer for the direct answer task as we aren't evaluated on them
            if example["difficult_direct_answer"]:
                continue

            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

        processed_datasets[split] = Dataset.from_list(examples)

    return DatasetDict(processed_datasets)
