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

    system_prompt = "REPLACED WITH TOOLS SYSTEM PROMPT"

    choices_str = "Possible answers: "
    for i, choice in enumerate(choices):
        choices_str += f"{i + 1}. {choice}\n"

    instruction = f"""  
    While thinking, please do the following:
    1. Identify relevant visual evidence from the image to help answer the question. Tools are available to help you with this task if needed.
    2. Recall relevant non-visual evidence from your knowledge of the world to help answer the question.
    3. Combine these datapoints, and consider the merits of all 4 possible answers.
    4. Select the best answer from the 4 choices.
    
    Question: {question}

    {choices_str}
    
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
        "multiple_choice_answer": multiple_choice_answer,
    }


def create_r1_aok_vqa_tool_use_dataset(max_examples_per_split=None):
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
