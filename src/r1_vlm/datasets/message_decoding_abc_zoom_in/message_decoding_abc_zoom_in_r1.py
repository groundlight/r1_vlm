import os

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER
load_dotenv(dotenv_path=find_dotenv())


def generate_r1_messages(example):
    coded_message = example["coded_message"]
    decoded_message = example["decoded_message"]
    mapping = example["mapping"]
    image = example["image"]
    full_coordinates = example["full_coordinates"]
    file_path = example["file_path"]

    # add spaces between each character to prevent tokenization issues
    coded_message = " ".join(coded_message)

    instruction = (
        "Use the decoder in the image to decode a coded message. "
        "The decoded message will be one or more characters from A, B, and C. "
        "You are very bad at determining this directly from the image, because the mappings in the decoder are too small to see. "
        "Instead, please use the tools available to you to solve this problem. "
        "For each round of your response, show your work of reasoning in <think> </think> tags first, "
        "then either calls the tool with <tool> </tool> tags or returns the answer in <answer> </answer> tags. "
        "You have only one single chance to call the tool, and after calling the tool and getting the response, you must think first and then return the answer immediately. "
        "However, if your first attempt of calling the tool results in an error, you can try again. "
        "Within each round, you can think about the problem for as long as you'd like. While thinking, you should robustly verify your solution. "
        f"Once you are done thinking and calling the tool, provide your answer in the <answer> section, e.g. <answer> ABC </answer>. The coded message is: {coded_message}."
    )

    r1_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "REPLACED WITH TOOLS SYSTEM PROMPT",
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
                {"type": "text", "text": "<think> Let me solve this step by step.\n"}
            ],
        },
    ]

    return {
        "messages": r1_messages,
        "coded_message": coded_message,
        "decoded_message": decoded_message,
        "mapping": mapping,
        "file_path": file_path,
        "image": image,
        "full_coordinates": full_coordinates,
    }


def create_r1_message_decoding_dataset():
    dataset_dict = {}
    for split in ["train", "validation"]:
        dataset = load_dataset(
            "Groundlight/message-decoding-abc-zoom-in", split=split
        )

        examples = []
        for example in tqdm(dataset, desc="Processing examples"):
            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

        processed_dataset = Dataset.from_list(examples)

        dataset_dict[split] = processed_dataset


    r1_dataset = DatasetDict(dataset_dict)
    r1_dataset.push_to_hub(
        "Groundlight/message-decoding-abc-zoom-in-r1",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )


if __name__ == "__main__":
    create_r1_message_decoding_dataset()