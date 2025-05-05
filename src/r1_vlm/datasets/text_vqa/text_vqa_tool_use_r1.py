import json

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from PIL import Image
from tqdm import tqdm

from r1_vlm.datasets.utils import IMAGE_PLACEHOLDER


def resize_image(image):
    # if the longer side is greater than 1024, resize it so the longer side is 1024
    if image.size[0] > 1024 or image.size[1] > 1024:
        longer_side = max(image.size[0], image.size[1])
        image = image.resize(
            (
                int(1024 * image.size[0] / longer_side),
                int(1024 * image.size[1] / longer_side),
            ),
            Image.Resampling.LANCZOS,
        )

    return image


def generate_r1_messages(example):
    # unpack the example
    image_id = example["image_id"]
    question_id = example["question_id"]
    question = example["question"]
    # NOTE: We resize the image here if it is too large
    image = resize_image(example["image"])
    image_size = image.size
    answers = example["answers"]

    system_prompt = "REPLACED WITH TOOLS SYSTEM PROMPT"

    instruction = f"""
    You are answering a question from the TextVQA benchmark, where you need to read and reason about text in images to answer questions about them.

    Please thoroughly think through the question and refine your answer while thinking. Then, provide your answer.
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
    zoom_demos_path: str
    | None = "/millcreek/home/sunil/r1_vlm_bumbershoot2/r1_vlm/src/r1_vlm/datasets/text_vqa/zoom_demos.jsonl",
):
    dataset = load_dataset("lmms-lab/textvqa")

    valid_splits = ["train", "validation", "test"]
    if splits_to_process is None:
        splits_to_process = valid_splits
    else:
        for split in splits_to_process:
            if split not in valid_splits:
                raise ValueError(f"Invalid split: {split}")

    # --- Load Zoom Demos if path provided ---
    zoom_demos_lookup = None
    if zoom_demos_path and "train" in splits_to_process:
        print(f"Loading zoom demonstrations from: {zoom_demos_path}")
        try:
            with open(zoom_demos_path, "r") as f:
                demos = [json.loads(line) for line in f]
            zoom_demos_lookup = {
                demo["question_id"]: demo["keypoint"] for demo in demos
            }
            print(f"Loaded {len(zoom_demos_lookup)} zoom demonstrations.")
        except FileNotFoundError:
            print(
                f"Warning: Zoom demos file not found at {zoom_demos_path}. Skipping augmentation."
            )
            zoom_demos_lookup = None
        except Exception as e:
            print(
                f"Warning: Error loading zoom demos from {zoom_demos_path}: {e}. Skipping augmentation."
            )
            zoom_demos_lookup = None
    # --- End Load Zoom Demos ---

    processed_datasets = {}
    for split in splits_to_process:
        print(f"Processing {split} split...")
        examples = []
        for example in tqdm(dataset[split], desc=f"Processing {split} examples"):
            processed_example = generate_r1_messages(example)
            examples.append(processed_example)

            if max_examples_per_split is not None:
                if len(examples) >= max_examples_per_split:
                    break

        processed_dataset = Dataset.from_list(examples)

        if split == "train" and zoom_demos_lookup is not None:
            print(f"Augmenting '{split}' split with zoom keypoints...")

            def add_zoom_keypoint(example):
                keypoint = zoom_demos_lookup.get(example["question_id"], None)
                return {"zoom_keypoint": keypoint}

            # Add the column
            processed_dataset = processed_dataset.map(
                add_zoom_keypoint
            )  # Added num_proc for potential speedup
            print(f"Added 'zoom_keypoint' column to '{split}' split.")

            # --- REORDERING START ---
            print(
                f"Reordering '{split}' split to place examples with keypoints first..."
            )
            # Filter examples with and without keypoints
            with_keypoint_ds = processed_dataset.filter(
                lambda example: example["zoom_keypoint"] is not None
            )
            without_keypoint_ds = processed_dataset.filter(
                lambda example: example["zoom_keypoint"] is None
            )

            # --- Shuffle the 'without_keypoint_ds' ---
            print("Shuffling examples without keypoints...")
            without_keypoint_ds = without_keypoint_ds.shuffle(
                seed=42
            )  # Added shuffle with seed
            print("Shuffled examples without keypoints.")
            # --- End Shuffle ---

            # Concatenate them in the desired order
            processed_dataset = concatenate_datasets(
                [with_keypoint_ds, without_keypoint_ds]
            )
            print(f"Reordered '{split}' split.")
            # --- REORDERING END ---

        processed_datasets[split] = processed_dataset

    return DatasetDict(processed_datasets)


if __name__ == "__main__":
    dataset = create_r1_text_vqa_tool_use_dataset(
        max_examples_per_split=10000, splits_to_process=["train"]
    )

    # count how many examples have zoom_keypoint
    num_with_keypoint = len(
        [
            example
            for example in dataset["train"]
            if example["zoom_keypoint"] is not None
        ]
    )
    print(f"Number of examples with zoom_keypoint: {num_with_keypoint}")

    # Verify the first few examples have keypoints (if any exist)
    print("\nVerifying first few examples have keypoints (if available):")
    for i in range(min(5, num_with_keypoint)):
        print(
            f"Example {i}: QID={dataset['train'][i]['question_id']}, Keypoint={dataset['train'][i]['zoom_keypoint']}"
        )

    # Verify an example after the keypoint block has None (if applicable)
    if num_with_keypoint < len(dataset["train"]):
        print("\nVerifying example after the keypoint block:")
        idx_after = num_with_keypoint  # First example expected to have None
        print(
            f"Example {idx_after}: QID={dataset['train'][idx_after]['question_id']}, Keypoint={dataset['train'][idx_after]['zoom_keypoint']}"
        )
