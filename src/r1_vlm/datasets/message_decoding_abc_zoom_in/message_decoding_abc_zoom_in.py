import os
import random
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from PIL import Image, ImageDraw
from tqdm import tqdm
from r1_vlm.datasets.message_decoding_words.message_decoding_words_dataset import get_font


def random_mapping(alphabet):
    """
    Generate a random mapping (permutation) of the given alphabet,
    allowing letters to map to themselves.
    """
    # random.sample returns a random permutation of the list.
    shuffled = random.sample(alphabet, len(alphabet))
    return dict(zip(alphabet, shuffled))

def generate_decoder_image(
    mapping,
    image_size=300,
    background_color="white",
    text_color="black",
    font_size=4,
    edge_padding=10,
):
    """
    Generate an image showing the letter mappings.
    Each line shows a mapping in the format: A→B
    """
    image = Image.new("RGB", (image_size, image_size), background_color)
    draw = ImageDraw.Draw(image)
    full_coordinates = {}

    # Calculate font size as a fraction of total height divided by number of items
    font = get_font(font_size)

    mapping_items = list(mapping.items())
    random.shuffle(mapping_items)

    # Calculate total height needed for text
    sample_text = "A\u2192B"
    bbox = draw.textbbox((0, 0), sample_text, font=font)
    text_height = bbox[3] - bbox[1]

    # Calculate spacing to distribute items evenly
    spacing = text_height + 5

    # Start at half spacing to center everything vertically
    current_y = random.randint(edge_padding, int(image_size - edge_padding - text_height))
    for source, target in mapping_items:
        mapping_text = f"{source}\u2192{target}"
        bbox = draw.textbbox((0, 0), mapping_text, font=font)
        text_width = bbox[2] - bbox[0]
        # randomly find the x. make sure it's not too close to the edge
        x = random.randint(edge_padding, int(image_size - edge_padding - text_width))
        draw.text((x, current_y), mapping_text, fill=text_color, font=font)
        full_coordinates[mapping_text] = (x, current_y, font_size)
        current_y += text_height + spacing

    return image, full_coordinates


def generate_sample(
    alphabet_str="ABC",
    min_length=1,
    max_length=5,
    image_width=300,
    image_height=300,
    font_path=None,
    font_size=20,
    margin=10,
    line_spacing=5,
):
    """
    Generate a sample consisting of:
      - An image of the letter-to-letter mapping (the decoder).
      - The decoded message: the result of applying the mapping to the coded message.
      - The coded message: a random sequence of letters from the alphabet.

    The mapping is a random permutation of the given alphabet, so letters may map to themselves.

    Args:
        alphabet_str: String containing the alphabet to use
        min_length: Minimum length of the coded message
        max_length: Maximum length of the coded message

    Returns:
        A tuple: (image, decoded_message, coded_message, mapping)
    """
    alphabet = list(alphabet_str)
    mapping = random_mapping(alphabet)

    # Generate random message length
    message_length = random.randint(min_length, max_length)

    # Generate random coded message by sampling from alphabet
    coded_message = "".join(random.choices(alphabet, k=message_length))

    # The decoded message is obtained by applying the mapping to the coded message
    decoded_message = "".join(mapping[letter] for letter in coded_message)

    image, full_coordinates = generate_decoder_image(mapping)

    return image, decoded_message, coded_message, mapping, full_coordinates


def create_dataset(
    num_train_samples=10000,
    num_val_samples=1000,
    alphabet="ABC",
    min_length=1,
    max_length=5,
    base_dir="dataset",
):
    """Create train and validation datasets for the decoder task"""

    base_dir = os.path.abspath(base_dir)

    def get_subdirectory(split, index):
        """Create subdirectory path based on index (1000 files per directory)"""
        subdir_index = index // 1000
        # Return both the full path for saving and relative path for dataset
        full_path = os.path.join(
            base_dir, "images", split, f"subset_{subdir_index:03d}"
        )
        relative_path = os.path.join("images", split, f"subset_{subdir_index:03d}")
        return full_path, relative_path

    def generate_samples(num_samples, split):
        data = {
            "coded_message": [],
            "decoded_message": [],
            "mapping": [],
            "file_path": [],
            "image": [],
        }

        for i in tqdm(range(num_samples), desc=f"Generating {split} samples"):
            image, decoded_msg, coded_msg, mapping, full_coordinates = generate_sample(
                alphabet_str=alphabet, min_length=min_length, max_length=max_length
            )

            # Get both full and relative paths
            full_subdir, relative_subdir = get_subdirectory(split, i)
            os.makedirs(full_subdir, exist_ok=True)

            # Save image using full path but store relative path
            filename = f"{i:05d}.png"
            full_path = os.path.join(full_subdir, filename)
            relative_path = os.path.join(relative_subdir, filename)

            image.save(full_path)

            data["coded_message"].append(coded_msg)
            data["decoded_message"].append(decoded_msg)
            data["mapping"].append(mapping)
            data["file_path"].append(relative_path)
            data["image"].append(image)
            data["full_coordinates"].append(full_coordinates)

        return Dataset.from_dict(data)

    # Generate train and validation datasets
    train_dataset = generate_samples(num_train_samples, "train")
    val_dataset = generate_samples(num_val_samples, "val")

    # Combine into a DatasetDict
    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    return dataset


# Example usage:
if __name__ == "__main__":
    dataset = create_dataset(base_dir="data/message_decoding/dataset")

    api = HfApi()
    repo_id = "sunildkumar/message-decoding-dataset"

    # Upload the dataset
    dataset.push_to_hub(repo_id, token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    # Upload the dataset directory containing the images subdirectory
    # if this fails, try uploading with cli.
    # uv run huggingface-cli upload-large-folder "sunildkumar/message-decoding-dataset" "dataset" --repo-type=dataset --no-bars --no-reports
    # UPDATE: The HF API won't stop rate limiting me... 
    #api.upload_large_folder(folder_path="data/message_decoding/dataset", repo_id=repo_id, repo_type="dataset")