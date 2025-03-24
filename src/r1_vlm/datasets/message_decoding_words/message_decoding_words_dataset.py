import os
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# setting a seed for reproducibility
random.seed(42)

# creates a dataset mapping scrambled words to their unscrambled form. Includes decoder as an image.


def generate_mapping(alphabet: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
    """
    maps each character in the alphabet to a unique random character in the alphabet,
    creating a bijective mapping.
    """
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    mapping = dict(zip(alphabet, shuffled))
    return mapping


def get_font(size):
    """Returns the font for the message decoding dataset."""
    font_path = Path(__file__).parent / "fonts" / "NotoSansSymbols-Regular.ttf"
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found: {font_path.resolve()}")
    return ImageFont.truetype(font_path, size)


def generate_decoder_image(
    mapping, image_size=300, background_color="white", text_color="black"
):
    """
    Generates an image of the decoder, which is a 5x5 grid plus one extra mapping,
    showing character mappings in A→B format.
    """
    image = Image.new("RGB", (image_size, image_size), background_color)
    draw = ImageDraw.Draw(image)

    # shuffle the order of the mapping items
    mapping_items = list(mapping.items())
    random.shuffle(mapping_items)

    # Calculate grid dimensions
    grid_width = image_size // 5  # 60px
    grid_height = (image_size - 50) // 5  # 50px (reserving 50px for bottom item)

    # Start with a relatively small font size - we can adjust this
    font = get_font(20)

    # Place first 25 mappings in the grid
    for idx in range(25):
        row = idx // 5
        col = idx % 5
        source, target = mapping_items[idx]

        x = col * grid_width + (grid_width // 2)  # center of cell
        y = row * grid_height + (grid_height // 2)

        # Draw actual mapping text centered in each cell
        mapping_text = f"{source}→{target}"
        bbox = draw.textbbox((0, 0), mapping_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center text in cell
        text_x = x - text_width // 2
        text_y = y - text_height // 2

        draw.text((text_x, text_y), mapping_text, fill=text_color, font=font)

    # Add the 26th mapping below the grid
    source, target = mapping_items[25]
    bottom_text = f"{source}→{target}"
    bbox = draw.textbbox((0, 0), bottom_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position for bottom center text - right after the grid
    bottom_x = (image_size - text_width) // 2
    bottom_y = (5 * grid_height) + 10  # 10px padding after grid

    draw.text((bottom_x, bottom_y), bottom_text, fill=text_color, font=font)

    return image


def create_sample(example):
    """
    Given an example from the popular_english_words dataset,
    create an example for the message decoding dataset.
    """
    word = example["word"]
    word = word.upper()

    # this mapping is used to scramble the word
    mapping = generate_mapping()

    coded_word = "".join(mapping[char] for char in word)

    # Create the decoder mapping by reversing the scramble mapping
    decoder_mapping = {v: k for k, v in mapping.items()}

    # Pass the decoder mapping to generate the image
    image = generate_decoder_image(decoder_mapping)

    return image, word, coded_word, decoder_mapping


def create_dataset():
    words_dataset = load_dataset("sunildkumar/popular_english_words", split="train")
    data = {
        "coded_message": [],
        "decoded_message": [],
        "mapping": [],
        "file_path": [],
        "image": [],
    }

    image_dir = Path(__file__).parent / "images"
    image_dir.mkdir(exist_ok=True)

    for i, example in tqdm(enumerate(words_dataset), total=len(words_dataset)):
        image, decoded_word, coded_word, decoder_mapping = create_sample(example)

        # Store only the relative path starting from 'images/'
        fpath = f"images/{i}.png"

        # Use full path for saving the image
        full_path = image_dir / f"{i}.png"
        image.save(full_path)

        data["coded_message"].append(coded_word)
        data["decoded_message"].append(decoded_word)
        data["mapping"].append(decoder_mapping)
        data["file_path"].append(fpath)
        data["image"].append(image)

    message_decoding_dataset = Dataset.from_dict(data)

    message_decoding_dataset.push_to_hub(
        "sunildkumar/message-decoding-words", token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


if __name__ == "__main__":
    message_decoding_dataset = create_dataset()
