import os
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from r1_vlm.datasets.message_decoding_words.message_decoding_words_dataset import (
    generate_mapping,
    get_font,
)

# setting a seed for reproducibility
random.seed(42)

def generate_zoom_in_decoder_image(
    mapping,
    image_size=300,
    background_color="white",
    text_color="black",
    large_font_size=20,
    small_font_size=4,
    font_size_variation=1,
    small_mappings_keys=[],
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

    small_mapping_keys = set(small_mappings_keys)
    full_coordinates = {}

    # Place first 25 mappings in the grid
    for idx in range(25):
        source, target = mapping_items[idx]
        large_font = get_font(large_font_size)
        small_font = get_font(small_font_size)

        row = idx // 5
        col = idx % 5

        x = col * grid_width + (grid_width // 2)  # center of cell
        y = row * grid_height + (grid_height // 2)

        # Draw actual mapping text centered in each cell
        if source not in small_mapping_keys:
            mapping_text = f"{source}→{target}"
            bbox = draw.textbbox((0, 0), mapping_text, font=large_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center text in cell
            # If using random_size, we will also randomly shift the text in the cell, but making sure it's still inside the cell
            text_x = x - text_width // 2
            text_y = y - text_height // 2
            x_shift = random.randint(-grid_width // 4, grid_width // 4)
            y_shift = random.randint(-grid_height // 4, grid_height // 4)
            text_x += x_shift
            text_y += y_shift

            draw.text((text_x, text_y), mapping_text, fill=text_color, font=large_font)
            full_coordinates[mapping_text] = (text_x, text_y, large_font_size)
        else:
            # draw the f"source→" with large font size, and the f"{target}" with small font size
            source_text = f"{source}→"
            target_text = f"{target}"

            source_bbox = draw.textbbox((0, 0), source_text, font=large_font)
            target_bbox = draw.textbbox((0, 0), target_text, font=small_font)
            source_text_width = source_bbox[2] - source_bbox[0]
            source_text_height = source_bbox[3] - source_bbox[1]
            target_text_width = target_bbox[2] - target_bbox[0]
            target_text_height = target_bbox[3] - target_bbox[1]

            # Center text in cell
            source_text_x = x - source_text_width // 2
            source_text_y = y - source_text_height // 2
            target_text_x = x + source_text_width - target_text_width // 2
            target_text_y = source_text_y

            draw.text((source_text_x, source_text_y), source_text, fill=text_color, font=large_font)
            draw.text((target_text_x, target_text_y), target_text, fill=text_color, font=small_font)
            full_coordinates[source_text] = (source_text_x, source_text_y, large_font_size)
            full_coordinates[target_text] = (target_text_x, target_text_y, small_font_size)
            
    # Add the 26th mapping below the grid
    source, target = mapping_items[25]
    if source not in small_mapping_keys:
        font = get_font(large_font_size)
        bottom_text = f"{source}→{target}"
        bbox = draw.textbbox((0, 0), bottom_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position for bottom center text - right after the grid
        bottom_x = (image_size - text_width) // 2
        bottom_y = (5 * grid_height) + 10  # 10px padding after grid

        x_shift = random.randint(-grid_width // 4, grid_width // 4)
        y_shift = random.randint(-grid_height // 4, 0) # we don't want the bottom text to shift to the bottom further
        bottom_x += x_shift
        bottom_y += y_shift

        draw.text((bottom_x, bottom_y), bottom_text, fill=text_color, font=font)
        full_coordinates[bottom_text] = (bottom_x, bottom_y, large_font_size)
    else:
        # draw the f"source→" with large font size, and the f"{target}" with small font size
        source_text = f"{source}→"
        target_text = f"{target}"
        
        source_bbox = draw.textbbox((0, 0), source_text, font=large_font)
        target_bbox = draw.textbbox((0, 0), target_text, font=small_font)
        source_text_width = source_bbox[2] - source_bbox[0]
        source_text_height = source_bbox[3] - source_bbox[1]
        target_text_width = target_bbox[2] - target_bbox[0]
        target_text_height = target_bbox[3] - target_bbox[1]
        
        # Center text in cell
        source_text_x = x - source_text_width // 2
        source_text_y = y - source_text_height // 2
        target_text_x = x + source_text_width - target_text_width // 2
        target_text_y = source_text_y

        draw.text((source_text_x, source_text_y), source_text, fill=text_color, font=large_font)
        draw.text((target_text_x, target_text_y), target_text, fill=text_color, font=small_font)
        full_coordinates[source_text] = (source_text_x, source_text_y, large_font_size)
        full_coordinates[target_text] = (target_text_x, target_text_y, small_font_size)
        
    return image, full_coordinates

def create_sample(example):
    """
    Creates a sample for the message decoding dataset.
    We want the sample to have different levels of difficulty regarding the zoom-in tool usage.
    The difficulty level is determined by the number of small positives present in the decoder image.
    Theoretically, the number of small positives represents the number of zoom-in tool usages in a single inference run.
    In general, we want about 80% of the data to have a single small positive, 15% to have two, and the rest to have three. (this can be modified later)
    Meanwhile, the number of small negatives is always set to `5 - small positives`.
    """
    message = example["text"]

    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    assert len(alphabet) == 26
    mapping = generate_mapping(alphabet)

    positive_mappings = {k: v for k, v in mapping.items() if v in message.lower()} # mappings that should be used to decode the message
    negative_mappings = {k: v for k, v in mapping.items() if v not in message.lower()} # mappings that is irrelevant to the message

    # Determining the mappings that are showing small on the decoder image
    # number of small positives: 80% of the time 1, 15% of the time 2, rest 3
    num_small_positives = 1
    num_small_negatives = 5 - num_small_positives
    num_small_positives = min(num_small_positives, len(positive_mappings))
    num_small_negatives = min(num_small_negatives, len(negative_mappings))

    # selecting the mappings that will be shown small on the decoder image
    small_mappings_keys = random.sample(list(positive_mappings.keys()), num_small_positives)
    small_mappings_keys.extend(random.sample(list(negative_mappings.keys()), num_small_negatives))

    decoder_image, full_coordinates = generate_zoom_in_decoder_image(
        mapping=mapping,
        image_size=300,
        large_font_size=20,
        small_font_size=2,
        font_size_variation=1,
        small_mappings_keys=small_mappings_keys,
    )
    
    # add a mapping for the underscore ("_") character. It will map to " " (space).
    # This is so we can effectively communicate the space character in the coded message.
    mapping["_"] = " "

    # reverse the mapping to encode the message
    reverse_mapping = {v: k for k, v in mapping.items()}

    # create the coded and decoded message. If we encounter a character that is not in the mapping,
    # we will map it to itself.
    coded_message = ""
    decoded_message = ""
    for char in message:
        # check if the character is in the mapping
        if char.isascii() and (char.isalpha() or char == " "):
            is_lower = char.islower()

            # pass the lowercase version of the character to the mapping
            # .lower() on the space character is a no-op
            key_char = char.lower() if not is_lower else char
            mapped_char = reverse_mapping[key_char]

            # add lowercase char to the coded message
            coded_message += mapped_char
            # use the lowercase version of the character in the decoded message too.
            decoded_message += char.lower() if not is_lower else char

        # if the character is not in the mapping, something is wrong
        else:
            raise ValueError(f"Character {char} is not in the mapping")

    return decoder_image, decoded_message, coded_message, mapping, num_small_positives, num_small_negatives, full_coordinates


def create_dataset():
    data = {
        "coded_message": [],
        "decoded_message": [],
        "mapping": [],
        "file_path": [],
        "image": [],
        "task": [],
        "num_small_positives": [],
        "num_small_negatives": [],
        "full_coordinates": [],
    }

    image_dir = Path(__file__).parent / "images"
    image_dir.mkdir(exist_ok=True)

    # verify that the image directory is empty
    if len(list(image_dir.glob("*.png"))) > 0:
        raise ValueError("Image directory is not empty")

    # create dataset of words, word pairs, and word triples
    words_dataset = load_dataset("sunildkumar/popular_english_words", split="train")
    words_list = [example["word"] for example in words_dataset]
    examples = []

    # single word examples
    for word in words_list:
        examples.append({"text": word, "task": "word"})

    # word pair examples
    for i in range(len(words_list)):
        word1 = random.choice(words_list)
        word2 = random.choice(words_list)
        examples.append({"text": f"{word1} {word2}", "task": "word_2"})

    # word triple examples
    for i in range(len(words_list)):
        word1 = random.choice(words_list)
        word2 = random.choice(words_list)
        word3 = random.choice(words_list)
        examples.append({"text": f"{word1} {word2} {word3}", "task": "word_3"})

    data = {
        "coded_message": [],
        "decoded_message": [],
        "mapping": [],
        "file_path": [],
        "image": [],
        "task": [],
        "num_small_positives": [],
        "num_small_negatives": [],
        "full_coordinates": [],
    }

    # create dataset from examples
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        text = example["text"]
        task = example["task"]

        decoder_image, message, coded_message, mapping, num_small_positives, num_small_negatives, full_coordinates = create_sample(example)

        fpath = f"images/{i}.png"
        full_path = image_dir / f"{i}.png"

        # verify that the image doesn't already exist, if it does, something is wrong as we should error
        if full_path.exists():
            raise ValueError(f"Image {full_path} already exists")

        # Use full path for saving the image
        full_path = image_dir / f"{i}.png"

        decoder_image.save(full_path)

        data["coded_message"].append(coded_message)
        data["decoded_message"].append(message)
        data["mapping"].append(mapping)
        data["file_path"].append(fpath)
        data["image"].append(decoder_image)
        data["task"].append(task)
        data["num_small_positives"].append(num_small_positives)
        data["num_small_negatives"].append(num_small_negatives)
        data["full_coordinates"].append(full_coordinates)
    decoding_dataset = Dataset.from_dict(data)

    decoding_dataset.push_to_hub(
        "Groundlight/message-decoding-words-and-sequences-target-zoom-in",
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    )


if __name__ == "__main__":
    create_dataset()
