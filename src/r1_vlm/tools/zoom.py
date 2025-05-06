# generic zoom tool
import json
import math  # Added import
from typing import List, Tuple

import pytest
from PIL import Image

from r1_vlm.environments.tool_vision_env import RawToolArgs, TypedToolArgs


def calculate_crop_coordinates(
    keypoint: List[int],
    image_size: Tuple[int, int],
    aspect_ratio_mode: str,
    base_target_size: int = 400,
) -> Tuple[int, int, int, int]:
    """
    Calculates the crop coordinates for a box centered around a keypoint,
    with a specified aspect ratio.

    The box aims for an area roughly equal to base_target_size * base_target_size.
    If the target box extends beyond the image boundaries, the box is shifted
    to stay within the image, maintaining the target size and aspect ratio.
    The keypoint will no longer be centered in this case.

    Args:
        keypoint: A list or tuple [x, y] representing the desired center coordinates.
        image_size: A tuple (width, height) of the original image.
        aspect_ratio_mode: The desired aspect ratio ("square", "horizontal", "vertical").
        base_target_size: The base dimension used for area calculation.

    Returns:
        A tuple (crop_left, crop_upper, crop_right, crop_lower) defining the
        region to crop from the original image.

    Raises:
        ValueError: If inputs are invalid types/formats, keypoint is outside
                    the image boundaries, or aspect_ratio_mode is invalid.
    """
    # --- Input Validation ---
    if (
        not isinstance(keypoint, (list, tuple))
        or len(keypoint) != 2
        or not all(isinstance(coord, int) for coord in keypoint)
    ):
        raise ValueError(
            "Error:keypoint must be a list or tuple of two integers [x, y]"
        )

    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in image_size)
    ):
        raise ValueError(
            "Error:image_size must be a tuple of two positive integers (width, height)"
        )

    if not isinstance(base_target_size, int) or base_target_size <= 0:
        raise ValueError("Error: base_target_size must be a positive integer")

    if aspect_ratio_mode not in ["square", "horizontal", "vertical"]:
        raise ValueError(
            "Error: aspect_ratio_mode must be 'square', 'horizontal', or 'vertical'"
        )

    x, y = keypoint
    width, height = image_size

    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"Error: keypoint [{x}, {y}] is outside the image boundaries "
            f"(width={width}, height={height})"
        )

    # --- Calculate Target Crop Dimensions based on Aspect Ratio ---
    sqrt2 = math.sqrt(2)
    if aspect_ratio_mode == "square":
        target_crop_width = base_target_size
        target_crop_height = base_target_size
    elif aspect_ratio_mode == "horizontal":  # 2:1
        target_crop_width = round(base_target_size * sqrt2)
        target_crop_height = round(base_target_size / sqrt2)
    else:  # aspect_ratio_mode == "vertical" (1:2)
        target_crop_width = round(base_target_size / sqrt2)
        target_crop_height = round(base_target_size * sqrt2)

    # Ensure dimensions are at least 1
    target_crop_width = max(1, target_crop_width)
    target_crop_height = max(1, target_crop_height)

    # --- Calculate Crop Box ---
    # Calculate half-sizes (integer division)
    half_width = target_crop_width // 2
    half_height = target_crop_height // 2

    # Calculate the ideal top-left corner to center the box
    ideal_x_min = x - half_width
    ideal_y_min = y - half_height

    # Initialize the actual top-left corner
    crop_left = ideal_x_min
    crop_upper = ideal_y_min

    # Adjust if the box extends beyond the right or bottom boundaries
    if crop_left + target_crop_width > width:
        crop_left = width - target_crop_width
    if crop_upper + target_crop_height > height:
        crop_upper = height - target_crop_height

    # Adjust if the box extends beyond the left or top boundaries
    if crop_left < 0:
        crop_left = 0
    if crop_upper < 0:
        crop_upper = 0

    # Calculate the final right and lower bounds for PIL crop (exclusive)
    # Clamp the right/lower bounds to the image dimensions, important if
    # target crop dimensions > image width/height.
    crop_right = min(crop_left + target_crop_width, width)
    crop_lower = min(crop_upper + target_crop_height, height)

    # Ensure left/upper are also clamped in case target_size > image dimensions
    crop_left = max(0, crop_left)
    crop_upper = max(0, crop_upper)

    return (crop_left, crop_upper, crop_right, crop_lower)


def zoom(
    image_name: str,
    keypoint: list[int],
    aspect_ratio_mode: str,
    **kwargs,
) -> Image.Image:
    """
    Returns an image zoomed in on the specified keypoint with a given aspect ratio.
    This is useful to see a region of interest with higher clarity, like for reading text or identifying objects.
    The choice of aspect ration is helpful depending on the task.
    If you are looking at horizontal text or some other wide region, you might use "horizontal" aspect ratio.
    If you are looking at vertical text or some other tall region, you might use "vertical" aspect ratio.
    If you are looking at a region more generally, you might use "square" aspect ratio.

    Args:
        image_name: str, the name of the image to zoom in on. Can only be called on the "input_image".
        keypoint: list[int], the [x, y] coordinates of the point to center the zoom on. Must be within the image boundaries.
        aspect_ratio_mode: str, the desired aspect ratio for the zoom window. Must be one of "horizontal","square", or "vertical". "horizontal" is 2:1 (wider), "square" is 1:1, "vertical" is 1:2 (taller).

    Returns:
        The image zoomed in on the specified keypoint with the requested aspect ratio.

    Examples:
        <tool>
        name: zoom
        image_name: input_image
        aspect_ratio_mode: horizontal
        keypoint: [500, 400]

        </tool>
        <tool>
        name: zoom
        image_name: input_image
        aspect_ratio_mode: square
        keypoint: [23, 44]

        </tool>
        <tool>
        name: zoom
        image_name: input_image
        aspect_ratio_mode: vertical
        keypoint: [723, 461]

        </tool>
    """
    # get and validate the image
    images = kwargs["images"]
    image = images.get(image_name, None)
    if image is None:
        valid_image_names = list(images.keys())
        raise ValueError(
            f"Error: Image {image_name} not found. Valid image names are: {valid_image_names}"
        )

    if image_name != "input_image":
        raise ValueError(
            f"Error: Image {image_name} is not the input_image. This tool can only be called on the input_image."
        )

    original_width, original_height = image.size
    # base size for cropping area calculation
    base_crop_target_size = 400

    # Validate aspect_ratio_mode and keypoint, then calculate crop box
    crop_box = calculate_crop_coordinates(
        keypoint,
        (original_width, original_height),
        aspect_ratio_mode,
        base_target_size=base_crop_target_size,
    )

    # crop the image to the calculated box
    cropped_image = image.crop(crop_box)
    crop_w, crop_h = cropped_image.size

    # If crop resulted in zero size (e.g., invalid crop box calculation edge case)
    if crop_w == 0 or crop_h == 0:
        raise ValueError("Error: Cropped image has zero dimension")

    # Determine the base dimension for resizing the output - use original image dims or 400 min
    base_resize_dim = max(original_width, original_height, 400)

    # Calculate the final output size, preserving aspect ratio of the cropped image
    # Scale so the LARGER dimension of the output matches base_resize_dim
    if crop_w >= crop_h:  # Wider or square crop
        output_w = base_resize_dim
        output_h = round(output_w * crop_h / crop_w)  # Maintain aspect ratio
    else:  # Taller crop
        output_h = base_resize_dim
        output_w = round(output_h * crop_w / crop_h)  # Maintain aspect ratio

    # Ensure minimum dimension is 1
    output_w = max(1, output_w)
    output_h = max(1, output_h)
    output_size = (output_w, output_h)

    # Resize the cropped image to the calculated output size, preserving aspect ratio
    output_image = cropped_image.resize(output_size, Image.Resampling.LANCZOS)

    print(
        f"Zoom tool cropped to {crop_w}x{crop_h}, resized output to: {output_image.size}"
    )

    return output_image


def parse_zoom_args(raw_args: RawToolArgs) -> TypedToolArgs:
    """
    Parses raw string arguments for the zoom tool (keypoint version).

    Expects keys: 'name', 'image_name', 'keypoint', 'aspect_ratio_mode'.
    Converts 'keypoint' from a JSON string to a list of integers.
    Detailed validation of values (e.g., keypoint coordinates)
    is deferred to the zoom function itself.

    Args:
        raw_args: Dictionary with string keys and string values from the general parser.

    Returns:
        A dictionary containing the arguments with basic type conversions applied,
        ready for the zoom function. Keys: 'image_name', 'keypoint', 'aspect_ratio_mode'.

    Raises:
        ValueError: If required keys are missing, extra keys are present,
                    basic type conversion fails (e.g., 'keypoint' is not valid JSON
                    or doesn't result in a list of two integers), or
                    'aspect_ratio_mode' has an invalid value.
    """
    required_keys = {
        "name",
        "image_name",
        "keypoint",
        "aspect_ratio_mode",
    }  # Added aspect_ratio_mode
    actual_keys = set(raw_args.keys())

    # 1. Check for Missing Keys
    missing_keys = required_keys - actual_keys
    if missing_keys:
        raise ValueError(
            f"Missing required arguments for zoom tool: {', '.join(sorted(missing_keys))}"
        )

    # 2. Check for extra keys
    extra_keys = actual_keys - required_keys
    if extra_keys:
        raise ValueError(
            f"Unexpected arguments for zoom tool: {', '.join(sorted(extra_keys))}"
        )

    # 3. Perform Basic Type Conversions
    typed_args: TypedToolArgs = {}
    try:
        # Keep image_name as string
        typed_args["image_name"] = raw_args["image_name"]

        # Convert keypoint string to list of ints
        keypoint_list = json.loads(raw_args["keypoint"])

        # Basic validation of the parsed keypoint structure and type
        if not isinstance(keypoint_list, list) or len(keypoint_list) != 2:
            raise ValueError("Error: 'keypoint' must be a JSON list of two elements.")
        if not all(isinstance(coord, int) for coord in keypoint_list):
            raise ValueError(
                "Error: Both elements in 'keypoint' list must be integers."
            )
        typed_args["keypoint"] = keypoint_list

        # Validate and clean aspect_ratio_mode
        aspect_mode = (
            raw_args["aspect_ratio_mode"].strip().strip('"')
        )  # Clean the input string
        if aspect_mode not in ["square", "horizontal", "vertical"]:
            raise ValueError(
                f"Error: aspect_ratio_mode must be 'square', 'horizontal', or 'vertical', but got '{raw_args['aspect_ratio_mode']}'"
            )
        typed_args["aspect_ratio_mode"] = aspect_mode  # Store the cleaned value

    except json.JSONDecodeError:
        raise ValueError(
            f"Error: Invalid JSON format for 'keypoint': '{raw_args['keypoint']}'"
        )
    except (
        ValueError
    ) as e:  # Catch specific ValueErrors raised during keypoint validation
        raise e
    except KeyError as e:
        # This should ideally be caught by the missing keys check above, but as a safeguard:
        raise ValueError(f"Error: Missing key '{e}' during conversion phase.")

    return typed_args


@pytest.fixture
def sample_image():
    # Create a larger test image
    img = Image.new("RGB", (1000, 800))
    return {"input_image": img}


# --- Tests for calculate_crop_coordinates (kept separate for clarity) ---
def test_calc_centered(sample_image):
    img_size = sample_image["input_image"].size
    kp = [500, 400]
    ts = 300
    box = calculate_crop_coordinates(kp, img_size, "square", ts)
    assert box == (350, 250, 650, 550)


def test_calc_top_left(sample_image):
    img_size = sample_image["input_image"].size
    kp = [50, 40]
    ts = 300
    box = calculate_crop_coordinates(kp, img_size, "square", ts)
    assert box == (0, 0, 300, 300)


def test_calc_bottom_right(sample_image):
    img_size = sample_image["input_image"].size
    kp = [950, 750]
    ts = 300
    box = calculate_crop_coordinates(kp, img_size, "square", ts)
    assert box == (700, 500, 1000, 800)


def test_calc_right_middle(sample_image):
    img_size = sample_image["input_image"].size
    kp = [950, 400]
    ts = 300
    box = calculate_crop_coordinates(kp, img_size, "square", ts)
    assert box == (700, 250, 1000, 550)


def test_calc_small_image(sample_image):
    small_img_size = (200, 150)
    kp = [100, 75]
    ts = 300
    box = calculate_crop_coordinates(kp, small_img_size, "square", ts)
    assert box == (0, 0, 200, 150)


def test_calc_invalid_keypoint_coords(sample_image):
    img_size = sample_image["input_image"].size
    ts = 300
    with pytest.raises(ValueError, match="outside the image boundaries"):
        calculate_crop_coordinates([1000, 400], img_size, "square", ts)
    with pytest.raises(ValueError, match="outside the image boundaries"):
        calculate_crop_coordinates([-1, 400], img_size, "square", ts)
    with pytest.raises(ValueError, match="outside the image boundaries"):
        calculate_crop_coordinates([500, 800], img_size, "square", ts)
    with pytest.raises(ValueError, match="outside the image boundaries"):
        calculate_crop_coordinates([500, -1], img_size, "square", ts)


def test_calc_invalid_input_types():
    with pytest.raises(ValueError, match="keypoint must be a list or tuple"):
        calculate_crop_coordinates("[100, 100]", (500, 500), "square", 300)
    with pytest.raises(ValueError, match="keypoint must be a list or tuple"):
        calculate_crop_coordinates([100], (500, 500), "square", 300)
    with pytest.raises(ValueError, match="keypoint must be a list or tuple"):
        calculate_crop_coordinates([100.0, 100], (500, 500), "square", 300)
    with pytest.raises(ValueError, match="image_size must be a tuple"):
        calculate_crop_coordinates([100, 100], [500, 500], "square", 300)
    with pytest.raises(ValueError, match="image_size must be a tuple"):
        calculate_crop_coordinates([100, 100], (500, 0), "square", 300)
    with pytest.raises(ValueError, match="target_size must be a positive integer"):
        calculate_crop_coordinates([100, 100], (500, 500), "square", 0)
    with pytest.raises(ValueError, match="target_size must be a positive integer"):
        calculate_crop_coordinates([100, 100], (500, 500), "square", -100)


# --- Tests for zoom function ---


def test_zoom_invalid_image_name(sample_image):
    with pytest.raises(ValueError, match="not found"):
        zoom("nonexistent_image", [100, 100], "square", images=sample_image)


# Add test for incorrect image name usage (e.g., "test_image" instead of "input_image")
def test_zoom_incorrect_image_name_usage(sample_image):
    with pytest.raises(ValueError, match="is not the input_image"):
        zoom(
            "test_image",
            [100, 100],
            "square",
            images={"test_image": sample_image["input_image"]},
        )


# Test invalid keypoint format passed directly (should be caught by calculate_crop_coordinates)
def test_zoom_invalid_keypoint_format(sample_image):
    with pytest.raises(ValueError, match="keypoint must be a list or tuple"):
        zoom(
            "input_image", "[100, 100]", "square", images=sample_image
        )  # Pass string instead of list
    with pytest.raises(ValueError, match="keypoint must be a list or tuple"):
        zoom(
            "input_image", [100], "square", images=sample_image
        )  # Pass list with 1 element
    with pytest.raises(ValueError, match="keypoint must be a list or tuple"):
        zoom("input_image", [100.0, 100], "square", images=sample_image)  # Pass float


# Test keypoint outside image bounds (should be caught by calculate_crop_coordinates)
def test_zoom_keypoint_out_of_bounds(sample_image):
    img_size = sample_image["input_image"].size
    with pytest.raises(ValueError, match="outside the image boundaries"):
        zoom(
            "input_image", [img_size[0], 100], "square", images=sample_image
        )  # x = width
    with pytest.raises(ValueError, match="outside the image boundaries"):
        zoom("input_image", [-1, 100], "square", images=sample_image)  # x < 0
    with pytest.raises(ValueError, match="outside the image boundaries"):
        zoom(
            "input_image", [100, img_size[1]], "square", images=sample_image
        )  # y = height
    with pytest.raises(ValueError, match="outside the image boundaries"):
        zoom("input_image", [100, -1], "square", images=sample_image)  # y < 0


def test_zoom_basic_centered(sample_image):
    keypoint = [500, 400]  # Center of 1000x800 image
    result = zoom("input_image", keypoint, "square", images=sample_image)
    assert isinstance(result, Image.Image)
    assert result.size == (300, 300)  # Always 300x300 output


def test_zoom_edge_case_top_left(sample_image):
    keypoint = [50, 40]  # Near top-left
    result = zoom("input_image", keypoint, "square", images=sample_image)
    assert isinstance(result, Image.Image)
    assert result.size == (300, 300)


def test_zoom_edge_case_bottom_right(sample_image):
    keypoint = [950, 750]  # Near bottom-right
    result = zoom("input_image", keypoint, "square", images=sample_image)
    assert isinstance(result, Image.Image)
    assert result.size == (300, 300)


def test_zoom_small_image(sample_image):
    # Use a smaller image where the crop box will be the whole image
    small_img = Image.new("RGB", (200, 150))
    small_sample = {"input_image": small_img}
    keypoint = [100, 75]  # Center of small image
    result = zoom("input_image", keypoint, "square", images=small_sample)
    assert isinstance(result, Image.Image)
    assert result.size == (300, 300)  # Output is still 300x300 after resize


# It's harder to precisely test *content* after resize, especially with edge cases
# We rely on the calculate_crop_coordinates tests and assume PIL's crop/resize work.
# A simple check: ensure the output is not blank for a non-blank input area.
def test_zoom_content_check(sample_image):
    # Fill a known area in the original image
    img = sample_image["input_image"].copy()
    img.paste((255, 0, 0), (400, 300, 600, 500))  # Red box in the center
    filled_sample = {"input_image": img}
    keypoint = [500, 400]  # Center point within the red box

    result = zoom("input_image", keypoint, "square", images=filled_sample)
    assert result.size == (300, 300)
    # Check a central pixel in the output (should correspond to the red box)
    # The original keypoint [500, 400] should map to the center [150, 150] in the 300x300 output
    center_pixel_value = result.getpixel((150, 150))
    assert center_pixel_value == (255, 0, 0)


# --- Tests for parse_zoom_args ---


def test_parse_valid_args():
    raw = {
        "name": "zoom",
        "image_name": "input_image",
        "keypoint": "[150, 250]",
        "aspect_ratio_mode": "square",
    }
    expected = {
        "image_name": "input_image",
        "keypoint": [150, 250],
        "aspect_ratio_mode": "square",
    }
    assert parse_zoom_args(raw) == expected


def test_parse_missing_key():
    raw = {"name": "zoom", "image_name": "input_image", "keypoint": "[150, 250]"}
    with pytest.raises(
        ValueError, match="Missing required arguments.*aspect_ratio_mode"
    ):
        parse_zoom_args(raw)


def test_parse_extra_key():
    raw = {
        "name": "zoom",
        "image_name": "input_image",
        "keypoint": "[100, 100]",
        "aspect_ratio_mode": "square",
        "extra": "bad",
    }
    with pytest.raises(ValueError, match="Unexpected arguments.*extra"):
        parse_zoom_args(raw)


def test_parse_invalid_json():
    raw = {
        "name": "zoom",
        "image_name": "input_image",
        "keypoint": "[100, 100",
        "aspect_ratio_mode": "square",
    }  # Missing closing bracket
    with pytest.raises(ValueError, match="Invalid JSON format for 'keypoint'"):
        parse_zoom_args(raw)


def test_parse_keypoint_not_list():
    raw = {
        "name": "zoom",
        "image_name": "input_image",
        "keypoint": '{"x": 100, "y": 100}',
        "aspect_ratio_mode": "square",
    }
    with pytest.raises(ValueError, match="'keypoint' must be a JSON list"):
        parse_zoom_args(raw)


def test_parse_keypoint_wrong_length():
    raw = {"name": "zoom", "image_name": "input_image", "keypoint": "[100]"}
    with pytest.raises(
        ValueError, match="'keypoint' must be a JSON list of two elements"
    ):
        parse_zoom_args(raw)
    raw = {"name": "zoom", "image_name": "input_image", "keypoint": "[100, 200, 300]"}
    with pytest.raises(
        ValueError, match="'keypoint' must be a JSON list of two elements"
    ):
        parse_zoom_args(raw)


def test_parse_keypoint_wrong_type():
    raw = {"name": "zoom", "image_name": "input_image", "keypoint": "[100.5, 200]"}
    with pytest.raises(
        ValueError, match="elements in 'keypoint' list must be integers"
    ):
        parse_zoom_args(raw)
    raw = {"name": "zoom", "image_name": "input_image", "keypoint": '["100", 200]'}
    with pytest.raises(
        ValueError, match="elements in 'keypoint' list must be integers"
    ):
        parse_zoom_args(raw)


# --- Remove old/irrelevant tests ---
# (test_invalid_bbox_length, test_invalid_bbox_value_types, test_bbox_coordinates_reversed, etc. are no longer needed)
# (test_zoom_fixed_magnification, test_zoom_max_size_constraint are replaced by fixed size tests)
# (test_zoom_different_regions, test_zoom_preserve_content, test_zoom_aspect_ratio are less relevant or covered by new tests)


# Keep the main execution block if desired for direct running
if __name__ == "__main__":
    # You can add specific calls here for manual testing if needed
    # e.g., create a sample image and call zoom
    print("Running manual tests from __main__...")
    img_w, img_h = 1000, 800
    kp_center = [500, 400]
    kp_edge = [50, 40]
    img = Image.new("RGB", (img_w, img_h), color="blue")
    images_dict = {"input_image": img}

    try:
        print(f"\nTesting center keypoint: {kp_center}")
        result_center = zoom("input_image", kp_center, "square", images=images_dict)
        print(f"Center zoom output size: {result_center.size}")
        # result_center.show() # Optionally display the image

        print(f"\nTesting edge keypoint: {kp_edge}")
        result_edge = zoom("input_image", kp_edge, "square", images=images_dict)
        print(f"Edge zoom output size: {result_edge.size}")
        # result_edge.show() # Optionally display the image

        print("\nTesting invalid keypoint:")
        try:
            zoom("input_image", [1200, 100], "square", images=images_dict)
        except ValueError as e:
            print(f"Caught expected error: {e}")

        print("\nTesting parser:")
        raw_valid = {
            "name": "zoom",
            "image_name": "input_image",
            "keypoint": "[150, 250]",
            "aspect_ratio_mode": "square",
        }
        parsed = parse_zoom_args(raw_valid)
        print(f"Parsed valid args: {parsed}")

        raw_invalid = {
            "name": "zoom",
            "image_name": "input_image",
            "keypoint": "[150.5, 250]",
            "aspect_ratio_mode": "square",
        }
        try:
            parse_zoom_args(raw_invalid)
        except ValueError as e:
            print(f"Caught expected parser error: {e}")

    except Exception as e:
        print(f"An unexpected error occurred during manual testing: {e}")

    print("\nManual tests finished.")
