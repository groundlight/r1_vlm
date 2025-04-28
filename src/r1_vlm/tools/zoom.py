# generic zoom tool
import json

import pytest
from PIL import Image

from r1_vlm.environments.tool_vision_env import RawToolArgs, TypedToolArgs


def zoom(
    image_name: str,
    bbox: list[int],
    **kwargs,
) -> Image.Image:
    """
    Returns the original image, cropped into the region specified by the bounding box,
    and then up-scaled by a fixed factor of 2.5x. This tool is useful to see a portion of an image in more detail.
    This is useful to see the region of interest with higher clarity, like for reading text or identifying objects.
    Generally, avoid selecting a region that is too small. Ideally, aim for a region that is at least 100 x 100 pixels.
    Your crop should be close to square for best results.

    Args:
        image_name: str, the name of the image to zoom in on. Can only be called on the "input_image" image.
        bbox: list[int], the bounding box to zoom in on. The bounding box is in the format of [x_min, y_min, x_max, y_max],
            where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner.

    Returns:
        The original image, zoomed into the region specified by the bounding box with a 2.5x scale factor applied.

    Examples:
        <tool>
        name: zoom
        image_name: input_image
        bbox: [250, 100, 300, 150]
        </tool>
        <tool>
        name: zoom
        image_name: input_image
        bbox: [130, 463, 224, 556]
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

    width, height = image.size

    # validate the bounding box:
    # 1. It is a tuple of 4 integers where x_min < x_max and y_min < y_max
    # 2. The coordinates are within the image size
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("Error: Invalid bbox: must be a list of 4 integers")

    x_min, y_min, x_max, y_max = bbox

    if (
        not isinstance(x_min, int)
        or not isinstance(y_min, int)
        or not isinstance(x_max, int)
        or not isinstance(y_max, int)
    ):
        raise ValueError("Error: Invalid bbox: must be a list of 4 integers")

    if x_min < 0 or x_min >= width or x_max < 0 or x_max > width:
        raise ValueError(
            f"Error: Invalid bbox: x_min and x_max must be within the image width. x_min: {x_min}, x_max: {x_max}, width: {width}"
        )

    if y_min < 0 or y_min >= height or y_max < 0 or y_max > height:
        raise ValueError(
            f"Error: Invalid bbox: y_min and y_max must be within the image height. y_min: {y_min}, y_max: {y_max}, height: {height}"
        )

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"Error: Invalid bbox: x_min must be less than x_max and y_min must be less than y_max. x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}"
        )

    # crop the image to the bounding box
    cropped_image = image.crop(bbox)

    # Calculate target size with size constraints
    MIN_SIZE = 50
    MAX_SIZE = 400
    FIXED_MAGNIFICATION = 2.5  # Hardcoded magnification factor

    target_width = int((x_max - x_min) * FIXED_MAGNIFICATION)
    target_height = int((y_max - y_min) * FIXED_MAGNIFICATION)

    # Apply both constraints while maintaining aspect ratio
    scale_min = max(
        MIN_SIZE / target_width if target_width > 0 else 0,
        MIN_SIZE / target_height if target_height > 0 else 0,
    )  # Avoid division by zero
    scale_max = min(
        MAX_SIZE / target_width if target_width > 0 else float("inf"),
        MAX_SIZE / target_height if target_height > 0 else float("inf"),
    )  # Avoid division by zero
    # scale = max(min(scale_max, 1.0), scale_min)  # Use the most constraining scale - Incorrect logic, should just clamp between min/max scale
    scale = max(scale_min, min(scale_max, 1.0))  # Clamp the scale factor

    target_width = int(target_width * scale)
    target_height = int(target_height * scale)

    # Validate aspect ratio - Ensure target dimensions are non-zero before division
    if min(target_width, target_height) <= 0:
        raise ValueError(
            "Error: Calculated target dimensions are zero or negative after scaling."
        )
    aspect_ratio = max(target_width, target_height) / min(target_width, target_height)
    if aspect_ratio > 200:
        raise ValueError("Error: absolute aspect ratio must be smaller than 200")

    output_image = cropped_image.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )

    print(f"Zoom tool output image size: {output_image.size}")

    return output_image


def parse_zoom_args(raw_args: RawToolArgs) -> TypedToolArgs:
    """
    Parses raw string arguments for the zoom tool, focusing on type conversion.

    Expects keys: 'name', 'image_name', 'bbox'.
    Converts 'bbox' from a JSON string to a list.
    Detailed validation of values (e.g., bbox contents)
    is deferred to the zoom function itself.

    Args:
        raw_args: Dictionary with string keys and string values from the general parser.

    Returns:
        A dictionary containing the arguments with basic type conversions applied,
        ready for the zoom function. Keys: 'image_name', 'bbox'.

    Raises:
        ValueError: If required keys are missing or basic type conversion fails
                    (e.g., 'bbox' is not valid JSON).
    """
    required_keys = {"name", "image_name", "bbox"}
    actual_keys = set(raw_args.keys())

    # 1. Check for Missing Keys (Essential for parsing)
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

        # Convert bbox string to list of ints
        typed_args["bbox"] = json.loads(raw_args["bbox"])

    except json.JSONDecodeError:
        raise ValueError(f"Error: Invalid JSON format for 'bbox': '{raw_args['bbox']}'")
    except KeyError as e:
        # This should ideally be caught by the missing keys check above, but as a safeguard:
        raise ValueError(f"Error: Missing key '{e}' during conversion phase.")

    return typed_args


@pytest.fixture
def sample_image():
    # Create a larger test image to allow testing MAX_SIZE
    img = Image.new("RGB", (200, 200))  # Increased size to 200x200
    return {"input_image": img}


def test_invalid_image_name(sample_image):
    with pytest.raises(ValueError, match="not found"):
        zoom("nonexistent_image", [0, 0, 50, 50], images=sample_image)


# Add test for incorrect image name usage (e.g., "test_image" instead of "input_image")
def test_incorrect_image_name_usage(sample_image):
    with pytest.raises(ValueError, match="is not the input_image"):
        zoom(
            "test_image",
            [0, 0, 50, 50],
            images={"test_image": sample_image["input_image"]},
        )


def test_invalid_bbox_length(sample_image):
    with pytest.raises(ValueError, match="must be a list of 4 integers"):
        zoom("input_image", [0, 0, 50], images=sample_image)


def test_invalid_bbox_value_types(sample_image):
    with pytest.raises(ValueError, match="must be a list of 4 integers"):
        zoom("input_image", [0.5, 0, 50, 50], images=sample_image)


def test_bbox_coordinates_reversed(sample_image):
    with pytest.raises(ValueError, match="x_min must be less than x_max"):
        zoom("input_image", [50, 0, 10, 50], images=sample_image)
    with pytest.raises(ValueError, match="y_min must be less than y_max"):
        zoom("input_image", [0, 50, 50, 10], images=sample_image)


def test_bbox_coordinates_out_of_bounds(sample_image):
    width, height = sample_image["input_image"].size  # Now 200, 200
    with pytest.raises(ValueError, match="must be within the image"):
        zoom("input_image", [-1, 0, 50, 50], images=sample_image)
    with pytest.raises(ValueError, match="must be within the image"):
        # Use coordinate > width (200)
        zoom("input_image", [0, 0, width + 1, 50], images=sample_image)
    # Add test for y out of bounds
    with pytest.raises(ValueError, match="must be within the image"):
        zoom("input_image", [0, -1, 50, 50], images=sample_image)
    with pytest.raises(ValueError, match="must be within the image"):
        # Use coordinate > height (200)
        zoom("input_image", [0, 0, 50, height + 1], images=sample_image)


def test_basic_zoom_fixed_magnification(sample_image):
    # Test a region that would normally be small but gets magnified by 2.5x
    bbox = [20, 20, 40, 40]  # 20x20 region
    result = zoom("input_image", bbox, images=sample_image)
    assert isinstance(result, Image.Image)
    # Expected size = 20 * 2.5 = 50, 20 * 2.5 = 50. Meets MIN_SIZE=50.
    assert result.size == (50, 50)


# Renamed test to reflect fixed magnification
def test_zoom_fixed_magnification(sample_image):
    bbox = [10, 10, 50, 50]  # 40x40 region
    result = zoom("input_image", bbox, images=sample_image)
    # Expected size = 40 * 2.5 = 100, 40 * 2.5 = 100. Within MIN/MAX.
    assert result.size == (100, 100)


def test_zoom_max_size_constraint(sample_image):
    # Bbox large enough that 2.5x scaling exceeds MAX_SIZE (400)
    # Use a bbox valid for the 200x200 image
    bbox = [0, 0, 180, 180]  # 180x180 region in a 200x200 image
    result = zoom("input_image", bbox, images=sample_image)
    # Expected initial size = 180 * 2.5 = 450. Capped by MAX_SIZE=400.
    # scale_max = min(400/450, 400/450) = 0.888...
    # scale = max(scale_min_for_50, min(0.888..., 1.0)) = 0.888...
    # Final size = int(450 * 0.888...) = 400
    assert result.size == (400, 400)  # Expect size capped at MAX_SIZE
    assert max(result.size) <= 400


def test_zoom_different_regions(sample_image):
    # Test corners and center with fixed 2.5x magnification
    regions = [
        [0, 0, 20, 20],  # Top-left (20x20 -> 50x50)
        [80, 0, 99, 20],  # Top-right (19x20 -> 47.5x50 -> 50x53 (scaled by min_size_y))
        [40, 40, 60, 60],  # Center (20x20 -> 50x50)
        [
            0,
            80,
            20,
            99,
        ],  # Bottom-left (20x19 -> 50x47.5 -> 53x50 (scaled by min_size_x))
        [
            80,
            80,
            99,
            99,
        ],  # Bottom-right (19x19 -> 47.5x47.5 -> 50x50 (scaled by min_size))
    ]

    expected_sizes = [
        (50, 50),
        (
            50,
            53,
        ),  # scale_min = max(50/47.5, 50/50) = 50/47.5 = 1.0526; 47.5*1.0526=50, 50*1.0526=52.6 -> (50, 53)
        (50, 50),
        (
            53,
            50,
        ),  # scale_min = max(50/50, 50/47.5) = 50/47.5 = 1.0526; 50*1.0526=52.6, 47.5*1.0526=50 -> (53, 50)
        (
            50,
            50,
        ),  # scale_min = max(50/47.5, 50/47.5) = 1.0526; 47.5*1.0526=50 -> (50, 50)
    ]

    for i, bbox in enumerate(regions):
        result = zoom("input_image", bbox, images=sample_image)
        # Calculate expected size based on 2.5x scaling and constraints
        # width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # target_w, target_h = int(width * 2.5), int(height * 2.5)
        # scale_min = max(MIN_SIZE / target_w if target_w > 0 else 0, MIN_SIZE / target_h if target_h > 0 else 0)
        # scale_max = min(MAX_SIZE / target_w if target_w > 0 else float('inf'), MAX_SIZE / target_h if target_h > 0 else float('inf'))
        # scale = max(min(scale_max, 1.0), scale_min)
        # expected_width = int(target_w * scale)
        # expected_height = int(target_h * scale)
        assert result.size == expected_sizes[i], (
            f"Region {bbox} failed: expected {expected_sizes[i]}, got {result.size}"
        )


def test_zoom_preserve_content(sample_image):
    bbox = [20, 20, 40, 40]
    original_crop = sample_image["input_image"].crop(bbox)
    zoomed = zoom("input_image", bbox, images=sample_image)  # 20x20 -> 50x50

    # Resize original crop for comparison
    original_resized = original_crop.resize((50, 50), Image.Resampling.LANCZOS)

    # Compare a few pixels to ensure content is preserved after resize
    assert original_resized.getpixel((0, 0)) == zoomed.getpixel((0, 0))
    # Check a pixel towards the center/end
    assert original_resized.getpixel((10, 10)) == zoomed.getpixel((10, 10))
    assert original_resized.getpixel((49, 49)) == zoomed.getpixel((49, 49))


def test_zoom_aspect_ratio(sample_image):
    # Test with non-square regions using fixed 2.5x magnification
    wide_bbox = [
        20,
        20,
        60,
        30,
    ]  # Wide rectangle (40x10) -> 100x25 (below min_size Y) -> 200x50 (scaled by min_size Y)
    tall_bbox = [
        20,
        20,
        30,
        60,
    ]  # Tall rectangle (10x40) -> 25x100 (below min_size X) -> 50x200 (scaled by min_size X)

    wide_result = zoom("input_image", wide_bbox, images=sample_image)
    tall_result = zoom("input_image", tall_bbox, images=sample_image)

    # Expected sizes after applying min_size constraint while maintaining aspect ratio
    expected_wide_size = (200, 50)
    expected_tall_size = (50, 200)

    assert wide_result.size == expected_wide_size
    assert tall_result.size == expected_tall_size

    # Check if aspect ratios are preserved (approximately)
    wide_ratio = wide_result.size[0] / wide_result.size[1]
    tall_ratio = tall_result.size[0] / tall_result.size[1]

    assert abs(wide_ratio - 4) < 0.01  # Should be close to 4:1
    assert abs(tall_ratio - 0.25) < 0.01  # Should be close to 1:4
