# generic zoom tool
import pytest
from PIL import Image


def zoom(
    image_name: str,
    bbox: list[int],
    magnification: float = 1.0,
    **kwargs,
) -> Image.Image:
    """
    Returns the original image, cropped into the region specified by the bounding box, and then resized to the specified magnification.
    This tool is useful to see a portion of an image in more detail. Generally, avoid selecting a region that is too small.
    Ideally, aim for a region that is at least 100 x 100 pixels.

    Args:
        image_name: str, the name of the image to zoom in on. Can only be called on the "input_image" image.
        bbox: list[int], the bounding box to zoom in on. The bounding box is in the format of [x_min, y_min, x_max, y_max],
            where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner.
        magnification: float, the magnification factor. Must be equal to or greater than 1.0.

    Returns:
        The original image, zoomed into the region specified by the bounding box.

    Examples:
        <tool>{"name": "zoom", "args": {"image_name": "input_image", "bbox": [250, 100, 300, 150], "magnification": 1.0}}</tool>
        <tool>{"name": "zoom", "args": {"image_name": "input_image", "bbox": [130, 463, 224, 556], "magnification": 2.4}}</tool>
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

    # validate the magnification
    if not isinstance(magnification, (float, int)) or float(magnification) < 1.0:
        raise ValueError(
            "Error: Invalid magnification: must be a float greater than or equal to 1.0"
        )

    # crop the image to the bounding box
    cropped_image = image.crop(bbox)

    # Calculate target size with size constraints
    MIN_SIZE = 50
    MAX_SIZE = 300

    target_width = int((x_max - x_min) * magnification)
    target_height = int((y_max - y_min) * magnification)

    # Apply both constraints while maintaining aspect ratio
    scale_min = max(MIN_SIZE / target_width, MIN_SIZE / target_height)
    scale_max = min(MAX_SIZE / target_width, MAX_SIZE / target_height)
    scale = max(min(scale_max, 1.0), scale_min)  # Use the most constraining scale

    target_width = int(target_width * scale)
    target_height = int(target_height * scale)

    # Validate aspect ratio
    aspect_ratio = max(target_width, target_height) / min(target_width, target_height)
    if aspect_ratio > 200:
        raise ValueError("Error: absolute aspect ratio must be smaller than 200")

    output_image = cropped_image.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )

    print(f"Zoom tool output image size: {output_image.size}")

    return output_image


@pytest.fixture
def sample_image():
    # Create a simple 100x100 test image
    img = Image.new("RGB", (100, 100))
    return {"test_image": img}


def test_invalid_image_name(sample_image):
    with pytest.raises(ValueError, match="not found"):
        zoom("nonexistent_image", [0, 0, 50, 50], 1.0, images=sample_image)


def test_invalid_bbox_length(sample_image):
    with pytest.raises(ValueError, match="must be a list of 4 integers"):
        zoom("test_image", [0, 0, 50], 1.0, images=sample_image)


def test_invalid_bbox_value_types(sample_image):
    with pytest.raises(ValueError, match="must be a list of 4 integers"):
        zoom("test_image", [0.5, 0, 50, 50], 1.0, images=sample_image)


def test_bbox_coordinates_reversed(sample_image):
    with pytest.raises(ValueError, match="x_min must be less than x_max"):
        zoom("test_image", [50, 0, 10, 50], 1.0, images=sample_image)
    with pytest.raises(ValueError, match="y_min must be less than y_max"):
        zoom("test_image", [0, 50, 50, 10], 1.0, images=sample_image)


def test_bbox_coordinates_out_of_bounds(sample_image):
    with pytest.raises(ValueError, match="must be within the image"):
        zoom("test_image", [-1, 0, 50, 50], 1.0, images=sample_image)
    with pytest.raises(ValueError, match="must be within the image"):
        zoom("test_image", [0, 0, 101, 50], 1.0, images=sample_image)


def test_invalid_magnification_type(sample_image):
    with pytest.raises(ValueError, match="must be a float"):
        zoom("test_image", [0, 0, 50, 50], "1.0", images=sample_image)


def test_invalid_magnification_value(sample_image):
    with pytest.raises(
        ValueError, match="must be a float greater than or equal to 1.0"
    ):
        zoom("test_image", [0, 0, 50, 50], 0.5, images=sample_image)


def test_basic_zoom_no_magnification(sample_image):
    result = zoom("test_image", [20, 20, 40, 40], 1.0, images=sample_image)
    assert isinstance(result, Image.Image)
    assert result.size == (28, 28)  # Should match minimum size constraint


def test_zoom_with_magnification(sample_image):
    result = zoom("test_image", [20, 20, 40, 40], 2.0, images=sample_image)
    assert result.size == (40, 40)  # Should be 2x the bbox dimensions


def test_zoom_max_size_constraint(sample_image):
    # Try to zoom beyond MAX_SIZE (300x300)
    result = zoom("test_image", [0, 0, 80, 80], 4.0, images=sample_image)
    assert max(result.size) <= 300  # Should be constrained to MAX_SIZE


def test_zoom_different_regions(sample_image):
    # Test corners and center
    regions = [
        [0, 0, 20, 20],  # Top-left
        [80, 0, 99, 20],  # Top-right
        [40, 40, 60, 60],  # Center
        [0, 80, 20, 99],  # Bottom-left
        [80, 80, 99, 99],  # Bottom-right
    ]

    for bbox in regions:
        result = zoom("test_image", bbox, 1.5, images=sample_image)
        expected_width = int(min((bbox[2] - bbox[0]) * 1.5, 300))
        expected_height = int(min((bbox[3] - bbox[1]) * 1.5, 300))
        assert result.size == (expected_width, expected_height)


def test_zoom_preserve_content(sample_image):
    bbox = [20, 20, 40, 40]
    original = sample_image["test_image"].crop(bbox)
    zoomed = zoom("test_image", bbox, 1.0, images=sample_image)

    # Compare a few pixels to ensure content is preserved
    assert original.getpixel((0, 0)) == zoomed.getpixel((0, 0))
    assert original.getpixel((10, 10)) == zoomed.getpixel((10, 10))


def test_zoom_aspect_ratio(sample_image):
    # Test with non-square regions
    wide_bbox = [20, 20, 60, 30]  # Wide rectangle (40x10)
    tall_bbox = [20, 20, 30, 60]  # Tall rectangle (10x40)

    wide_result = zoom("test_image", wide_bbox, 1.5, images=sample_image)
    tall_result = zoom("test_image", tall_bbox, 1.5, images=sample_image)

    # Check if aspect ratios are preserved
    wide_ratio = wide_result.size[0] / wide_result.size[1]
    tall_ratio = tall_result.size[0] / tall_result.size[1]

    assert abs(wide_ratio - 4) < 0.1  # Should be close to 4:1
    assert abs(tall_ratio - 0.25) < 0.1  # Should be close to 1:4
