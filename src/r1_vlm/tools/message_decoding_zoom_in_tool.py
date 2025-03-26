from tqdm import tqdm
from datasets import Dataset
from PIL import Image, ImageDraw
from r1_vlm.datasets.message_decoding_words_and_sequences_zoom_in.message_decoding_words_and_sequences_zoom_in import get_font
from r1_vlm.tools.digits_answer_tool import ImageHashTableTool


class ImageHashZoomInTool(ImageHashTableTool):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        
        self.build_hash_table(dataset)
        
    def build_hash_table(self, dataset: Dataset) -> None:
        for example in tqdm(dataset, desc="Building hash table"):
            image = example["image"]
            full_coordinates = example["full_coordinates"]

            assert isinstance(image, Image.Image)
            assert isinstance(full_coordinates, dict)

            valid_coordinates = {k: v for k, v in full_coordinates.items() if v is not None}
            self.add_image(image, {"coordinates": valid_coordinates})

def coordinates_based_zoom_in(full_coordinates, bbox, image_size=300) -> tuple[Image.Image, dict[str, tuple[int, int, int]]]:
    """
    Given the full coordinates of the decoder image, and a bounding box representing the area of the image to zoom in on,
    return the reconstructed zoomed-in image and the zoomed-in full coordinates.
    Since we want the zoomed-in image to be in high resolution, we need to build the image up from the full coordinates, rather than cropping the image.
    """
    # get the coordinates of the bounding box
    x1, y1, x2, y2 = bbox
    valid_full_coordinates = {k: v for k, v in full_coordinates.items() if v is not None}
    shorter_side = min(x2 - x1, y2 - y1)
    scale_factor = image_size / shorter_side
    new_image_size = (int((x2 - x1) * scale_factor), int((y2 - y1) * scale_factor))

    # create the new image
    new_image = Image.new("RGB", new_image_size, "white")
    draw = ImageDraw.Draw(new_image)
    
    zoomed_in_full_coordinates = {}
    # iterate over the valid full coordinates and draw them on the new image
    # note that the positions and the font sizes are adjusted and scaled up by the scale factor
    for mapping_text, (x, y, font_size) in valid_full_coordinates.items():
        # adjust the position by the scale factor
        new_x = int(x - x1) * scale_factor
        new_y = int(y - y1) * scale_factor
        new_font_size = int(font_size * scale_factor)
        font = get_font(new_font_size)
        draw.text((new_x, new_y), mapping_text, fill="black", font=font)
        zoomed_in_full_coordinates[mapping_text] = (new_x, new_y, new_font_size)
    return new_image, zoomed_in_full_coordinates

# Make zoom_in_tool accessible to zoom_in
_zoom_in_tool = None

def set_zoom_in_tool(tool: ImageHashZoomInTool):
    global _zoom_in_tool
    _zoom_in_tool = tool
    
def zoom_in(image_name: str, bbox: tuple[float, float, float, float], **kwargs) -> Image.Image:
    '''
    Returns the zoomed-in image given the image and the bounding box to zoom in on.

    Args:
        image_name: str, the name of the image to zoom in on.
        bbox: tuple[float, float, float, float], the bounding box to zoom in on. The bounding box is in the format of [x1, y1, x2, y2],
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. The coordinates are normalized to the image size and range from 0 to 1.
            Also note that a valid bbox should have x1 < x2 and y1 < y2.

    Returns:
        The zoomed-in image.

    Examples:
        <tool>{"name": "zoom_in", "args": {"image_name": "input_image", "bbox": [0.25, 0.30, 0.45, 0.40]}}</tool>
        <tool>{"name": "zoom_in", "args": {"image_name": "input_image", "bbox": [0.80, 0.10, 0.95, 0.25]}}</tool>

    '''
    if _zoom_in_tool is None:
        raise ValueError("ZoomInTool not initialized. Call set_zoom_in_tool first.")

    coordinates_names = ["x1", "y1", "x2", "y2"]
    # check each coordinate and see if it's in the range of 0 to 1
    invalid_bbox_coordinates = []
    for coordinate_name, coordinate_value in zip(coordinates_names, bbox):
        if coordinate_value < 0 or coordinate_value > 1:
            invalid_bbox_coordinates.append(coordinate_name)

    if len(invalid_bbox_coordinates) > 0:
        raise ValueError(f"Invalid bbox coordinates: {invalid_bbox_coordinates}. The coordinates should be normalized to the image size and range from 0 to 1.")

    if bbox[0] >= bbox[2]:
        raise ValueError("Invalid bbox coordinates: x1 should be less than x2.")

    if bbox[1] >= bbox[3]:
        raise ValueError("Invalid bbox coordinates: y1 should be less than y2.")
    
    images = kwargs["images"]

    image_to_use = images.get(image_name, None)

    if image_to_use is None:
        raise ValueError(f"Error: Image {image_name} not found. Valid image names are: {images.keys()}")
    
    coordinates = _zoom_in_tool.lookup_image(image_to_use)["coordinates"]

    # convert the bbox from the normalized format to the absolute format
    bbox = (bbox[0] * image_to_use.width, bbox[1] * image_to_use.height, bbox[2] * image_to_use.width, bbox[3] * image_to_use.height)

    zoomed_in_image, zoomed_in_full_coordinates = coordinates_based_zoom_in(coordinates, bbox, image_size=image_to_use.width)

    # add the zoomed-in full coordinates and the zooomed-in image to the hash table in the tool
    _zoom_in_tool.add_image(zoomed_in_image, {"coordinates": zoomed_in_full_coordinates})

    return zoomed_in_image