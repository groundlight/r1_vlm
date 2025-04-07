from PIL import Image, ImageDraw


def display_bounding_box(image_name: str, bbox: list[int], **kwargs) -> Image.Image:
    '''
    Displays an image with a bounding box overlaid on it.
    
        Args:
        image_name: str, the name of the image to zoom in on.
        bbox: list[int], the bounding box to zoom in on. The bounding box is in the format of [x_min, y_min, x_max, y_max],
            where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner.
            
        Returns:
            The original image with the bounding box overlaid on it, in red.
            
        Examples:
            <tool>{"name": "display_bounding_box", "args": {"image_name": "input_image", "bbox": [250, 100, 300, 150]}}</tool>
            <tool>{"name": "display_bounding_box", "args": {"image_name": "tool_result_1", "bbox": [130, 463, 224, 556]}}</tool>
    '''
    images = kwargs["images"]
    
    image = images.get(image_name, None)
    
    if image is None:
        valid_image_names = list(images.keys())
        raise ValueError(
            f"Error: Image {image_name} not found. Valid image names are: {valid_image_names}"
        )
        
    # we must make a copy to avoid modifying the original image
    image = image.copy()
    
    
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
        
    # draw the bounding box
    draw = ImageDraw.Draw(image)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    
    return image