from PIL import Image, ImageDraw
from r1_vlm.datasets.message_decoding_words_and_sequences_zoom_in.message_decoding_words_and_sequences_zoom_in import get_font

def zoom_in(full_coordinates, bbox, image_size=300) -> Image.Image:
    """
    Given the full coordinates of the decoder image, and a bounding box representing the area of the image to zoom in on,
    return the reconstructed zoomed-in image.
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
    
    # iterate over the valid full coordinates and draw them on the new image
    # note that the positions and the font sizes are adjusted and scaled up by the scale factor
    for mapping_text, (x, y, font_size) in valid_full_coordinates.items():
        # adjust the position by the scale factor
        new_x = int(x - x1) * scale_factor
        new_y = int(y - y1) * scale_factor
        new_font_size = int(font_size * scale_factor)
        font = get_font(new_font_size)
        draw.text((new_x, new_y), mapping_text, fill="black", font=font)

    return new_image