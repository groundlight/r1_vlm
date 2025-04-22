import base64
import io
import os

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

BUMBERSHOOT2_IP = str(os.getenv("BUMBERSHOOT2_IP"))
BUMBERSHOOT2_PORT = int(os.getenv("BUMBERSHOOT2_PORT"))


def detect_objects(image_name: str, classes: list[str], confidence: float, **kwargs) -> tuple[list[dict], Image.Image]:
    """
    Calls an open vocabulary object detection model on the image. Filters to detections with confidence greater than or equal to the specified confidence threshold.
    
    Args:
        image_name: str, the name of the image to detect objects in.
        classes: list[str], the classes to detect. As the model is open vocabulary, your classes can be any string, even referring phrases about the scene, like "the man in the red shirt" or "the dog on the left".
        confidence: float, the confidence threshold for the detections.

    Returns:
        1. A list of dictionaries, each containing the following keys:
            - "bbox_2d": list[int], the bounding box of the object in the format of [x_min, y_min, x_max, y_max] in pixel coordinates
            - "label": str, the label of the object
            - "confidence": float, the confidence score of the detection
        2. The original image with the detections overlaid on it.
    """
    
    images = kwargs["images"]
    image = images.get(image_name, None)
    if image is None:
        valid_image_names = list(images.keys())
        raise ValueError(
            f"Error: Image {image_name} not found. Valid image names are: {valid_image_names}"
        )
    
    # construct the API request
    url = f"http://{BUMBERSHOOT2_IP}:{BUMBERSHOOT2_PORT}/detect?confidence={confidence}"
    
    files = {"image": image}
    data = {}
    for c in classes:
        data.setdefault("classes", []).append(c)
    
    # send the request
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
    else:
        raise Exception(f"Error: API request failed with status code {response.status_code}")
    
    detections = result["results"]["detections"]
    
    dets = []
    for detection in detections:
        dets.append({
            "bbox_2d": detection["bbox_2d"],
            "label": detection["label"],
            "confidence": detection["confidence"]
        })
    
    # convert the annotated image(base64 encoded) to a PIL Image
    annotated_image_data = base64.b64decode(result["annotated_image"])
    annotated_image = Image.open(io.BytesIO(annotated_image_data))
    
    
    return {"text_data": dets, "image_data": annotated_image}
    

    
    

