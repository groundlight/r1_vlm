import base64
import io
import os

# Add imports for numpy and cv2
import cv2
import numpy as np
import pytest
import requests
from dotenv import load_dotenv
from imgcat import imgcat
from PIL import Image

load_dotenv()

API_IP = str(os.getenv("API_IP"))
API_PORT = int(os.getenv("API_PORT"))


def detect_objects(image_name: str, classes: list[str], **kwargs) -> tuple[list[dict], Image.Image]:
    """
    Calls an open vocabulary object detection model on the image. Useful for localizing objects in an image or determining if an object is present.
    
    Args:
        image_name: str, the name of the image to detect objects in. Can only be called on the "input_image" image.
        classes: list[str], the classes to detect. As the model is open vocabulary, your classes can be any string, even referring phrases about the scene, like "the man in the red shirt" or "the dog on the left".

    Returns:
        1. A list of dictionaries, each containing the following keys:
            - "bbox_2d": list[int], the bounding box of the object in the format of [x_min, y_min, x_max, y_max] in pixel coordinates
            - "label": str, the label of the object
            - "confidence": float, the confidence score of the detection
        2. The original image with the detections overlaid on it.
    
    Examples:
        <tool>{"name": "detect_objects", "args": {"image_name": "input_image", "classes": ["car", "person on the sidewalk"]}}</tool>
        <tool>{"name": "detect_objects", "args": {"image_name": "input_image", "classes": ["elephant on the right", "white jeep"]}}</tool>
    """
    
    images = kwargs["images"]
    image = images.get(image_name, None)
    if image is None:
        valid_image_names = list(images.keys())
        raise ValueError(
            f"Error: Image {image_name} not found. Valid image names are: {valid_image_names}"
        )
    
    # only allow the input_image to be used, as the model tends to call this tool on very small zooms, which is not helpful
    if image_name != "input_image":
        raise ValueError(f"Error: Image {image_name} is not the input_image. This tool can only be called on the input_image.")
    
    # construct the API request
    # I decided to fix the confidence threshold at 0.25, as the model tends to set this value very high, which leads to a lot of false negatives
    url = f"http://{API_IP}:{API_PORT}/detect?confidence={0.25}"
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    files = {"image": img_byte_arr}
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
            "confidence": round(detection["confidence"], 2)
        })
    
    if len(dets) == 0:
        dets_string = "No objects detected."
    else:
        dets_string = ""
        for index, det in enumerate(dets):
            dets_string += f"{index+1}. {det}"
        
            if index < len(dets) - 1:
                dets_string += "\n"
        
    
    # convert the annotated image(base64 encoded) to a PIL Image
    annotated_image_data = base64.b64decode(result["annotated_image"])
    annotated_image_pil = Image.open(io.BytesIO(annotated_image_data))

    # Convert PIL Image to NumPy array (OpenCV format)
    # PIL images with mode 'RGB' are loaded as NumPy arrays with shape (H, W, 3) in RGB order.
    # PIL images with mode 'RGBA' are loaded as NumPy arrays with shape (H, W, 4) in RGBA order.
    annotated_image_np = np.array(annotated_image_pil)

    # Convert BGR(A) to RGB(A) using OpenCV if it's a color image
    # Assuming the source API sent BGR/BGRA data, which np.array converted retaining channel order relative to PIL's interpretation.
    # If PIL interpreted as RGB, the np array is RGB. If RGBA, the np array is RGBA.
    # Since the *source* was BGR/BGRA, we convert the numpy array from BGR/BGRA to RGB/RGBA.
    if annotated_image_np.ndim == 3 and annotated_image_np.shape[2] == 3: # RGB/BGR
        annotated_image_np_rgb = cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB)
    elif annotated_image_np.ndim == 3 and annotated_image_np.shape[2] == 4: # RGBA/BGRA
        annotated_image_np_rgb = cv2.cvtColor(annotated_image_np, cv2.COLOR_BGRA2RGBA)
    else:
        # Grayscale or other formats, no conversion needed
        annotated_image_np_rgb = annotated_image_np

    # Convert NumPy array back to PIL Image
    annotated_image = Image.fromarray(annotated_image_np_rgb)
    
    
    return {"text_data": dets_string, "image_data": annotated_image}
    

@pytest.fixture
def sample_image_fixture():
    """Provides a simple dummy image for testing."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img = Image.open(os.path.join(current_dir, "cars.jpeg"))
    return {"input_image": img}


def test_basic_detection_integration(sample_image_fixture):
    """Tests basic object detection call against the running API."""
    # Call the function under test - this will make a real HTTP request
    # Using classes unlikely to be in a plain red image might be safer
    # depending on the actual model behavior. Let's use "object".
    try:
        result = detect_objects(
            image_name="input_image",
            # there should be cars, but no dogs
            classes=["car", "dog"], 
            images=sample_image_fixture,
        )

       
        assert isinstance(result, dict)
        assert "text_data" in result
        assert "image_data" in result
        assert isinstance(result["text_data"], str)
        assert isinstance(result["image_data"], Image.Image)
        
        # visualize the annotated image
        annotated_image = result["image_data"]
        imgcat(annotated_image)
        
        # visualize the text data
        print(result["text_data"])

    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"API connection failed. Is the server running at http://{API_IP}:{API_PORT}? Error: {e}")
    except Exception as e:
        # Catch other potential errors during the API call or processing
        pytest.fail(f"An unexpected error occurred: {e}")


    


