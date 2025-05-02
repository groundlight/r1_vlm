import base64  # For encoding/decoding images
import io  # For handling image bytes
import json
import os
import time

import requests  # To make HTTP requests to the API server

# Remove multiprocessing imports
# from multiprocessing import Pipe, Process
# Remove YOLO import from here
# from ultralytics import YOLO
# Add imports for numpy and cv2 (if still needed for other parts, unlikely now)
from dotenv import load_dotenv
from PIL import Image

from r1_vlm.environments.tool_vision_env import RawToolArgs, TypedToolArgs

load_dotenv()

# --- Configuration for the Detection API Server ---
# Get the API server's URL from environment variables, default to localhost:8001
DETECTION_API_HOST = os.getenv("DETECTION_API_HOST", "localhost")
DETECTION_API_PORT = int(os.getenv("DETECTION_API_PORT", 8001))
DETECTION_API_URL = f"http://{DETECTION_API_HOST}:{DETECTION_API_PORT}/detect"
# --- End Configuration ---

_object_detection_tool = None


class ObjectDetectionTool:
    def __init__(self):
        # Store the URL for the detection API server
        self.api_url = DETECTION_API_URL

    def detect_objects(self, image: Image.Image) -> dict:
        """Sends image to detection API server and returns results."""
        t_client_start = time.time()
        annotated_image = None  # Default
        dets_string = "Error: Detection failed."  # Default error message

        try:
            # 1. Prepare Image for Sending
            buffer = io.BytesIO()
            # Save image to buffer in a common format like PNG
            image.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            t_encoded = time.time()

            # 2. Prepare Request Payload
            payload = {"image_base64": img_base64}

            # 3. Call the API Server
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60.0,  # Set a reasonable timeout (e.g., 60 seconds)
            )
            t_responded = time.time()

            # 4. Process Response
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    dets_string = response_data.get(
                        "text_data", "Error: Missing text_data in response."
                    )
                    image_data_base64 = response_data.get("image_data_base64")

                    if image_data_base64:
                        try:
                            annotated_bytes = base64.b64decode(image_data_base64)
                            annotated_image = Image.open(io.BytesIO(annotated_bytes))
                        except Exception as img_err:
                            raise ValueError(
                                f"Failed to decode/load annotated image from response: {img_err}"
                            )
                            # Keep annotated_image as None

                except json.JSONDecodeError as json_err:
                    raise ValueError(
                        f"Failed to decode JSON response from API: {json_err}"
                    )

                except Exception as proc_err:  # Catch other errors processing response
                    raise ValueError(
                        f"Error processing successful API response: {proc_err}"
                    )

            else:
                # Handle HTTP errors
                error_msg = f"Error from detection API: {response.status_code}"
                try:
                    error_detail = response.json().get("detail", response.text)
                    error_msg += f" - {error_detail}"
                except json.JSONDecodeError:
                    error_msg += f" - {response.text}"
                raise ValueError(error_msg)

        except requests.exceptions.Timeout:
            raise ValueError("Request to detection API timed out after 60s.")
        except requests.exceptions.RequestException as req_err:
            raise ValueError(f"Request to detection API failed: {req_err}")

        except Exception as e:
            # Catch-all for other unexpected errors in the client logic
            raise ValueError(
                f"Unexpected error in detect_objects client: {e}", exc_info=True
            )

        t_client_end = time.time()
        print(
            f"detect_objects client timings (s): "
            f"Encode: {t_encoded - t_client_start:.3f}, "
            f"API Call: {t_responded - t_encoded:.3f}, "
            f"Decode/Process: {t_client_end - t_responded:.3f}, "
            f"Total: {t_client_end - t_client_start:.3f}"
        )

        return {"text_data": dets_string, "image_data": annotated_image}

    def __del__(self):
        """Cleanup method - nothing persistent to clean up"""
        pass


def set_object_detection_tool(tool: ObjectDetectionTool):
    global _object_detection_tool
    _object_detection_tool = tool


def detect_objects(image_name: str, **kwargs) -> tuple[list[dict], Image.Image]:
    """
    Calls an object detection model on the image. Useful for localizing objects in an image or determining if an object is present.

    Args:
        image_name: str, the name of the image to detect objects in. Can only be called on the "input_image" image.

    Returns:
        1. A list of dictionaries, each containing the following keys:
            - "bbox_2d": list[int], the bounding box of the object in the format of [x_min, y_min, x_max, y_max] in pixel coordinates
            - "label": str, the label of the object
        2. The original image with the detections overlaid on it.

    Examples:
        <tool>
        name: detect_objects
        image_name: input_image
        </tool>
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
        raise ValueError(
            f"Error: Image {image_name} is not the input_image. This tool can only be called on the input_image."
        )

    if _object_detection_tool is None:
        raise RuntimeError(
            "ObjectDetectionTool not initialized. Call set_object_detection_tool first."
        )

    # Call the method which now calls the API
    return _object_detection_tool.detect_objects(image)  # Return type is now dict


def parse_detect_objects_args(raw_args: RawToolArgs) -> TypedToolArgs:
    """
    Parses raw string arguments for the detect_objects tool.

    Expects keys: 'name', 'image_name'
    Detailed validation of values (e.g., 'image_name' validity)
    is deferred to the detect_objects function itself.

    Args:
        raw_args: Dictionary with string keys and string values from the general parser.

    Returns:
        A dictionary containing the arguments. Keys: 'image_name'.

    Raises:
        ValueError: If required keys are missing or extra keys are present.
    """
    required_keys = {"name", "image_name"}
    actual_keys = set(raw_args.keys())

    # 1. Check for Missing Keys
    missing_keys = required_keys - actual_keys
    if missing_keys:
        raise ValueError(
            f"Error: Missing required arguments for detect_objects tool: {', '.join(sorted(missing_keys))}"
        )

    # 2. Check for extra keys
    extra_keys = actual_keys - required_keys
    if extra_keys:
        raise ValueError(
            f"Error: Unexpected arguments for detect_objects tool: {', '.join(sorted(extra_keys))}"
        )

    # 3. Prepare typed args (only image_name needed)
    typed_args: TypedToolArgs = {}
    try:
        # Keep image_name as string
        typed_args["image_name"] = raw_args["image_name"]

    except ValueError as e:
        # Catch the list type error from above
        raise ValueError(f"Error: processing 'classes': {e}")
    except KeyError as e:
        # Safeguard for missing keys during access
        raise ValueError(f"Error: Missing key '{e}' during conversion phase.")

    return typed_args
