import contextlib
import json
import os
import time  # Import the time module

# Add imports for numpy and cv2
from dotenv import load_dotenv
from imgcat import imgcat
from PIL import Image
from tritonclient.http import InferenceServerClient
from ultralytics import YOLO

from r1_vlm.environments.tool_vision_env import RawToolArgs, TypedToolArgs

load_dotenv()

API_IP = str(os.getenv("API_IP"))
API_PORT = int(os.getenv("API_PORT"))

_object_detection_tool = None


class ObjectDetectionTool:
    def __init__(self):
        url = f"{API_IP}:{API_PORT}/yolo"
        self.triton_client = InferenceServerClient(url=url, verbose=False, ssl=False)

        # Wait until model is ready
        for _ in range(10):
            with contextlib.suppress(Exception):
                assert self.triton_client.is_model_ready("yolo")
                break
            time.sleep(1)

        self.model = YOLO(f"http://{url}", task="detect")

    def detect_objects(self, image: Image.Image) -> list[dict]:
        result = self.model(image, conf=0.3)[0]
        boxes = [[int(round(x)) for x in box] for box in result.boxes.xyxy.tolist()]
        labels = [result.names[int(cls)] for cls in result.boxes.cls.tolist()]

        detections = [
            {"bbox_2d": box, "label": label} for box, label in zip(boxes, labels)
        ]

        if len(detections) == 0:
            dets_string = "No objects detected."
            annotated_image = None
        else:
            dets_string = ""
            for index, det in enumerate(detections):
                dets_string += f"{index + 1}. {det}"

                if index < len(detections) - 1:
                    dets_string += "\n"

            annotated_image = result.plot(conf=False, labels=True)

        return {"text_data": dets_string, "image_data": annotated_image}


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

    return _object_detection_tool.detect_objects(image)


def parse_detect_objects_args(raw_args: RawToolArgs) -> TypedToolArgs:
    """
    Parses raw string arguments for the detect_objects tool, focusing on type conversion.

    Expects keys: 'name', 'image_name'
    Detailed validation of values (e.g., 'image_name' validity)
    is deferred to the detect_objects function itself.

    Args:
        raw_args: Dictionary with string keys and string values from the general parser.

    Returns:
        A dictionary containing the arguments with basic type conversions applied,
        ready for the detect_objects function. Keys: 'image_name'.

    Raises:
        ValueError: If required keys are missing or basic type conversion fails
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

    # 3. Perform Basic Type Conversions
    typed_args: TypedToolArgs = {}
    try:
        # Keep image_name as string
        typed_args["image_name"] = raw_args["image_name"]

    except json.JSONDecodeError:
        raise ValueError(
            f"Error: Invalid JSON format for 'classes': '{raw_args['classes']}'"
        )
    except ValueError as e:
        # Catch the list type error from above
        raise ValueError(f"Error: processing 'classes': {e}")
    except KeyError as e:
        # Safeguard for missing keys during access
        raise ValueError(f"Error: Missing key '{e}' during conversion phase.")

    return typed_args


if __name__ == "__main__":
    tool = ObjectDetectionTool()
    set_object_detection_tool(tool)
    image = Image.open(
        "/millcreek/home/sunil/r1_vlm_bumbershoot0/r1_vlm/src/r1_vlm/tools/cars.jpeg"
    )
    for i in range(10):
        detections = detect_objects(
            image_name="input_image", images={"input_image": image}
        )

        image_data = detections["image_data"]
        imgcat(image_data)
        text_data = detections["text_data"]
        print(text_data)
