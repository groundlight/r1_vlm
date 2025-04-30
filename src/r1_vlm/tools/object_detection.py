import contextlib
import json
import os
import pickle
import time  # Import the time module
from multiprocessing import Pipe, Process

# Add imports for numpy and cv2
from dotenv import load_dotenv
from imgcat import imgcat
from PIL import Image
from tritonclient.http import InferenceServerClient

from r1_vlm.environments.tool_vision_env import RawToolArgs, TypedToolArgs

load_dotenv()

API_IP = str(os.getenv("API_IP"))
API_PORT = int(os.getenv("API_PORT"))

_object_detection_tool = None


# --- Worker Process Function ---
def _yolo_worker(conn, url, image_bytes):
    """Runs YOLO detection in a separate process, forcing CPU."""
    # --- Force CPU for this process ---
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # --- End Force CPU ---

    try:
        # --- Import YOLO *inside* the worker, *after* setting CUDA_VISIBLE_DEVICES ---
        from ultralytics import YOLO
        # Note: This will also import torch internally here in the child process
        # --- End Import ---

        t_start = time.time()

        # Deserialize image
        image = pickle.loads(image_bytes)
        t_deserialized = time.time()

        # Create transient model instance INSIDE the subprocess (will use CPU)
        model = YOLO(f"http://{url}", task="detect")  # Should now default to CPU
        t_model_created = time.time()
        result = model(image, conf=0.3)[0]
        t_inference_done = time.time()
        del model

        # Extract necessary data
        boxes = [[int(round(x)) for x in box] for box in result.boxes.xyxy.tolist()]
        labels = [result.names[int(cls)] for cls in result.boxes.cls.tolist()]
        plot_img_array = result.plot(conf=False, labels=True)  # Get plotted numpy array
        t_results_extracted = time.time()

        # Serialize results (only send data, not complex objects)
        output = {
            "boxes": boxes,
            "labels": labels,
            "plot_img_array": plot_img_array,
        }
        serialized_output = pickle.dumps(output)
        t_results_serialized = time.time()

        # Print worker timings (original)
        # print(...) # Keep this or comment out if too verbose

        t_before_send = time.time()
        print(f"  [Worker {os.getpid()} CPU] Attempting to send results...")
        conn.send(serialized_output)
        t_after_send = time.time()
        print(f"  [Worker {os.getpid()} CPU] Results sent.")

        # Print worker timings (updated with send time)
        print(
            f"  [Worker {os.getpid()} CPU] Timings (s): "
            f"Deserialize: {t_deserialized - t_start:.4f}, "
            f"ModelCreate: {t_model_created - t_deserialized:.4f}, "
            f"Inference: {t_inference_done - t_model_created:.4f}, "
            f"Extract: {t_results_extracted - t_inference_done:.4f}, "
            f"Serialize: {t_results_serialized - t_results_extracted:.4f}, "
            f"Send: {t_after_send - t_before_send:.4f}, "  # Added Send time
            f"Total: {t_after_send - t_start:.4f}"  # Use t_after_send for total now
        )

    except Exception as e:
        # Send back exception info if something goes wrong
        print(f"YOLO Worker Error: {e}")  # Log error in worker
        import traceback

        traceback.print_exc()
        conn.send(pickle.dumps(e))
    finally:
        conn.close()


# --- End Worker Process Function ---


class ObjectDetectionTool:
    def __init__(self):
        self.url = f"{API_IP}:{API_PORT}/yolo"
        # Keep triton client for readiness check, but not for inference here
        self.triton_client = InferenceServerClient(
            url=self.url, verbose=False, ssl=False
        )

        # Wait until model is ready
        for _ in range(10):
            with contextlib.suppress(Exception):
                assert self.triton_client.is_model_ready("yolo")
                break
            time.sleep(1)

    def detect_objects(self, image: Image.Image) -> dict:
        """Performs object detection using a separate worker process."""
        t_parent_start = time.time()
        parent_conn, child_conn = Pipe()

        # Serialize image data
        try:
            image_bytes = pickle.dumps(image)
        except Exception as e:
            raise RuntimeError(f"Failed to pickle input image: {e}") from e

        proc = None  # Initialize proc to None
        t_process_start = 0.0
        t_process_end = 0.0
        try:
            # Create and start the worker process
            proc = Process(
                target=_yolo_worker, args=(child_conn, self.url, image_bytes)
            )
            t_process_start = time.time()
            proc.start()
            child_conn.close()  # Close child end in parent immediately after start

            # Wait for result from worker using poll (with timeout)
            if parent_conn.poll(timeout=60.0):  # Wait up to 60 seconds
                result_bytes = parent_conn.recv()
                t_process_end = time.time()  # Record time when result received
            else:
                t_process_end = time.time()  # Record time even on timeout
                raise TimeoutError("YOLO worker process timed out waiting for result.")

        except EOFError:  # Handle case where child exits unexpectedly before sending
            result_bytes = pickle.dumps(
                RuntimeError("YOLO worker process exited before sending results.")
            )
        except Exception as e:  # Catch other potential errors during process management
            result_bytes = pickle.dumps(
                RuntimeError(f"Error managing YOLO worker process: {e}")
            )
        finally:
            # Ensure process cleanup
            if proc is not None:
                proc.join(timeout=5.0)  # Short wait for graceful exit
                if proc.is_alive():
                    print("Warning: YOLO worker process unresponsive, terminating.")
                    proc.terminate()
                    proc.join(timeout=1.0)  # Wait after terminate
            # Ensure parent connection is closed
            if "parent_conn" in locals() and not parent_conn.closed:
                parent_conn.close()
            # Ensure child connection is closed (belt and suspenders)
            if "child_conn" in locals() and not child_conn.closed:
                child_conn.close()

        # Deserialize result
        try:
            result_data = pickle.loads(result_bytes)
        except Exception as e:
            # If deserialization fails, result_bytes might contain partial/error data
            raise RuntimeError(
                f"Failed to deserialize result from YOLO worker. Raw data: {result_bytes!r}. Error: {e}"
            ) from e

        # Re-raise exception if worker sent one
        if isinstance(result_data, Exception):
            raise RuntimeError("YOLO worker process failed") from result_data

        # Process results back into the expected format
        boxes = result_data["boxes"]
        labels = result_data["labels"]
        plot_img_array = result_data["plot_img_array"]

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
            # Convert plotted numpy array back to PIL Image
            annotated_image = Image.fromarray(
                plot_img_array[..., ::-1]
            )  # BGR->RGB for PIL

        t_parent_end = time.time()
        # Print parent timings
        process_duration = (
            t_process_end - t_process_start
            if t_process_start > 0 and t_process_end > 0
            else -1.0
        )
        print(
            f"[Parent {os.getpid()}] Subprocess call duration: {process_duration:.3f} s, "
            f"Total detect_objects duration: {t_parent_end - t_parent_start:.3f} s"
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
