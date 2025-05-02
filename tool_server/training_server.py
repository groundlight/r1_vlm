import base64
import io
import logging
import os
import time
from typing import Optional

# --- Force CPU Usage for this Server ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)  # Get logger early for info message
logger.info("CUDA_VISIBLE_DEVICES set to -1. Server will attempt to use CPU for YOLO.")
# --- End Force CPU ---

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# --- Configuration & Initialization ---

# Load environment variables (pointing to the *actual* YOLO/Triton backend)
load_dotenv()
API_IP = str(os.getenv("API_IP"))
API_PORT = int(os.getenv("API_PORT"))
BACKEND_URL = f"http://{API_IP}:{API_PORT}/yolo"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Global variable to hold the loaded YOLO model
yolo_model: Optional[YOLO] = None

# --- Pydantic Models ---


class DetectionRequest(BaseModel):
    """Request body for the /detect endpoint."""

    image_base64: str


class DetectionResponse(BaseModel):
    """Successful response body for the /detect endpoint."""

    text_data: str
    image_data_base64: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str


# --- FastAPI App ---

app = FastAPI(title="YOLO Detection API Server")


@app.on_event("startup")
async def startup_event():
    """Load the YOLO model on server startup (will use CPU)."""
    global yolo_model
    logger.info(
        f"Attempting to load YOLO model targeting backend: {BACKEND_URL} (Forcing CPU)"
    )  # Added CPU note
    start_time = time.time()
    try:
        # Model will initialize on CPU due to CUDA_VISIBLE_DEVICES=-1
        yolo_model = YOLO(BACKEND_URL, task="detect")
        # Perform a dummy inference to ensure connection/initialization on CPU
        dummy_img = Image.new("RGB", (64, 64), color="red")
        _ = yolo_model(dummy_img, verbose=False)
        end_time = time.time()
        logger.info(
            f"YOLO model loaded successfully on CPU in {end_time - start_time:.2f} seconds."  # Added CPU note
        )
    except Exception as e:
        logger.error(f"Failed to load YOLO model on startup: {e}", exc_info=True)
        yolo_model = None


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Perform object detection on an image",
)
async def detect_objects_api(request: DetectionRequest):
    """
    Accepts a base64 encoded image, performs YOLO detection using the
    pre-loaded model targeting the backend service, and returns results.
    """
    global yolo_model
    if yolo_model is None:
        logger.error("YOLO model is not loaded. Cannot process request.")
        raise HTTPException(
            status_code=503, detail="Model service unavailable"
        )  # 503 Service Unavailable

    logger.info("Received detection request.")
    t_start = time.time()

    try:
        # 1. Decode Base64 Image
        try:
            image_bytes = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure RGB
        except Exception as e:
            logger.error(f"Failed to decode/load image from base64: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        t_decoded = time.time()

        # 2. Run YOLO Inference
        try:
            # Use the globally loaded model
            result = yolo_model(image, conf=0.3, verbose=False)[
                0
            ]  # verbose=False is quieter
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}", exc_info=True)
            # Consider more specific error codes if YOLO/Triton provide them
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
        t_inferred = time.time()

        # 3. Process Results
        boxes = [[int(round(x)) for x in box] for box in result.boxes.xyxy.tolist()]
        labels = [result.names[int(cls)] for cls in result.boxes.cls.tolist()]
        detections = [
            {"bbox_2d": box, "label": label} for box, label in zip(boxes, labels)
        ]

        # 4. Format Output String
        if not detections:
            dets_string = "No objects detected."
            annotated_image = None
            plot_img_array = None
        else:
            dets_string = ""
            for index, det in enumerate(detections):
                dets_string += f"{index + 1}. {det}"
                if index < len(detections) - 1:
                    dets_string += "\n"
            # Generate annotated image array
            plot_img_array = result.plot(conf=False, labels=True)
            annotated_image = Image.fromarray(plot_img_array[..., ::-1])  # BGR->RGB

        t_processed = time.time()

        # 5. Encode Annotated Image (if any)
        image_data_base64: Optional[str] = None
        if annotated_image:
            try:
                with io.BytesIO() as buffer:
                    # Save as PNG (generally lossless)
                    annotated_image.save(buffer, format="PNG")
                    img_bytes = buffer.getvalue()
                    image_data_base64 = base64.b64encode(img_bytes).decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to encode annotated image: {e}")
                # Proceed without annotated image if encoding fails
        t_encoded = time.time()

        logger.info(
            f"Detection successful. Timings (s): "
            f"Decode: {t_decoded - t_start:.3f}, "
            f"Inference: {t_inferred - t_decoded:.3f}, "
            f"Process: {t_processed - t_inferred:.3f}, "
            f"Encode: {t_encoded - t_processed:.3f}, "
            f"Total: {t_encoded - t_start:.3f}"
        )

        return DetectionResponse(
            text_data=dets_string,
            image_data_base64=image_data_base64,
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like 400 Bad Request)
        raise http_exc
    except Exception as e:
        # Catch-all for unexpected server errors during processing
        logger.error(f"Unexpected error during detection request: {e}", exc_info=True)
        # Return a generic 500 error response
        return JSONResponse(
            status_code=500, content={"error": f"Internal server error: {e}"}
        )


# --- Run Server ---

if __name__ == "__main__":
    # Set default port if not specified in environment
    server_port = int(os.getenv("DETECTION_API_PORT", 8001))
    num_workers = int(os.getenv("DETECTION_API_WORKERS", 6))  # Default to 6 workers
    logger.info(
        f"Starting YOLO detection server on port {server_port} with {num_workers} workers (CPU forced)..."  # Added CPU note
    )
    # Note: Using uvicorn.run() with workers > 1 might have limitations
    # compared to running via the command line with a process manager like gunicorn.
    # See Uvicorn documentation for details on multi-process modes.
    uvicorn.run(
        "training_server:app",  # Need to specify app string for reload/workers
        host="0.0.0.0",
        port=server_port,
        workers=num_workers,
        # reload=False # Ensure reload is False when using workers programmatically
    )
