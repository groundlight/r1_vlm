import asyncio
import base64
import io
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel


# Model configs and types
class Detection(BaseModel):
    bbox_2d: List[int]  # [x1, y1, x2, y2]
    label: str
    confidence: float

class DetectionResult(BaseModel):
    detections: List[Detection]
    
class DetectionResponse(BaseModel):
    results: DetectionResult
    annotated_image: str  # base64 encoded image

# Global model instance
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up model on startup
    global model
    from ultralytics import YOLOE
    model = YOLOE("yoloe-11l-seg.pt")  # Using large model for best performance
    
    if torch.cuda.is_available():
        model.to("cuda:0")
    yield
    # Cleanup on shutdown
    model = None

app = FastAPI(lifespan=lifespan)

# Semaphore to control concurrent access to GPU
# Adjust max_requests based on GPU memory and batch size
MAX_CONCURRENT_REQUESTS = 1
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@app.post("/detect", response_model=DetectionResponse)
async def detect(
    classes: List[str],
    confidence: float,
    image: UploadFile = File(...),
) -> DetectionResponse:
    # Use semaphore to control concurrent access
    async with request_semaphore:
        # Convert image to PIL
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        
        # validate the classes is a non-empty list of strings
        if classes:
            if not isinstance(classes, list):
                raise ValueError("Error: classes must be a list")
            if not all(isinstance(c, str) for c in classes):
                raise ValueError("Error: classes must be a list of strings")
            if len(classes) == 0:
                raise ValueError("Error: classes must be a non-empty list")
        
        print(f"Setting classes: {classes}")
        # Set detection classes
        model.set_classes(classes, model.get_text_pe(classes))
        
        print(f"Predicting with confidence {confidence}")
        results = model.predict(img, conf=confidence)
        
        # Extract actual detection results
        result = results[0]
        boxes = [[int(round(x)) for x in box] for box in result.boxes.xyxy.tolist()]
        scores = result.boxes.conf.tolist()
        labels = [result.names[int(cls)] for cls in result.boxes.cls.tolist()]
        
        # Format detections as requested
        detections = [
            {"bbox_2d": box, "label": label, "confidence": score}
            for box, label, score in zip(boxes, labels, scores)
        ]
        
        # Create annotated image
        annotated_img = result.plot(conf=True, labels=True)
        buffered = io.BytesIO()
        Image.fromarray(annotated_img).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Log the classes and visualize the image inline using iTerm2 escape codes
        print(f"Detection complete for classes: {classes}")
        # iTerm2 inline image protocol: \033]1337;File=inline=1:[BASE64_DATA]\a
        print(f"\033]1337;File=inline=1:{img_str}\a")

        return DetectionResponse(
            results=DetectionResult(
                detections=detections
            ),
            annotated_image=img_str
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
