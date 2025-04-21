import asyncio
import base64
import io
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel


# Model configs and types
class DetectionResult(BaseModel):
    boxes: List[List[float]]  # [x1, y1, x2, y2]
    scores: List[float]
    labels: List[str]
    
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
    confidence: float = 0.0,
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
        
        print("Setting classes")
        # Set detection classes
        model.set_classes(classes, model.get_text_pe(classes))
        
        print("Predicting")
        results = model.predict(img, conf=confidence)
        
        print(results)
        
        #results[0].show()
            
        # TODO: Actual model inference will go here
        # For now returning dummy data
        dummy_result = DetectionResult(
            boxes=[[0, 0, 100, 100]],
            scores=[0.95],
            labels=["dummy"]
        )
        
        # Convert annotated image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return DetectionResponse(
            results=dummy_result,
            annotated_image=img_str
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
