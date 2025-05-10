import contextlib
import os
import time

import numpy as np
from dotenv import load_dotenv
from imgcat import imgcat
from PIL import Image
from tritonclient.http import InferenceServerClient
from ultralytics import YOLO

load_dotenv()

API_IP = str(os.getenv("API_IP"))
API_PORT = int(os.getenv("API_PORT"))
url = f"{API_IP}:{API_PORT}/yolo"
print(url)

# Wait for the Triton server to start
triton_client = InferenceServerClient(url=url, verbose=False, ssl=False)

# Wait until model is ready
for _ in range(10):
    with contextlib.suppress(Exception):
        print("checking if model is ready")
        assert triton_client.is_model_ready("yolo")
        print("model is ready")
        break
    time.sleep(1)

print("loading model")
# Load the Triton Server model
model = YOLO(f"http://{url}", task="detect")

# load the image via PIL
img = Image.open(
    "/millcreek/home/sunil/r1_vlm_bumbershoot0/r1_vlm/tool_server/cars.jpeg"
)


results = model(img)  # Pass the cropped image

# Convert the cropped image to numpy for visualization
vis_img = np.array(img)
# Plot directly on the cropped image
plotted = results[0].plot(img=vis_img)
imgcat(Image.fromarray(plotted))
