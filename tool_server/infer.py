import contextlib
import time

import numpy as np
from imgcat import imgcat
from PIL import Image
from tritonclient.http import InferenceServerClient
from ultralytics import YOLO

# Wait for the Triton server to start
triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

# Wait until model is ready
for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready("yolo")
        break
    time.sleep(1)


# Load the Triton Server model
model = YOLO("http://localhost:8000/yolo", task="detect")

# load the image via PIL
img = Image.open(
    "/millcreek/home/sunil/r1_vlm_bumbershoot0/r1_vlm/tool_server/cars.jpeg"
)

# create 10 noisy copies and their crops
test_images = []
crop_ratios = [(2, 1), (1, 1), (1, 2)]

for i in range(10):
    # Create noisy image
    arr = np.array(img)
    noise = np.random.normal(0, 5, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)

    # Create crops for this noisy image
    img_w, img_h = noisy_img.size
    for w_ratio, h_ratio in crop_ratios:
        if img_w / img_h > w_ratio / h_ratio:
            crop_h = img_h
            crop_w = int(crop_h * w_ratio / h_ratio)
        else:
            crop_w = img_w
            crop_h = int(crop_w * h_ratio / w_ratio)

        x0 = np.random.randint(0, img_w - crop_w + 1)
        y0 = np.random.randint(0, img_h - crop_h + 1)
        cropped = noisy_img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        test_images.append(
            {"image": cropped, "ratio": f"{w_ratio}:{h_ratio}", "noise_id": i}
        )
speeds = []

# run inference on each variant
for test_case in test_images:
    start = time.time()
    results = model(test_case["image"])  # Pass the cropped image
    end = time.time()
    speeds.append(end - start)
    print(
        f"Noise #{test_case['noise_id']}, Aspect ratio {test_case['ratio']} – time taken: {end - start} seconds"
    )

    # Convert the cropped image to numpy for visualization
    vis_img = np.array(test_case["image"])
    # Plot directly on the cropped image
    plotted = results[0].plot(img=vis_img)
    imgcat(Image.fromarray(plotted))

print(speeds)
