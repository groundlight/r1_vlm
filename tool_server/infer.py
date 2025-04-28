import numpy as np
import tritonclient.http as httpclient
from PIL import Image

# --- Configuration ---
TRITON_URL = "localhost:8000"  # Triton HTTP endpoint (host:port)
MODEL_NAME = "yoloe"
IMAGE_PATH = "/millcreek/home/sunil/r1_vlm_bumbershoot2/r1_vlm/tool_server/cars.jpeg"
MODEL_INPUT_SHAPE = (
    640,
    640,
)  # Expected input height/width for the ONNX model
# ---------------------


def preprocess(img: Image.Image, target_shape: tuple) -> np.ndarray:
    """Preprocesses PIL Image for YOLO/ONNX inference."""
    target_h, target_w = target_shape
    original_w, original_h = img.size

    # Resize while maintaining aspect ratio (letterboxing/padding)
    ratio = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * ratio)
    new_h = int(original_h * ratio)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create a blank canvas (black background)
    img_padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    # Paste the resized image onto the canvas
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    img_padded.paste(img_resized, (paste_x, paste_y))

    # Convert to numpy array, normalize to [0, 1], and change layout HWC -> CHW
    image_data = np.array(img_padded, dtype=np.float32) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))  # HWC -> CHW

    # Add batch dimension: CHW -> NCHW
    image_data = np.expand_dims(image_data, axis=0)
    return image_data


# --- Main Inference Logic ---
try:
    print(f"Creating Triton client for URL: {TRITON_URL}")
    # Create Triton HTTP client
    # Note: verbose=True can help debug connection issues
    triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    print("Loading and preprocessing image...")
    input_image = Image.open(IMAGE_PATH).convert("RGB")
    processed_image = preprocess(input_image, MODEL_INPUT_SHAPE)
    print(f"Preprocessed image shape: {processed_image.shape}")

    # --- Create Inference Inputs ---
    inputs = []
    inputs.append(httpclient.InferInput("images", processed_image.shape, "FP32"))
    inputs[0].set_data_from_numpy(processed_image)

    # --- Define Inference Outputs (requesting specific outputs) ---
    outputs = []
    # Request the outputs defined in config.pbtxt
    outputs.append(httpclient.InferRequestedOutput("output0"))
    outputs.append(httpclient.InferRequestedOutput("output1"))

    # --- Run Inference ---
    print(f"Sending inference request to model '{MODEL_NAME}'...")
    results = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
    print("Inference request complete.")

    # --- Process Results ---
    # Get the raw numpy arrays from the response
    raw_output0 = results.as_numpy("output0")
    raw_output1 = results.as_numpy("output1")

    print("\n--- Raw Triton Output ---")
    print(f"Output 'output0' shape: {raw_output0.shape}")
    print(f"Output 'output1' shape: {raw_output1.shape}")

    # Optional: Print some data to verify it's not all zeros
    print(
        f"\nSample data from output0 (first few elements):\n{raw_output0.flatten()[:10]}"
    )
    print(
        f"\nSample data from output1 (first few elements):\n{raw_output1.flatten()[:10]}"
    )

    # Simple check if any detections might be present in output0
    # Shape is typically (batch, num_predictions, box_coords + classes)
    # For YOLO, usually 4 box coords + 1 obj score + num_classes scores
    # Here it's (1, 116, 8400) -> (batch, 4 coords + 80 classes + 32 mask_coeffs ?, 8400 proposals)
    # This structure needs careful parsing based on the exact YOLOE export format.
    # A simpler check for now is just non-zero elements.
    if np.any(raw_output0):
        print("\nDetected non-zero values in 'output0', potential detections present.")
    else:
        print("\n'output0' appears to be all zeros.")

    if np.any(raw_output1):
        print("Detected non-zero values in 'output1', potential masks present.")
    else:
        print("'output1' appears to be all zeros.")


except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n-------------------------")
