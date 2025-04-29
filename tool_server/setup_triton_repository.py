import shutil
from pathlib import Path

from ultralytics import YOLOE

# Ensure tmp directory exists
Path("tmp").mkdir(exist_ok=True)

# Define paths
model_name = "yoloe"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name
onnx_export_path = Path(
    f"{model_name}.onnx"
)  # Define where the ONNX file will be saved initially

# --- Clean up previous attempts ---
if onnx_export_path.exists():
    onnx_export_path.unlink()
if triton_repo_path.exists():
    shutil.rmtree(triton_repo_path)
# ---------------------------------

print("Loading model...")
model = YOLOE("yoloe-11l-seg.pt")  # Load the base model

# Export the model to a predictable path
print(f"Exporting model to {onnx_export_path}...")
# Note: Ultralytics saves with a modified name by default, but export returns the actual path
actual_onnx_file_path = model.export(format="onnx", dynamic=True)
# Ensure the exported file is at the expected path for clarity
Path(actual_onnx_file_path).rename(onnx_export_path)
print(f"Model exported to {onnx_export_path}")


# --- Setup Triton directories ---
print(f"Creating Triton repository structure at {triton_repo_path}...")
triton_model_version_path = triton_model_path / "1"
triton_model_version_path.mkdir(parents=True, exist_ok=True)

# Move ONNX model to Triton Model path
print(f"Moving ONNX model to {triton_model_version_path / 'model.onnx'}")
onnx_export_path.rename(triton_model_version_path / "model.onnx")
# --------------------------------

# --- Create config.pbtxt ---
# Based on export logs:
# Input: (1, 3, 640, 640) -> name="images", shape=[-1, 3, 640, 640]
# Output 0: (1, 116, 8400) -> name="output0", shape=[-1, 116, 8400]
# Output 1: (1, 32, 160, 160) -> name="output1", shape=[-1, 32, 160, 160]
config_pbtxt_content = """
name: "yoloe"
platform: "onnxruntime_onnx"
max_batch_size: 0 # Support dynamic batch size

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ -1, 3, 640, 640 ]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
  },
  {
    name: "output1"
    data_type: TYPE_FP32
    dims: [ -1, 32, -1, -1 ]
  }
]

# Optional TensorRT optimization (keep commented out for now)
# optimization {
#   execution_accelerators {
#     gpu_execution_accelerator [
#       {
#         name : "tensorrt"
#         parameters { key: "precision_mode" value: "FP16" }
#         parameters { key: "max_workspace_size_bytes" value: "3221225472" }
#       }
#     ]
#   }
# }

# --- Change back to KIND_GPU and specify GPU 0 ---
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
# --- End Change ---
"""

print(f"Writing config.pbtxt to {triton_model_path / 'config.pbtxt'} for GPU execution")
config_file_path = triton_model_path / "config.pbtxt"
with open(config_file_path, "w") as f:
    f.write(config_pbtxt_content)

print("Triton repository setup complete.")
