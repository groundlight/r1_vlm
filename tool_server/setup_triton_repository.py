from pathlib import Path

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")  # load an official model

# Retrieve metadata during export. Metadata needs to be added to config.pbtxt. See next section.
metadata = []


def export_cb(exporter):
    metadata.append(exporter.metadata)


model.add_callback("on_export_end", export_cb)

# Export the model
onnx_file = model.export(format="onnx", dynamic=True)


# Define paths
model_name = "yolo"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name

# Create directories
(triton_model_path / "1").mkdir(parents=True, exist_ok=True)

# Move ONNX model to Triton Model path
Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")

# Create config file
(triton_model_path / "config.pbtxt").touch()


data = """
# Add metadata
parameters {
  key: "metadata"
  value {
    string_value: "%s"
  }
}
""" % metadata[0]  # noqa

with open(triton_model_path / "config.pbtxt", "w") as f:
    f.write(data)
