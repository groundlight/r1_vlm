import contextlib
import subprocess
import time
from pathlib import Path

# import os # Still not needed
from tritonclient.http import InferenceServerClient

model_name = "yolo"
triton_repo_path = Path("tmp") / "triton_repo"

# Get the absolute path
absolute_triton_repo_path = triton_repo_path.resolve()
triton_model_path = triton_repo_path / model_name
# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "nvcr.io/nvidia/tritonserver:24.09-py3"  # 8.57 GB

subprocess.call(f"docker pull {tag}", shell=True)

container_id = (
    subprocess.check_output(
        # Use the absolute path here
        f"docker run -d --gpus 0 -v {absolute_triton_repo_path}:/models -p 0.0.0.0:8000:8000 {tag} tritonserver --model-repository=/models",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)

triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)


print(f"Triton server started with container ID: {container_id}")
