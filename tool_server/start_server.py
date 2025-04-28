import contextlib
import subprocess
import time
from pathlib import Path

from tritonclient.http import InferenceServerClient

model_name = "yoloe"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name

# Define image https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
tag = "nvcr.io/nvidia/tritonserver:24.09-py3"  # 8.57 GB

# Pull the image
subprocess.call(f"docker pull {tag}", shell=True)

# Run the Triton server and capture the container ID
container_id = (
    subprocess.check_output(
        f"docker run -d --rm --gpus 0 -v {triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)

# Wait for the Triton server to start
triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

# Wait until model is ready
for _ in range(10):
    with contextlib.suppress(Exception):
        assert triton_client.is_model_ready(model_name)
        break
    time.sleep(1)
