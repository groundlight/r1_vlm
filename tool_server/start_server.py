import subprocess
import time
from pathlib import Path

from tritonclient.http import InferenceServerClient

model_name = "yoloe"
triton_repo_path = Path("tmp") / "triton_repo"
# Get the absolute path for the Docker volume mount
absolute_triton_repo_path = triton_repo_path.resolve()
triton_model_path = triton_repo_path / model_name

# Define image - trying an older tag potentially more compatible with driver
tag = "nvcr.io/nvidia/tritonserver:24.05-py3"  # Changed from 24.09

# Pull the image
print(f"Pulling Triton image: {tag}")
subprocess.call(f"docker pull {tag}", shell=True)

# Run the Triton server and capture the container ID
# Change --gpus 0 to --gpus all
print("Starting Triton server container with --gpus all...")
container_id = (
    subprocess.check_output(
        f"docker run -d --rm --gpus all -v {absolute_triton_repo_path}:/models -p 8000:8000 {tag} tritonserver --model-repository=/models",
        shell=True,
    )
    .decode("utf-8")
    .strip()
)
print(
    f"Started container with ID: {container_id}"
)  # Keep this to see the ID if it fails quickly

# Wait for the Triton server to start
triton_client = InferenceServerClient(url="localhost:8000", verbose=False, ssl=False)

# Wait until model is ready
print(f"Waiting for model '{model_name}' to become ready...")
model_ready = False
for i in range(20):  # Increased timeout slightly
    try:
        if triton_client.is_model_ready(model_name):
            print(f"Model '{model_name}' is ready.")
            model_ready = True
            break
    except Exception as e:
        print(
            f"Waiting for server/model... attempt {i + 1}/20 ({e.__class__.__name__})"
        )
    time.sleep(2)  # Increased sleep time

if not model_ready:
    print(f"Error: Model '{model_name}' did not become ready.")
    # Optional: Try to get logs if the container might still exist briefly
    print("Attempting to fetch logs...")
    time.sleep(1)  # Give logs a moment to flush
    subprocess.call(f"docker logs {container_id}", shell=True)
else:
    print("Triton server started successfully with the model loaded on GPU.")

# Note: The script will exit here, and the --rm flag will remove the container
# if the Triton process stops for any reason (including normal shutdown if it were interactive).
# For a persistent server, you'd typically run the docker command outside the script
# or use docker-compose/kubernetes.
