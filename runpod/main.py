import runpod
import os
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")


# print(runpod.get_gpus())


def create_test_pod(
    image_name: str = "runpod/base:0.1.0",
    gpu_type_id: str = "NVIDIA RTX A5000",
    cloud_type: str = "ALL",
    gpu_count: int = 1,
    container_disk_in_gb: int = 20,
    volume_in_gb: int = 100,
    min_vcpu_count: int = 9,
    min_memory_in_gb: int = 16,
    min_download: int = 100,
    min_upload: int = 100,
):
    """
    This creates an inexpensive pod for general testing purposes. As of April 2, 2025, this pod costs around 25 cents per hour with the default parameters.
    """
    runpod.create_pod(
        name="test",
        image_name=image_name,
        gpu_type_id=gpu_type_id,
        cloud_type=cloud_type,
        country_code="US",
        gpu_count=gpu_count,
        # container disk is ephemeral, so we need to set it to a large enough value so we can install all the dependencies.
        # DONT save any data here, or you will lose it when the pod is shut down for the day.
        container_disk_in_gb=container_disk_in_gb,
        # volume is persistent, so we can save data here. It should be mounted to /workspace. You must offload data from volume prior to terminating the pod.
        volume_in_gb=volume_in_gb,
        min_vcpu_count=min_vcpu_count,
        min_memory_in_gb=min_memory_in_gb,
        volume_mount_path="/workspace",
        min_download=min_download,
        min_upload=min_upload,
    )
