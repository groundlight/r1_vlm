# Troubleshooting Installation

## Errors installing `flash-attn`

The version of flash attention we're using really wants CUDA 12.4.  You can manually install that alongside your existing CUDA installation with:

```
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
# If NVIDIA removes this file, find a current one at https://developer.nvidia.com/cuda-toolkit-archive
# pick the "runfile (local)" version.

sudo sh cuda_12.4.1_550.54.15_linux.run \
  --toolkit \
  --silent \
  --override \
  --no-drm \
  --no-opengl-libs \
  --toolkitpath=/usr/local/cuda-12.4.1 \
  --defaultroot=/usr/local/cuda-12.4.1

export CUDA_HOME=/usr/local/cuda-12.4.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Now...!
uv sync --no-build-isolation
```

## Running in Docker

If you want to give up on a local installation, the supplied [Dockerfile](Dockerfile) should work reliably.  Assuming of course you have all the nvidia-docker stuff sorted out, which is its own can of worms.

```
docker build -t r1_vlm .
docker run -it --gpus all r1_vlm
```

## Still having issues?

If you're still having issues, please file an issue and we'll get back to you as soon as possible: [link](https://github.com/groundlight/r1_vlm/issues).