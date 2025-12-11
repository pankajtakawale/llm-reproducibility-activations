#!/bin/bash
# Interactive shell in NVIDIA PyTorch container with GPU support

echo "Starting interactive shell with GPU support..."
docker run --gpus all -it --rm --ipc=host \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:24.11-py3 \
    /bin/bash
