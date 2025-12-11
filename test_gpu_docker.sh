#!/bin/bash
# Quick test to verify GPU is available in Docker container

echo "Testing GPU availability in NVIDIA PyTorch container..."
docker run --gpus all --rm --ipc=host nvcr.io/nvidia/pytorch:24.11-py3 python -c "
import torch
print('=' * 60)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))
    print('Number of GPUs:', torch.cuda.device_count())
else:
    print('WARNING: CUDA not available!')
print('=' * 60)
"
