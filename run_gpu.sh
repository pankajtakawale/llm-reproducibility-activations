#!/bin/bash
# Quick run script using NVIDIA PyTorch container directly (no build needed)

set -e

echo "=========================================="
echo "Running LLM Reproducibility Experiments"
echo "Using NVIDIA PyTorch Container with GPU"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p results results_backup checkpoints data

echo "Starting experiments..."
sudo docker run --gpus all --rm --ipc=host \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:24.11-py3 \
    bash -c "
echo '==> Installing dependencies...'
pip install -q matplotlib pandas scipy tabulate requests
echo '==> Dependencies installed!'
echo '==> Starting run.sh...'
bash run.sh
"

echo ""
echo "=========================================="
echo "Experiments completed!"
echo "Results saved to ./results/"
echo "=========================================="
