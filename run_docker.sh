#!/bin/bash
# Script to build and run experiments in NVIDIA PyTorch container with GPU support

set -e

echo "Building Docker image with NVIDIA PyTorch (CUDA-enabled)..."
docker build -t llm-reproducibility-gpu .

echo ""
echo "Running experiments with GPU support..."
docker run --gpus all --rm --ipc=host \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/results_backup:/workspace/results_backup \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -v $(pwd)/data:/workspace/data \
    llm-reproducibility-gpu \
    bash run.sh

echo ""
echo "Experiments completed! Results saved to ./results/"
