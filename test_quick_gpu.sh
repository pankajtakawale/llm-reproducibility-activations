#!/bin/bash
# Simple test - run one quick experiment with GPU to verify everything works

echo "Running quick GPU test with one model..."
sudo docker run --gpus all --rm --ipc=host \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:24.11-py3 \
    bash -c "
pip install -q matplotlib pandas scipy tabulate requests && \
python3 run_all_experiments.py --models charlm --activations relu
"

echo ""
echo "Test completed! Check if GPU was used in the output above."
