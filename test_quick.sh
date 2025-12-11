#!/bin/bash
# Quick test with 1 model, 1 activation, 3 trials, 200 iterations

echo "=========================================="
echo "Quick GPU Test - Single Model"
echo "Model: charlm | Activation: relu"
echo "Trials: 3 | Iterations: 200"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p results results_backup checkpoints data

echo "Starting quick test..."
sudo docker run --gpus all --rm --ipc=host \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:24.11-py3 \
    bash -c "
echo '==> Installing dependencies...'
pip install -q matplotlib pandas scipy tabulate requests
echo '==> Dependencies installed!'
echo '==> Running quick test...'
python3 run_all_experiments.py --models charlm --activations relu
echo '==> Processing results...'
python3 process_results.py
"

echo ""
echo "=========================================="
echo "Quick test completed!"
echo "Check ./results/ for output"
echo "=========================================="
