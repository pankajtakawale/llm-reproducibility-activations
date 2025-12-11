#!/bin/bash
# Script to build PyTorch with CUDA support on ARM64

echo "Building PyTorch with CUDA support for ARM64..."
echo "This will take 1-2 hours. Press Ctrl+C to cancel."
sleep 3

# Activate virtual environment
source venv/bin/activate

# Install build dependencies
pip install ninja pyyaml mkl mkl-include cmake cffi typing_extensions

# Clone PyTorch repository
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Set environment variables for CUDA build
export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;9.0"  # Adjust based on your GPU architecture
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which python))/../"}

# Build and install
python setup.py install

echo "PyTorch with CUDA support has been installed!"
