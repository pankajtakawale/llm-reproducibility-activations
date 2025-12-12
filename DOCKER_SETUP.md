# Running with GPU Support on NVIDIA DGX (ARM64)

## Quick Start

**Simplest way - Direct run (recommended):**
```bash
sudo ./run_gpu.sh
```

This will:
- Use NVIDIA's PyTorch container with CUDA support
- Automatically install dependencies
- Run all experiments with GPU acceleration
- Save results to `./results/`

## Alternative Methods

### Option 1: Interactive Shell
Open an interactive shell in the GPU-enabled container:
```bash
sudo ./docker_shell.sh
```

Then run commands manually:
```bash
pip install -r requirements.txt
python3 run_all_experiments.py --models all
```

### Option 2: Build Custom Image
Build your own Docker image (takes longer but only needed once):
```bash
sudo docker build -t llm-reproducibility-gpu .
sudo ./run_docker.sh
```

### Option 3: Test GPU First
Verify GPU is working before running experiments:
```bash
sudo ./test_gpu_docker.sh
```

## Why Docker?

Your system is ARM64 architecture. PyTorch doesn't provide pre-built CUDA wheels for ARM64 via pip. NVIDIA's containers have PyTorch with full CUDA support pre-built for ARM64.

## GPU Compatibility Note

Your NVIDIA GB10 GPU is very new (Blackwell architecture, sm_121). If you see compatibility warnings, the code will still run but may not be fully optimized. For best performance, consider:
- Using the latest NVIDIA container release
- Building PyTorch from source targeting sm_121

## Without Docker (CPU-only)

If you can't use Docker:
```bash
source venv/bin/activate
bash run.sh
```

Note: This runs on CPU only and will be significantly slower.
