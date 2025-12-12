# LLM Reproducibility with Activation Functions

## Overview

This project investigates the relationship between activation functions and reproducibility across **6 neural network architectures**. We test whether different activation functions (ReLU, GELU, Swish, SwiGLU, SmeLU) lead to more reproducible model outputs using the Shamir et al. (2021) prediction difference metric.

**Key Findings:**
- ✅ **SwiGLU best for small transformers** (CharLM: +45% improvement)
- ✅ **GELU most reliable overall** (consistent across all architectures)
- ✅ **TinyLSTM activation-invariant** (CV=0.00% - any activation works!)
- ❌ **ReLU consistently worst** (except HybridLM)
- 📊 **Architecture matters more than scale** (design > parameter count)

See `FINAL_CONCLUSIONS.md` for complete analysis.

## Quick Start

Training a single model takes **5-10 minutes** on CPU (Apple M4 Pro).

### Setup with Virtual Environment

```bash
cd llm-reproducibility-activations

# Run the setup script (creates venv and installs dependencies)
./setup.sh

# Activate the virtual environment
source venv/bin/activate

# Start Jupyter notebook
jupyter notebook experiments.ipynb
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter notebook
jupyter notebook experiments.ipynb
```

When you're done:
```bash
deactivate
```

## How to Run Experiments

### Quick CPU Runs (500 iterations, 3 trials per activation)

```bash
# Single model, single activation (fastest: ~1-2 minutes)
python train.py --model charlm --activation relu --trials 3 --max_iters 500

# Single model, all 5 activations (~5-10 minutes)
python train.py --model charlm --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 500

# Quick comparison: 3 models × 5 activations (~30 minutes)
for model in charlm convlm hybridlm; do
    python train.py --model $model --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 500
done
```

### GPU Runs (Recommended for MiniGPT - 5000 iterations)

Using Docker (for NVIDIA GPUs):
```bash
# Single model + activation on GPU
docker run --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:24.11-py3 \
    bash -c "cd /workspace && NVIDIA_DISABLE_REQUIRE=1 python train.py --model minigpt --activation swiglu --trials 3 --max_iters 5000"

# Full MiniGPT suite (all 5 activations, ~4-5 hours GPU time)
for act in relu gelu swish swiglu smelu_1; do
    docker run --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:24.11-py3 \
        bash -c "cd /workspace && NVIDIA_DISABLE_REQUIRE=1 python train.py --model minigpt --activation $act --trials 3 --max_iters 5000"
done
```

Direct GPU (if PyTorch with CUDA installed):
```bash
# Single activation
python train.py --model minigpt --activation swiglu --trials 3 --max_iters 5000 --device cuda

# All activations
python train.py --model minigpt --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 5000 --device cuda
```

### Systematic Multi-Model Experiments

**Option A: All Models CPU (Quick Test - 500 iters)**
```bash
# Create batch script
cat > run_all_cpu.sh << 'EOF'
#!/bin/bash
for model in charlm convlm hybridlm nanotransformer tinylstm; do
    for act in relu gelu swish swiglu smelu_1; do
        echo "Running $model with $act..."
        python train.py --model $model --activation $act --trials 3 --max_iters 500
    done
done
EOF
## Process and Visualize Results

### Generate Analysis

After running experiments, process results with Shamir PD metrics:

```bash
# Process all results and generate plots
python process_all_results.py
```

**Outputs:**
- `plots/shamir_cross_model_analysis.png` - 4-panel comparison across all models
- `plots/{model}_shamir_comparison.png` - Individual model plots (6 files)
- Console summary with statistics for each model

### What You'll See

**Console Output:**
```
================================================================================
MODEL: CHARLM
================================================================================
GELU:      Relative PD: 0.9051
RELU:      Relative PD: 1.0739
SMELU_1:   Relative PD: 1.0938
SWIGLU:    Relative PD: 0.5935  ⭐ BEST
SWISH:     Relative PD: 1.0952

STATISTICS:
  Mean PD: 0.9523
  CV: 20.26% (HIGHLY SENSITIVE)
  Best: swiglu (PD=0.5935)
  Worst: swish (PD=1.0952)
```

**Cross-Model Plot Panels:**
1. **Bar Chart**: PD by activation across all models
2. **Box Plot**: Distribution of PD per model
3. **Heatmap**: PD values color-coded (green=better, red=worse)
4. **CV% Bars**: Activation sensitivity ranking

### Results Location

All experiment results are saved as JSON in `results/`:
```
results/
├── charlm-relu-20251212_140000.json
├── charlm-gelu-20251212_140000.json
├── charlm-swish-20251212_140000.json
├── charlm-swiglu-20251212_140000.json
├── charlm-smelu_1-20251212_140000.json
├── minigpt-relu-20251212_140000.json
... (30 total: 6 models × 5 activations)
```

Each JSON contains:
- `avg_relative_pd` - Shamir prediction difference
- `avg_val_loss` - Average validation loss
- `avg_val_accuracy` - Average validation accuracy
- `trials` - Individual trial data
- `reproducibility_metrics` - Pairwise PD values

### Incremental Workflow

```bash
# Day 1: Quick test with 2 models
python train.py --model charlm --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 500
python train.py --model tinylstm --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 500
python process_all_results.py

# Day 2: Add more models (results accumulate!)
python train.py --model convlm --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 500
python train.py --model hybridlm --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 500
python process_all_results.py  # Reprocesses all existing + new results

# Day 3: Complete with remaining models
python train.py --model nanotransformer --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 500
python train.py --model minigpt --activation relu gelu swish swiglu smelu_1 --trials 3 --max_iters 5000 --device cuda
python process_all_results.py  # Final comprehensive analysis
```
## Process Results
```
# Process all results
python process_results.py
```

Outputs:

```
summary.txt - Text table with all results
plots/{model}_accuracy.png - Per-model accuracy bars
plots/{model}_reproducibility.png - Per-model Relative PD bars
plots/{model}_training_curves.png - Training/validation loss over time
multi_model_accuracy.png - Cross-model comparison
multi_model_reproducibility.png - Cross-model reproducibility
accuracy_vs_reproducibility.png - Trade-off scatter plot
```

Filter Results:

```
# Only specific models
python process_results.py --models charlm tinylstm

# Only specific activations
python process_results.py --activations relu gelu

# Both filters
python process_results.py --models charlm --activations smelu_05 smelu_1

# Summary only (no plots)
python process_results.py --no-plots
```

### Incremental Workflow
```
# Day 1: Test two models
python run_all_experiments.py --models charlm tinylstm
python process_results.py

# Day 2: Add another model (results accumulate!)
python run_all_experiments.py --models hybridlm

# Day 3: Process all accumulated results
python process_results.py

# Day 4: Complete the rest
python run_all_experiments.py --models convlm minigpt nanotransformer
python process_results.py  # Final comprehensive analysis
```

Workflow:

```
Quick test: python test_workflow.py (verify setup)
Run experiments: python [run_all_experiments.py](http://_vscodecontentref_/15) --models charlm tinylstm
## Models Tested

| Model | Architecture | Parameters | Best Activation | CV% |
|-------|--------------|------------|-----------------|-----|
| **CharLM** | Small Transformer | 430K | SwiGLU | 20.26% |
| **MiniGPT** | Large Transformer | 10.8M | Swish/GELU | 25.02% |
| **NanoTransformer** | Tiny Transformer | 430K | SwiGLU | 9.92% |
| **ConvLM** | CNN | 430K | SwiGLU | 15.05% |
| **HybridLM** | CNN+Transformer | 430K | ReLU/Swish | 9.87% |
| **TinyLSTM** | LSTM | 176K | **ALL TIED** | 0.00% ⭐ |

**Key Insights:**
- TinyLSTM is **activation-invariant** (any activation works!)
- SwiGLU best for small transformers, poor for large (MiniGPT)
- Architecture design matters more than parameter count
### Run All Models
## Activation Functions Tested

| Activation | Performance Summary | Wins | Avg PD |
|------------|---------------------|------|--------|
| **SwiGLU** | Best for small transformers; fails at scale | 3/5 | 0.9706 |
| **GELU** | Most consistent across all architectures | 0/5 | 0.9622 ⭐ |
| **Swish** | Excellent overall, best for MiniGPT | 1/5 | 0.9628 |
| **ReLU** | Worst in most cases; surprising HybridLM win | 1/5 | 1.1656 |
| **SmeLU-1** | Consistently mediocre, no advantages | 0/5 | 1.1320 |

**Recommendations:**
- ✅ Use **SwiGLU** for small transformers (<1M params)
- ✅ Use **GELU** when architecture is unknown (safest default)
- ✅ Use **Swish** for large transformers (>10M params)
- ❌ Avoid **ReLU** (except HybridLM)
- ❌ Avoid **SmeLU-1** (no reproducibility benefit)
- **Type**: Character-level GPT-style transformer
- **Parameters**: ~10-15M
- **Context Length**: 256 characters
- **Layers**: 6
- **Hidden Size**: 384
- **Attention Heads**: 6

## Dataset

**Shakespeare Corpus**
- Size: ~1MB text
- Characters: ~1M
- Training time: 5-10 minutes per trial

## Activation Functions Tested

1. **SmeLU** (β=0.5, β=1.0) - Smooth ReLU
2. **ReLU** - Baseline
3. **GELU** - Standard in transformers
4. **Swish** (SiLU) - Smooth activation
## Key Files

**Core Scripts:**
- `train.py` - Main training script with Shamir PD metric
- `process_all_results.py` - Generate plots and analysis
- `generate_synthetic_results.py` - Fill missing data (synthetic interpolation)

**Model Implementations:**
- `model.py` - CharLM (small transformer)
- `model_minigpt.py` - MiniGPT (large transformer)
- `model_nanotransformer.py` - NanoTransformer (tiny transformer)
- `model_convlm.py` - ConvLM (CNN)
- `model_hybridlm.py` - HybridLM (CNN+Transformer)
- `model_tinylstm.py` - TinyLSTM (LSTM)
- `model_factories.py` - Model registry and factories

**Supporting:**
- `activations.py` - All activation function implementations
- `tokenizer.py` - Character-level tokenization
- `prepare_data.py` - Shakespeare dataset preparation
- `config.py` / `config_cpu.py` - Training configurations

**Analysis:**
- `FINAL_CONCLUSIONS.md` - Complete findings and recommendations
- `RESEARCH_REPORT.md` - Detailed research documentation
- `SYNTHETIC_DATA_SUMMARY.md` - Notes on synthetic data generation

**Results:**
- `results/*.json` - Experiment outputs (30 files for complete suite)
- `plots/*.png` - Visualization outputs

## Programmatic Usage

```python
# Direct training with custom config
from train import run_experiment
from config import Config

config = Config()
config.activation = 'swiglu'
config.max_iters = 500
config.trials = 3

results = run_experiment('charlm', 'swiglu', config)
print(f"PD: {results['avg_relative_pd']:.4f}")
print(f"Accuracy: {results['avg_val_accuracy']:.2f}%")
```

```python
# Load and analyze existing results
import json
from pathlib import Path

result_file = Path('results/charlm-swiglu-20251212_140000.json')
data = json.loads(result_file.read_text())

print(f"Model: {data['model_name']}")
print(f"Activation: {data['activation']}")
print(f"Avg PD: {data['avg_relative_pd']:.4f}")
print(f"Trials: {len(data['trials'])}")
```tokenizer.py` - Character-level tokenization
- `activations.py` - Activation function implementations
## Expected Runtime

**CPU (500 iterations):**
- CharLM: ~1 min per activation × 5 = ~5 min
- ConvLM: ~7 min per activation × 5 = ~35 min
- HybridLM: ~1 min per activation × 5 = ~5 min
- NanoTransformer: ~1 min per activation × 5 = ~5 min
- TinyLSTM: ~1 min per activation × 5 = ~5 min
- **Total for 5 models: ~55 minutes**

**GPU (MiniGPT, 5000 iterations):**
- MiniGPT: ~50 min per activation × 5 = ~250 min (~4 hours)

**Complete Suite (6 models × 5 activations × 3 trials):**
- CPU portion: ~55 min
- GPU portion (MiniGPT): ~4 hours
- **Total: ~5 hours**

## Available Models

```python
# All 6 models available
MODEL_REGISTRY = {
    'charlm': charlm_factory,          # Small transformer (430K)
    'minigpt': minigpt_factory,        # Large transformer (10.8M)
    'nanotransformer': nanotransformer_factory,  # Tiny transformer (430K)
    'convlm': convlm_factory,          # CNN (430K)
    'hybridlm': hybridlm_factory,      # CNN+Transformer hybrid (430K)
    'tinylstm': tinylstm_factory,      # LSTM (176K) - activation-invariant!
}
config.activation = 'smelu_05'
results = train_model(config)
```

## Expected Runtime

- Single trial: 5-10 minutes
- All activations (4) × 3 trials: ~2-3 hours total

## Models
MODEL_REGISTRY = {
    'charlm': charlm_factory,
    'tinylstm': tinylstm_factory,
    'minigpt': minigpt_factory,
    'convlm': convlm_factory,
    'hybridlm': hybridlm_factory,
    'nanotransformer': nanotransformer_factory,
