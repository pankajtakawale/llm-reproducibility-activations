# LLM Reproducibility with Activation Functions

## Overview

This project investigates the relationship between activation functions and reproducibility in character-level language models. Following the methodology from the parent project, we test whether smooth activation functions lead to more reproducible model outputs.

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

## How to Run

### Single Model
```
# CharLM only (fastest: ~7s per activation = 35s total)
python run_all_experiments.py --models charlm

# TinyLSTM only (~15s per activation = 75s total)
python run_all_experiments.py --models tinylstm

# MiniGPT only (~25s per activation = 125s total)
python run_all_experiments.py --models minigpt
```

### Multiple Models

```
# 2 models (fastest pair: ~2 minutes)
python run_all_experiments.py --models charlm tinylstm

# 3 models (recommended for comparison: ~5 minutes)
python run_all_experiments.py --models charlm tinylstm hybridlm

# All 6 models (~9 minutes)
python run_all_experiments.py --models charlm tinylstm minigpt convlm hybridlm nanotransformer
```

### Custom Activations
```
# Only ReLU and GELU (fastest)
python run_all_experiments.py --models charlm --activations relu gelu

# Only SmeLU variants
python run_all_experiments.py --models charlm tinylstm --activations smelu_05 smelu_1
```

### Background Run

```
# Run in background
nohup python run_all_experiments.py --models all > experiments.log 2>&1 &

# Get process ID
echo $!

# Monitor progress (in another terminal)
tail -f experiments.log

# Or check periodically
tail -100 experiments.log

# Kill if needed
kill <PID>
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
Analyze results: python process_results.py
Check outputs:
summary.txt - Table of all results
plots - Visual comparisons
Iterate: Add more models or activations as needed

```

### Run All Models
```
python run_all_experiments.py --models all
```

## Model Architecture

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

## Reproducibility Metrics

- **Relative Prediction Difference (PD)**: Measures output consistency between models trained with identical configs
- **Perplexity Variance**: Standard deviation across trials
- **Character-level Output Consistency**: Exact match percentage on test prompts

## Experiment Structure

```
For each activation function:
  - Train 2-3 models with identical hyperparameters
  - Evaluate on same test set
  - Calculate reproducibility metrics
  - Compare accuracy vs reproducibility trade-offs
```

## Files

- `prepare_data.py` - Downloads and prepares Shakespeare dataset
- `tokenizer.py` - Character-level tokenization
- `activations.py` - Activation function implementations
- `model.py` - Character-level transformer architecture
- `train.py` - Training loop with reproducibility tracking
- `experiments.ipynb` - Main notebook for running experiments
- `config.py` - Configuration settings

## Usage

```python
# Run in experiments.ipynb
from train import train_model
from config import Config

config = Config()
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
