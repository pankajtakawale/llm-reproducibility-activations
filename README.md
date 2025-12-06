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
