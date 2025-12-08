"""
Configuration for LLM reproducibility experiments.
Based on the original project's Config structure.
"""
import torch


class Config:
    """Configuration for character-level language model experiments."""
    
    # Activation functions to evaluate
    # Will be defined in activations.py and imported during training
    activation_functions = {
        'smelu_05': None,  # SmeLU(beta=0.5)
        'smelu_1': None,   # SmeLU(beta=1.0)
        'relu': None,      # ReLU
        'gelu': None,      # GELU
        'swish': None      # Swish/SiLU
    }
    
    # Number of training trials per activation function
    trials_per_activation = 3  # 3 trials for reproducibility metrics
    
    # Model architecture - Partial model optimized for 3-hour full experiment run
    vocab_size = None  # Will be set after loading data
    n_embd = 256       # Embedding dimension (reduced from 384)
    n_head = 4         # Number of attention heads (reduced from 6)
    n_layer = 2        # Number of transformer layers (reduced from 6)
    block_size = 256   # Context length (characters)
    dropout = 0.2      # Dropout rate
    
    # Training hyperparameters
    batch_size = 64
    max_iters = 500   # Number of training iterations (6 models × 5 activations in ~3 hours)
    learning_rate = 3e-4
    eval_interval = 100  # Evaluate every 100 steps
    eval_iters = 200
    
    # Reproducibility settings
    seed_base = 42     # Base seed (will increment for each trial)
    
    # Device
    device = 'cpu'  # M4 Pro CPU
    
    # Data
    dataset = 'shakespeare'
    train_split = 0.9
    
    # Results
    save_checkpoints = True
    results_dir = 'results'
    checkpoint_dir = 'checkpoints'
    plots_dir = 'plots'
    checkpoint_dir = 'checkpoints'
    results_dir = 'results'
    
    def __repr__(self):
        return f"Config(model={self.n_layer}L-{self.n_embd}H, iters={self.max_iters}, batch={self.batch_size})"
