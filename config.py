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
    trials_per_activation = 3
    
    # Model architecture
    vocab_size = None  # Will be set after loading data
    n_embd = 384       # Embedding dimension
    n_head = 6         # Number of attention heads
    n_layer = 6        # Number of transformer layers
    block_size = 256   # Context length (characters)
    dropout = 0.2      # Dropout rate
    
    # Training hyperparameters
    batch_size = 64
    max_iters = 5000   # Number of training iterations (~5-10 min)
    learning_rate = 3e-4
    eval_interval = 500
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
    checkpoint_dir = 'checkpoints'
    results_dir = 'results'
    
    def __repr__(self):
        return f"Config(model={self.n_layer}L-{self.n_embd}H, iters={self.max_iters}, batch={self.batch_size})"
