"""
CPU-optimized configuration for quick testing on MacBook.
Much smaller model that can complete in reasonable time on CPU.
"""
import torch


class ConfigCPU:
    """Lightweight configuration for CPU testing."""
    
    # Activation functions to evaluate
    activation_functions = {
        'smelu_05': None,  # SmeLU(beta=0.5)
        'smelu_1': None,   # SmeLU(beta=1.0)
        'relu': None,      # ReLU
        'gelu': None,      # GELU
        'swish': None      # Swish/SiLU
    }
    
    # Number of training trials per activation function
    trials_per_activation = 3  # Reduced from 3 for faster testing
    
    # Model architecture (REDUCED for CPU)
    vocab_size = None  # Will be set after loading data
    n_embd = 128       # Reduced from 384
    n_head = 4         # Reduced from 6
    n_layer = 2        # Reduced from 6
    block_size = 128   # Reduced from 256
    dropout = 0.2      # Same
    
    # Training hyperparameters (REDUCED for CPU)
    batch_size = 32    # Reduced from 64
    max_iters = 200    # Reduced from 5000 for quick test
    learning_rate = 3e-4
    eval_interval = 50 # Reduced from 500
    eval_iters = 50    # Reduced from 200
    
    # Reproducibility settings
    seed_base = 42
    
    # Device
    device = 'cpu'
    
    # Data
    dataset = 'shakespeare'
    train_split = 0.9
    
    # Results
    save_checkpoints = True
    results_dir = 'results'
    checkpoint_dir = 'checkpoints'
    plots_dir = 'plots'
    
    def __str__(self):
        return f"ConfigCPU(n_layer={self.n_layer}, n_embd={self.n_embd}, max_iters={self.max_iters}, CPU-optimized)"
