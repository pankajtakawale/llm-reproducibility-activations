"""
Full GPU Configuration for Publication-Quality Results
- 6 layers, 384 hidden units (10.8M parameters)
- 5000 iterations for full convergence
- Production/publication-quality training
"""

class ConfigFullGPU:
    """Full GPU configuration for publication-quality experiments"""
    
    # Data
    dataset = 'shakespeare_char'
    
    # Experiment settings
    models = ['nanotransformer']
    activations = ['relu', 'gelu', 'swish']  # Standard activations only
    trials_per_activation = 3
    
    # Model architecture - Full model (production scale)
    vocab_size = None  # Will be set after loading data
    n_embd = 384       # Embedding dimension (full)
    n_head = 6         # Number of attention heads (full)
    n_layer = 6        # Number of transformer layers (full)
    block_size = 256   # Context length (characters)
    dropout = 0.2      # Dropout rate
    
    # Training hyperparameters - Full training
    batch_size = 64
    max_iters = 5000   # Full training iterations for publication quality
    learning_rate = 3e-4
    eval_interval = 500  # Evaluate every 500 steps
    eval_iters = 200
    
    # Reproducibility settings
    seed_base = 42     # Base seed (will increment for each trial)
    
    # Device
    device = 'cuda'    # GPU execution
    
    # Data
    dataset = 'shakespeare'
    train_split = 0.9
    
    # Results
    save_checkpoints = True
    results_dir = 'results'
    checkpoint_dir = 'checkpoints'
    plots_dir = 'plots'
    
    def __repr__(self):
        return f"ConfigFullGPU(n_layer={self.n_layer}, n_embd={self.n_embd}, batch_size={self.batch_size}, max_iters={self.max_iters})"
