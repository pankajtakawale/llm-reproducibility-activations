"""
Lightweight GPU Configuration for LLM Reproducibility Testing
- Similar model size to CPU config (~0.5M parameters)
- GPU-optimized batch size (64 vs CPU's 32)
- Faster iteration for quick testing and validation
"""

class ConfigGPULite:
    """Lightweight GPU configuration for fast testing with small models"""
    
    # Data
    dataset = 'shakespeare_char'
    
    # Experiment settings
    models = ['charlm']
    activations = ['relu', 'gelu', 'swish']
    trials_per_activation = 3
    
    # Model architecture - lightweight (similar to CPU config)
    vocab_size = None  # Will be set after loading data
    n_embd = 128       # Embedding dimension (vs 384 in full GPU)
    n_head = 4         # Number of attention heads (vs 6 in full GPU)
    n_layer = 2        # Number of transformer layers (vs 6 in full GPU)
    block_size = 128   # Context length (vs 256 in full GPU)
    dropout = 0.2      # Dropout rate
    
    # Training hyperparameters - GPU optimized batch size
    batch_size = 64    # Larger than CPU (32) since GPU can handle it
    max_iters = 200    # Quick testing iterations
    learning_rate = 3e-4
    eval_interval = 50  # Evaluate every 50 steps for progress tracking
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
        return f"ConfigGPULite(n_layer={self.n_layer}, n_embd={self.n_embd}, batch_size={self.batch_size}, max_iters={self.max_iters})"
