"""
Model factory functions for different architectures.
Makes it easy to plug in different models for reproducibility experiments.
"""
from model import CharLM


def charlm_factory(config, activation):
    """
    Factory for CharLM (character-level transformer).
    
    Args:
        config: Configuration object
        activation: Activation function
    
    Returns:
        CharLM model instance
    """
    return CharLM(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        activation=activation,
        dropout=config.dropout
    )


def gpt2_factory(config, activation):
    """
    Factory for GPT-2 style model.
    
    Args:
        config: Configuration object
        activation: Activation function
    
    Returns:
        GPT-2 model instance
    
    Note: You'll need to implement GPT2 class in model.py or import from transformers
    """
    # Example implementation - adjust based on your GPT-2 implementation
    try:
        from model import GPT2
        return GPT2(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            block_size=config.block_size,
            activation=activation,
            dropout=config.dropout
        )
    except ImportError:
        raise NotImplementedError("GPT2 model not yet implemented. Add GPT2 class to model.py")


def nanogpt_factory(config, activation):
    """
    Factory for NanoGPT style model.
    
    Args:
        config: Configuration object
        activation: Activation function
    
    Returns:
        NanoGPT model instance
    """
    try:
        from model import NanoGPT
        return NanoGPT(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            block_size=config.block_size,
            activation=activation,
            dropout=config.dropout
        )
    except ImportError:
        raise NotImplementedError("NanoGPT model not yet implemented. Add NanoGPT class to model.py")


def babygpt_factory(config, activation):
    """
    Factory for BabyGPT style model (smaller/simpler variant).
    
    Args:
        config: Configuration object
        activation: Activation function
    
    Returns:
        BabyGPT model instance
    """
    try:
        from model import BabyGPT
        return BabyGPT(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            block_size=config.block_size,
            activation=activation,
            dropout=config.dropout
        )
    except ImportError:
        raise NotImplementedError("BabyGPT model not yet implemented. Add BabyGPT class to model.py")


# Registry of available model factories
MODEL_REGISTRY = {
    'charlm': charlm_factory,
    'gpt2': gpt2_factory,
    'nanogpt': nanogpt_factory,
    'babygpt': babygpt_factory,
}


def get_model_factory(model_name):
    """
    Get a model factory by name.
    
    Args:
        model_name: Name of the model ('charlm', 'gpt2', 'nanogpt', 'babygpt')
    
    Returns:
        Model factory function
    
    Raises:
        ValueError: If model_name not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    
    return MODEL_REGISTRY[model_name]


def register_model_factory(name, factory_func):
    """
    Register a custom model factory.
    
    Args:
        name: Name for the model
        factory_func: Factory function with signature (config, activation) -> model
    
    Example:
        >>> def my_custom_model_factory(config, activation):
        ...     return MyCustomModel(...)
        >>> register_model_factory('mymodel', my_custom_model_factory)
        >>> run_experiment(config, 'relu', model_factory=get_model_factory('mymodel'))
    """
    MODEL_REGISTRY[name] = factory_func
    print(f"Registered model factory: {name}")
