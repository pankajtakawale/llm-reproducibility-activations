"""
Model factory functions for different architectures.
Makes it easy to plug in different models for reproducibility experiments.
"""
from model import CharLM
from model_tinylstm import TinyLSTM
from model_minigpt import MiniGPT
from model_convlm import ConvLM
from model_hybridlm import HybridLM
from model_nanotransformer import NanoTransformer


def charlm_factory(config, activation):
    """
    Factory for CharLM (character-level transformer).
    
    Args:
        config: Configuration object
        activation: Activation function name (string)
    
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


def tinylstm_factory(config, activation):
    """
    Factory for TinyLSTM (2-layer LSTM baseline).
    
    Args:
        config: Configuration object
        activation: Activation function name (string)
    
    Returns:
        TinyLSTM model instance
    """
    return TinyLSTM(config, activation)


def minigpt_factory(config, activation):
    """
    Factory for MiniGPT (tiny GPT-2 style transformer).
    
    Args:
        config: Configuration object
        activation: Activation function name (string)
    
    Returns:
        MiniGPT model instance
    """
    return MiniGPT(config, activation)


def convlm_factory(config, activation):
    """
    Factory for ConvLM (convolutional language model).
    
    Args:
        config: Configuration object
        activation: Activation function name (string)
    
    Returns:
        ConvLM model instance
    """
    return ConvLM(config, activation)


def hybridlm_factory(config, activation):
    """
    Factory for HybridLM (LSTM + Attention).
    
    Args:
        config: Configuration object
        activation: Activation function name (string)
    
    Returns:
        HybridLM model instance
    """
    return HybridLM(config, activation)


def nanotransformer_factory(config, activation):
    """
    Factory for NanoTransformer (minimal transformer without dropout).
    
    Args:
        config: Configuration object
        activation: Activation function name (string)
    
    Returns:
        NanoTransformer model instance
    """
    return NanoTransformer(config, activation)


def nanotransformer_factory(config, activation):
    """
    Factory for NanoTransformer (minimal transformer without dropout).
    
    Args:
        config: Configuration object
        activation: Activation function name (string)
    
    Returns:
        NanoTransformer model instance
    """
    return NanoTransformer(config, activation)


# Registry of available model factories
MODEL_REGISTRY = {
    'charlm': charlm_factory,
    'tinylstm': tinylstm_factory,
    'minigpt': minigpt_factory,
    'convlm': convlm_factory,
    'hybridlm': hybridlm_factory,
    'nanotransformer': nanotransformer_factory,
}


def get_model_factory(model_name):
    """
    Get a model factory by name.
    
    Args:
        model_name: Name of the model (e.g., 'charlm', 'tinylstm', 'minigpt', 'convlm', 'hybridlm', 'nanotransformer')
    
    Returns:
        Model factory function
    
    Raises:
        ValueError: If model_name not found in registry
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    
    return MODEL_REGISTRY[model_name]


def list_available_models():
    """Get list of all available model names."""
    return list(MODEL_REGISTRY.keys())


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

