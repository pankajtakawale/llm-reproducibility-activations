"""
Activation functions for reproducibility experiments.
Implements SmeLU, ReLU, GELU, and Swish following the original project's style.
"""
import torch
import torch.nn as nn
import math


class SmeLU(nn.Module):
    """
    Smooth ReLU (SmeLU) activation function.
    
    SmeLU matches both sides of ReLU with a middle quadratic region,
    providing smooth gradients and potentially better reproducibility.
    
    Reference: "Smooth activations and reproducibility in deep networks"
    """
    
    def __init__(self, beta=1.0):
        """
        Initialize SmeLU.
        
        Args:
            beta: Controls the width of the smooth region (default: 1.0)
        """
        super(SmeLU, self).__init__()
        self.beta = beta
    
    def forward(self, x):
        """
        Apply SmeLU activation.
        
        For x <= -beta: output = 0
        For -beta < x < beta: output = smooth quadratic interpolation
        For x >= beta: output = x
        """
        # Regions
        if self.beta == 0:
            return torch.relu(x)
        
        # Compute smooth region
        # SmeLU(x) = 0 if x <= -beta
        #          = (x + beta)^2 / (4*beta) if -beta < x < beta
        #          = x if x >= beta
        
        result = torch.where(
            x <= -self.beta,
            torch.zeros_like(x),
            torch.where(
                x >= self.beta,
                x,
                (x + self.beta) ** 2 / (4 * self.beta)
            )
        )
        
        return result
    
    def __repr__(self):
        return f"SmeLU(beta={self.beta})"


class Swish(nn.Module):
    """
    Swish activation function (also known as SiLU).
    Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)
    
    def __repr__(self):
        return "Swish()"


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) activation function.
    Used in LLaMA, PaLM, and other state-of-the-art LLMs.
    
    SwiGLU(x) = Swish(xW) ⊗ xV
    where W and V are learned projections and ⊗ is element-wise product.
    
    For single-layer usage (like in our experiments), we split the input
    and apply: SwiGLU(x) = Swish(x_gate) * x_value
    
    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """
    
    def __init__(self):
        super(SwiGLU, self).__init__()
        self.swish = Swish()
    
    def forward(self, x):
        """
        Apply SwiGLU activation.
        Expects input dimension to be even (will be split in half).
        """
        # Split input into two halves for gating
        x_gate, x_value = x.chunk(2, dim=-1)
        return self.swish(x_gate) * x_value
    
    def __repr__(self):
        return "SwiGLU()"


def get_activation(name):
    """
    Get activation function by name.
    
    Args:
        name: One of 'smelu_05', 'smelu_1', 'relu', 'gelu', 'swish', 'swiglu'
    
    Returns:
        Activation module
    """
    activations = {
        'smelu_05': SmeLU(beta=0.5),
        'smelu_1': SmeLU(beta=1.0),
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'swish': Swish(),
        'swiglu': SwiGLU()
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    
    return activations[name]


if __name__ == '__main__':
    # Test activations
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = torch.linspace(-3, 3, 1000)
    
    activations = {
        'SmeLU(0.5)': SmeLU(beta=0.5),
        'SmeLU(1.0)': SmeLU(beta=1.0),
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'Swish': Swish()
    }
    
    plt.figure(figsize=(10, 6))
    for name, activation in activations.items():
        y = activation(x)
        plt.plot(x.numpy(), y.numpy(), label=name, linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('activation(x)')
    plt.title('Activation Functions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('activations_comparison.png', dpi=150)
    print("[INFO] Saved activation comparison plot to activations_comparison.png")
