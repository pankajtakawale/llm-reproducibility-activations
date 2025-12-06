"""
ConvLM: Convolutional Language Model.
Uses dilated convolutions for character-level modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import SmeLU


class ConvBlock(nn.Module):
    """Convolutional block with residual connection."""
    
    def __init__(self, channels, kernel_size, dilation, activation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, 
                             padding=padding, dilation=dilation)
        self.activation = activation
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x: (B, C, T)
        residual = x
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C) for LayerNorm
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.activation(x)
        return x + residual


class ConvLM(nn.Module):
    """
    Convolutional language model with dilated convolutions.
    
    Args:
        config: Configuration object
        activation_name: Activation function name
    """
    
    def __init__(self, config, activation_name='relu'):
        super().__init__()
        self.config = config
        self.activation_name = activation_name
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Get activation
        activation = self._get_activation(activation_name)
        
        # Convolutional blocks with increasing dilation
        self.conv_blocks = nn.ModuleList([
            ConvBlock(config.n_embd, kernel_size=3, dilation=1, activation=activation),
            ConvBlock(config.n_embd, kernel_size=3, dilation=2, activation=activation),
            ConvBlock(config.n_embd, kernel_size=3, dilation=4, activation=activation),
            ConvBlock(config.n_embd, kernel_size=3, dilation=8, activation=activation),
            ConvBlock(config.n_embd, kernel_size=3, dilation=16, activation=activation),
            ConvBlock(config.n_embd, kernel_size=3, dilation=1, activation=activation),
        ])
        
        # Output projection
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize
        self.apply(self._init_weights)
        
    def _get_activation(self, name):
        """Get activation function by name."""
        if name == 'smelu_05':
            return SmeLU(beta=0.5)
        elif name == 'smelu_1':
            return SmeLU(beta=1.0)
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'swish':
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target indices (B, T), optional
            
        Returns:
            logits: (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        # Embed
        x = self.embedding(idx)  # (B, T, C)
        x = self.dropout(x)
        
        # Transpose for conv1d: (B, C, T)
        x = x.transpose(1, 2)
        
        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Transpose back: (B, T, C)
        x = x.transpose(1, 2)
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
