"""
TinyLSTM: Character-level LSTM language model.
Lightweight LSTM baseline for reproducibility experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import SmeLU


class TinyLSTM(nn.Module):
    """
    Simple LSTM-based character-level language model.
    
    Args:
        config: Configuration object with vocab_size, n_embd, n_layer, etc.
        activation_name: Name of activation function (affects output layer)
    """
    
    def __init__(self, config, activation_name='relu'):
        super().__init__()
        self.config = config
        self.activation_name = activation_name
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.n_embd,
            num_layers=config.n_layer,
            dropout=config.dropout if config.n_layer > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)
        
        # Activation function (applied before final projection)
        self.activation = self._get_activation(activation_name)
        
        # Initialize weights
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
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T), optional
            
        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        
        # Embed tokens
        x = self.embedding(idx)  # (B, T, n_embd)
        
        # LSTM forward
        x, _ = self.lstm(x)  # (B, T, n_embd)
        
        # Apply activation
        x = self.activation(x)
        
        # Project to vocabulary
        logits = self.fc_out(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def get_num_params(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
