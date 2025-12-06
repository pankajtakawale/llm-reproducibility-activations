"""
HybridLM: LSTM + Attention hybrid model.
Combines recurrent and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import SmeLU


class HybridLM(nn.Module):
    """
    Hybrid LSTM + Attention language model.
    
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
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.n_embd,
            num_layers=config.n_layer,
            dropout=config.dropout if config.n_layer > 1 else 0,
            batch_first=True
        )
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            config.n_embd,
            config.n_head,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Feed-forward
        self.activation = self._get_activation(activation_name)
        self.ff = nn.Sequential(
            nn.Linear(config.n_embd, 2 * config.n_embd),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
        
        # Output
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)
        
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
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
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
        B, T = idx.shape
        
        # Embed
        x = self.embedding(idx)  # (B, T, n_embd)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        x = self.ln1(lstm_out)
        
        # Self-attention with causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        mask = mask.masked_fill(mask == True, float('-inf'))
        
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + attn_out  # Residual
        
        # Feed-forward
        x = self.ln2(x)
        x = x + self.ff(x)  # Residual
        
        # Output
        logits = self.fc_out(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
