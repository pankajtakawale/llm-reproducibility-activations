"""
NanoTransformer: Minimal transformer without dropout.
Tests if regularization affects reproducibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from activations import SmeLU


class NanoAttention(nn.Module):
    """Simple multi-head self-attention."""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)
        
        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.proj(y)
        return y


class NanoBlock(nn.Module):
    """Transformer block without dropout."""
    
    def __init__(self, n_embd, n_head, activation):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = NanoAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            activation,
            nn.Linear(4 * n_embd, n_embd)
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoTransformer(nn.Module):
    """
    Minimal transformer without dropout for reproducibility testing.
    
    Args:
        config: Configuration object
        activation_name: Activation function name
    """
    
    def __init__(self, config, activation_name='relu'):
        super().__init__()
        self.config = config
        self.activation_name = activation_name
        
        # Embeddings (no dropout)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        
        # Get activation
        activation = self._get_activation(activation_name)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            NanoBlock(config.n_embd, config.n_head, activation)
            for _ in range(config.n_layer)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
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
        assert T <= self.config.block_size
        
        # Embeddings
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
