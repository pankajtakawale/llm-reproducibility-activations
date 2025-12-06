"""
MiniGPT: Tiny GPT-2 style transformer.
Slightly larger than CharLM but still CPU-friendly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from activations import SmeLU


class MiniGPTBlock(nn.Module):
    """Transformer block with multi-head attention."""
    
    def __init__(self, config, activation):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(
            config.n_embd, 
            config.n_head,
            dropout=config.dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            activation,
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x, mask=None):
        # Attention with residual
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    Mini GPT-2 style model for character-level language modeling.
    
    Args:
        config: Configuration object
        activation_name: Activation function name
    """
    
    def __init__(self, config, activation_name='relu'):
        super().__init__()
        self.config = config
        self.activation_name = activation_name
        
        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        # Get activation function
        activation = self._get_activation(activation_name)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MiniGPTBlock(config, activation) 
            for _ in range(config.n_layer)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
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
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
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
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} > block size {self.config.block_size}"
        
        # Token + position embeddings
        tok_emb = self.tok_emb(idx)  # (B, T, n_embd)
        pos_emb = self.pos_emb[:, :T, :]  # (1, T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Causal mask for attention
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        mask = mask.masked_fill(mask == True, float('-inf'))
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
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
