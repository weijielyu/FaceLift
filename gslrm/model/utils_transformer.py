# Copyright (C) 2025, FaceLift Research Group
# https://github.com/weijielyu/FaceLift
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact: wlyu3@ucmerced.edu

"""
Transformer utilities for GSLRM.

This module contains the core transformer components used by the GSLRM model,
including self-attention, MLP layers, and transformer blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import xformers.ops as xops
except ImportError as e:
    print("Please install xformers to use flashatt v2")
    raise e


def _init_weights(module):
    """
    Initialize weights for transformer modules.
    
    Reference: https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L162-L168
    
    Args:
        module: Neural network module to initialize
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation.
    
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    """

    def __init__(
        self,
        d,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        mlp_dim=None,
    ):
        """
        Initialize MLP layer.
        
        Args:
            d: Input/output dimension
            mlp_ratio: Hidden dimension ratio (hidden_dim = d * mlp_ratio)
            mlp_bias: Whether to use bias in linear layers
            mlp_dropout: Dropout probability
            mlp_dim: Explicit hidden dimension (overrides mlp_ratio if provided)
        """
        super().__init__()
        if mlp_dim is None:
            mlp_dim = d * mlp_ratio
            
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim, bias=mlp_bias),
            nn.GELU(),
            nn.Linear(mlp_dim, d, bias=mlp_bias),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x):
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d)
            
        Returns:
            Output tensor of shape (batch, seq_len, d)
        """
        return self.mlp(x)


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with flash attention support.
    
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        d,
        d_head,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        use_flashatt_v2=True,
    ):
        """
        Initialize self-attention layer.
        
        Args:
            d: Token dimension
            d_head: Head dimension
            attn_qkv_bias: Whether to use bias in QKV projection
            attn_dropout: Attention dropout probability
            attn_fc_bias: Whether to use bias in output projection
            attn_fc_dropout: Output projection dropout probability
            use_flashatt_v2: Whether to use flash attention v2
        """
        super().__init__()
        assert d % d_head == 0, f"Token dimension {d} should be divisible by head dimension {d_head}"
        
        self.d = d
        self.d_head = d_head
        self.attn_dropout = attn_dropout
        self.use_flashatt_v2 = use_flashatt_v2

        # QKV projection (projects to 3*d for Q, K, V)
        self.to_qkv = nn.Linear(d, 3 * d, bias=attn_qkv_bias)
        
        # Output projection
        self.fc = nn.Linear(d, d, bias=attn_fc_bias)
        self.attn_fc_dropout = nn.Dropout(attn_fc_dropout)

    def forward(self, x, subset_attention_size=None):
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d)
            subset_attention_size: Optional size for subset attention
            
        Returns:
            Output tensor of shape (batch, seq_len, d)
        """
        # Generate Q, K, V
        q, k, v = self.to_qkv(x).split(self.d, dim=2)

        if self.use_flashatt_v2:
            # Use xformers flash attention
            q, k, v = map(
                lambda t: rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.d_head),
                (q, k, v),
            )

            if subset_attention_size is not None and subset_attention_size < q.shape[1]:
                # Handle subset attention for memory efficiency
                x_subset = xops.memory_efficient_attention(
                    q[:, :subset_attention_size, :, :].contiguous(),
                    k[:, :subset_attention_size, :, :].contiguous(),
                    v[:, :subset_attention_size, :, :].contiguous(),
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
                x_rest = xops.memory_efficient_attention(
                    q[:, subset_attention_size:, :, :].contiguous(),
                    k,
                    v,
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
                x = torch.cat([x_subset, x_rest], dim=1)
            else:
                # Standard flash attention
                x = xops.memory_efficient_attention(
                    q, k, v,
                    attn_bias=None,
                    op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                )
                
            x = rearrange(x, "b l nh dh -> b l (nh dh)")
        else:
            # Use PyTorch scaled dot product attention
            q, k, v = (
                rearrange(q, "b l (nh dh) -> b nh l dh", dh=self.d_head),
                rearrange(k, "b l (nh dh) -> b nh l dh", dh=self.d_head),
                rearrange(v, "b l (nh dh) -> b nh l dh", dh=self.d_head),
            )
            
            dropout_p = self.attn_dropout if self.training else 0.0
            
            if subset_attention_size is not None and subset_attention_size < q.shape[2]:
                # Handle subset attention
                x_subset = F.scaled_dot_product_attention(
                    q[:, :, :subset_attention_size, :].contiguous(),
                    k[:, :, :subset_attention_size, :].contiguous(),
                    v[:, :, :subset_attention_size, :].contiguous(),
                    dropout_p=dropout_p,
                )
                x_rest = F.scaled_dot_product_attention(
                    q[:, :, subset_attention_size:, :].contiguous(),
                    k, v,
                    dropout_p=dropout_p,
                )
                x = torch.cat([x_subset, x_rest], dim=2)
            else:
                # Standard attention
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
                
            x = rearrange(x, "b nh l dh -> b l (nh dh)")

        # Apply output projection and dropout
        return self.attn_fc_dropout(self.fc(x))


class TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-normalization.
    
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    """

    def __init__(
        self,
        d,
        d_head,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
    ):
        """
        Initialize transformer block.
        
        Args:
            d: Token dimension
            d_head: Attention head dimension
            ln_bias: Whether to use bias in layer norm
            attn_qkv_bias: Whether to use bias in attention QKV projection
            attn_dropout: Attention dropout probability
            attn_fc_bias: Whether to use bias in attention output projection
            attn_fc_dropout: Attention output dropout probability
            mlp_ratio: MLP hidden dimension ratio
            mlp_bias: Whether to use bias in MLP layers
            mlp_dropout: MLP dropout probability
        """
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d, bias=ln_bias)
        self.norm2 = nn.LayerNorm(d, bias=ln_bias)
        
        # Self-attention
        self.attn = SelfAttention(
            d=d,
            d_head=d_head,
            attn_qkv_bias=attn_qkv_bias,
            attn_dropout=attn_dropout,
            attn_fc_bias=attn_fc_bias,
            attn_fc_dropout=attn_fc_dropout,
        )
        
        # MLP
        self.mlp = MLP(
            d=d,
            mlp_ratio=mlp_ratio,
            mlp_bias=mlp_bias,
            mlp_dropout=mlp_dropout,
        )

    def forward(self, x, subset_attention_size=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d)
            subset_attention_size: Optional size for subset attention
            
        Returns:
            Output tensor of shape (batch, seq_len, d)
        """
        # Pre-norm attention with residual connection
        x = x + self.attn(self.norm1(x), subset_attention_size=subset_attention_size)
        
        # Pre-norm MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x