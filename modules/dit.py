"""Diffusion Transformer (DiT) backbone for image generation.

DiT replaces the traditional U-Net backbone with a Transformer architecture,
achieving state-of-the-art image generation quality.

Reference:
    Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023.
"""

from __future__ import annotations

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm (adaLN-Zero)."""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )
        
        # adaLN-Zero: 6 modulation parameters (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        
        # Initialize last layer to zero for residual
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tokens
            c: [B, D] conditioning (time + condition embedding)
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with modulation
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class DiTBlockWithContext(nn.Module):
    """DiT block with attention bias and context tokens support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _attn(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn = attn + attn_bias.unsqueeze(1)
            else:
                attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj_drop(self.proj(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self._attn(x_norm, attn_bias=attn_bias)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    """Final layer for DiT: LayerNorm + Linear projection."""
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        
        # Zero init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiT2D(nn.Module):
    """Diffusion Transformer for 2D image generation with conditioning.
    
    This model takes a noisy image and condition image, and predicts the noise
    (for diffusion) or velocity (for flow matching).
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize DiT.
        
        Args:
            img_size: Input image size (assumes square)
            patch_size: Patch size for embedding
            in_channels: Number of input channels
            out_channels: Number of output channels
            cond_channels: Number of conditioning channels
            hidden_size: Transformer hidden dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.hidden_size = hidden_size
        
        # Input embeddings
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.cond_embedder = PatchEmbed(img_size, patch_size, cond_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize pos_embed
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embeddings
        for embedder in [self.x_embedder, self.cond_embedder]:
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view(w.size(0), -1))
            nn.init.zeros_(embedder.proj.bias)
        
        # Initialize transformer blocks
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to image.
        
        Args:
            x: [B, N, P*P*C] patch tokens
            
        Returns:
            [B, C, H, W] image
        """
        B = x.shape[0]
        h = w = int(self.num_patches ** 0.5)
        p = self.patch_size
        c = self.out_channels
        
        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(B, c, h * p, w * p)
        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass for conditional DiT.

        Args:
            x: Noisy input [B, C, H, W]
            cond: Conditioning image [B, Cc, H, W]
            t: Timestep [B]

        Returns:
            Predicted output [B, C, H, W]
        """
        x_emb = self.x_embedder(x)
        c_emb = self.cond_embedder(cond)
        t_emb = self.t_embedder(t)

        tokens = x_emb + c_emb + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens, t_emb)

        tokens = self.final_layer(tokens, t_emb)
        return self.unpatchify(tokens)


class DiT2DWithContext(nn.Module):
    """DiT for 2D image generation with extra context tokens and attention bias."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        context_tokens: int = 1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.hidden_size = hidden_size
        self.context_tokens = max(0, int(context_tokens))

        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.cond_embedder = PatchEmbed(img_size, patch_size, cond_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        if self.context_tokens > 0:
            self.context_pos_embed = nn.Parameter(torch.zeros(1, self.context_tokens, hidden_size))
        else:
            self.context_pos_embed = None

        self.blocks = nn.ModuleList([
            DiTBlockWithContext(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        if self.context_pos_embed is not None:
            nn.init.normal_(self.context_pos_embed, std=0.02)

        for embedder in [self.x_embedder, self.cond_embedder]:
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view(w.size(0), -1))
            nn.init.zeros_(embedder.proj.bias)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = w = int(self.num_patches ** 0.5)
        p = self.patch_size
        c = self.out_channels
        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(B, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_emb = self.x_embedder(x) + self.pos_embed
        c_emb = self.cond_embedder(cond)
        t_emb = self.t_embedder(t)

        tokens = x_emb + c_emb + self.pos_embed

        if context_tokens is not None and self.context_tokens > 0:
            if context_tokens.dim() == 2:
                context_tokens = context_tokens.unsqueeze(1)
            if context_tokens.size(1) != self.context_tokens:
                if context_tokens.size(1) == 1:
                    context_tokens = context_tokens.repeat(1, self.context_tokens, 1)
                else:
                    context_tokens = context_tokens[:, :self.context_tokens]
            if self.context_pos_embed is not None:
                context_tokens = context_tokens + self.context_pos_embed[:, :context_tokens.size(1)]
            tokens = torch.cat([tokens, context_tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens, t_emb, attn_bias=attn_bias)

        tokens = tokens[:, :self.num_patches]
        tokens = self.final_layer(tokens, t_emb)
        return self.unpatchify(tokens)

class DiT2DConditioned(nn.Module):
    """DiT with condition concatenation using a unified forward interface."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        img_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs  # Ignore extra args for compatibility
    ):
        super().__init__()
        self.dit = DiT2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with unified conditioned interface."""
        return self.dit(x, cond, t)


class DiT2DConditionedWithContext(nn.Module):
    """DiT with extra context tokens and attention bias support."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        img_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        context_tokens: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.dit = DiT2DWithContext(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            cond_channels=cond_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            context_tokens=context_tokens,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.dit(x, cond, t, context_tokens=context_tokens, attn_bias=attn_bias)


# ========================= DiT Size Configurations =========================

def DiT_S(img_size=224, patch_size=16, **kwargs):
    """DiT-Small: 12 layers, 384 hidden, 6 heads (22M params)."""
    return DiT2DConditioned(
        img_size=img_size, patch_size=patch_size,
        hidden_size=384, depth=12, num_heads=6, **kwargs
    )

def DiT_B(img_size=224, patch_size=16, **kwargs):
    """DiT-Base: 12 layers, 768 hidden, 12 heads (86M params)."""
    return DiT2DConditioned(
        img_size=img_size, patch_size=patch_size,
        hidden_size=768, depth=12, num_heads=12, **kwargs
    )

def DiT_L(img_size=224, patch_size=16, **kwargs):
    """DiT-Large: 24 layers, 1024 hidden, 16 heads (304M params)."""
    return DiT2DConditioned(
        img_size=img_size, patch_size=patch_size,
        hidden_size=1024, depth=24, num_heads=16, **kwargs
    )

def DiT_XL(img_size=224, patch_size=16, **kwargs):
    """DiT-XL: 28 layers, 1152 hidden, 16 heads (458M params)."""
    return DiT2DConditioned(
        img_size=img_size, patch_size=patch_size,
        hidden_size=1152, depth=28, num_heads=16, **kwargs
    )
