from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor, nn


@dataclass
class CrossAttentionConfig:
    """Configuration for the cross-attention fusion module."""

    embed_dim: int = 512
    num_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.0
    noise_dim: int = 100


class CrossAttentionFusion(nn.Module):
    """Fuses a noise vector with an image feature vector via cross-attention.

    The noise vector ``z`` is treated as the query while the feature vector ``f_d``
    acts as the key/value pair. The resulting attended representation is passed
    through a position-wise feed-forward network with residual connections.
    """

    def __init__(self, config: Optional[CrossAttentionConfig] = None) -> None:
        super().__init__()
        self.config = config or CrossAttentionConfig()
        self.query_proj = nn.Linear(self.config.noise_dim, self.config.embed_dim)
        self.key_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim)
        self.value_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.config.embed_dim)

        ff_dim = self.config.embed_dim * self.config.ff_multiplier
        self.feed_forward = nn.Sequential(
            nn.Linear(self.config.embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(ff_dim, self.config.embed_dim),
        )
        self.ff_norm = nn.LayerNorm(self.config.embed_dim)

    def forward(self, f_d: Tensor, z: Tensor) -> Tensor:
        """Run cross-attention fusion.

        Args:
            f_d: Tensor with shape ``(batch, embed_dim)`` containing image features.
            z: Tensor with shape ``(batch, noise_dim)`` representing noise samples.

        Returns:
            Tensor with shape ``(batch, embed_dim)`` containing fused features.
        """

        if f_d.dim() != 2 or z.dim() != 2:
            raise ValueError("Inputs must be 2D tensors (batch, feature_dim).")

        # Project to attention space and add a sequence dimension of length 1.
        query = self.query_proj(z).unsqueeze(1)
        key = self.key_proj(f_d).unsqueeze(1)
        value = self.value_proj(f_d).unsqueeze(1)

        attn_out, _ = self.attn(query, key, value)
        attn_out = self.attn_norm(attn_out + query)

        ff_out = self.feed_forward(attn_out)
        ff_out = self.ff_norm(ff_out + attn_out)

        return ff_out.squeeze(1)
