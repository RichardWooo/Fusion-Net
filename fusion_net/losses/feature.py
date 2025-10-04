from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


def feature_similarity_loss(
    real_features: Tensor,
    fake_features: Tensor,
    mode: Literal["l1", "cosine"] = "l1",
) -> Tensor:
    """Compute feature similarity loss between real and generated features."""

    if mode == "l1":
        return torch.mean(torch.abs(real_features - fake_features))
    if mode == "cosine":
        cos = nn.CosineEmbeddingLoss()
        target = real_features.new_ones(real_features.size(0))
        return cos(real_features, fake_features, target)
    raise ValueError(f"Unsupported mode: {mode}")
