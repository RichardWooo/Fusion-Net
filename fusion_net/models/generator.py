from __future__ import annotations

from dataclasses import replace
from typing import Optional

from torch import Tensor, nn

from .attention import CrossAttentionConfig, CrossAttentionFusion


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False) -> None:
        super().__init__()
        self.upsample = upsample
        self.learned_shortcut = in_channels != out_channels or upsample

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest") if upsample else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if self.learned_shortcut:
            shortcut_layers = []
            if upsample:
                shortcut_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            shortcut_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + self.shortcut(x)


class FusionGenerator(nn.Module):
    """Generator that fuses feature and noise vectors via cross-attention."""

    def __init__(
        self,
        noise_dim: int = 100,
        feature_dim: int = 512,
        base_channels: int = 64,
        attention_config: Optional[CrossAttentionConfig] = None,
    ) -> None:
        super().__init__()
        if base_channels % 4 != 0:
            raise ValueError("base_channels must be divisible by 4 to reach RGB output.")
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        if attention_config is None:
            attention_config = CrossAttentionConfig(noise_dim=noise_dim, embed_dim=feature_dim)
        else:
            if attention_config.noise_dim != noise_dim or attention_config.embed_dim != feature_dim:
                attention_config = replace(
                    attention_config,
                    noise_dim=noise_dim,
                    embed_dim=feature_dim,
                )
        self.fusion = CrossAttentionFusion(attention_config)

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 8 * base_channels * 4 * 4),
            nn.BatchNorm1d(8 * base_channels * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.initial_channels = 8 * base_channels
        self.blocks = nn.Sequential(
            ResidualBlock(8 * base_channels, 8 * base_channels, upsample=True),
            ResidualBlock(8 * base_channels, 4 * base_channels, upsample=True),
            ResidualBlock(4 * base_channels, 2 * base_channels, upsample=True),
            ResidualBlock(2 * base_channels, base_channels, upsample=True),
            ResidualBlock(base_channels, base_channels // 2, upsample=True),
            ResidualBlock(base_channels // 2, base_channels // 4, upsample=True),
        )

        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 4, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, f_d: Tensor, z: Tensor) -> Tensor:
        fused = self.fusion(f_d, z)
        x = self.fc(fused)
        x = x.view(x.size(0), self.initial_channels, 4, 4)
        x = self.blocks(x)
        return self.to_rgb(x)
