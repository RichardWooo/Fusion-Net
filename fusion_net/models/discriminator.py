from __future__ import annotations

from torch import Tensor, nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator producing a 30x30 score map for 256x256 inputs."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for idx, mult in enumerate([1, 2, 4, 8]):
            layers.append(
                nn.Conv2d(
                    channels,
                    base_channels * mult,
                    kernel_size=4,
                    stride=2 if idx < 3 else 1,
                    padding=1,
                )
            )
            if idx > 0:
                layers.append(nn.InstanceNorm2d(base_channels * mult, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = base_channels * mult

        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))
