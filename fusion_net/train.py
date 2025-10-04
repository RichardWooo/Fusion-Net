from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from .losses.feature import feature_similarity_loss
from .models.discriminator import PatchGANDiscriminator
from .models.generator import FusionGenerator


@dataclass
class TrainingConfig:
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    feature_loss_weight: float = 10.0
    noise_dim: int = 100
    feature_dim: int = 512
    device: str = "cuda"
    feature_loss_mode: str = "l1"


def hinge_d_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    return torch.relu(1.0 - real_scores).mean() + torch.relu(1.0 + fake_scores).mean()


def hinge_g_loss(fake_scores: Tensor) -> Tensor:
    return -fake_scores.mean()


class FeatureEncoder(nn.Module):
    """Simple encoder producing 512-dim features from 256x256 images."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 512) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for mult in [64, 128, 256, 512]:
            layers.extend(
                [
                    nn.Conv2d(channels, mult, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(mult),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = mult
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return self.head(x)


def train_gan(
    dataloader: DataLoader,
    generator: Optional[FusionGenerator] = None,
    discriminator: Optional[PatchGANDiscriminator] = None,
    encoder: Optional[nn.Module] = None,
    config: Optional[TrainingConfig] = None,
    num_steps: Optional[int] = None,
    noise_sampler: Optional[Callable[[int, int, str], Tensor]] = None,
) -> Dict[str, float]:
    """Train FusionNet generator and PatchGAN discriminator."""

    cfg = config or TrainingConfig()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    G = generator or FusionGenerator(noise_dim=cfg.noise_dim, feature_dim=cfg.feature_dim)
    D = discriminator or PatchGANDiscriminator()
    E = encoder or FeatureEncoder(feature_dim=cfg.feature_dim)

    G.to(device)
    D.to(device)
    E.to(device)

    for param in E.parameters():
        param.requires_grad_(False)

    opt_g = optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    stats = {"g_loss": 0.0, "d_loss": 0.0, "feature_loss": 0.0}
    total_steps = num_steps or len(dataloader)

    def sample_noise(batch_size: int, noise_dim: int, device: str) -> Tensor:
        return torch.randn(batch_size, noise_dim, device=device)

    sampler = noise_sampler or sample_noise

    for step, batch in enumerate(dataloader):
        if step >= total_steps:
            break
        real_images = batch[0] if isinstance(batch, (list, tuple)) else batch
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        with torch.no_grad():
            real_features = E(real_images)

        z = sampler(batch_size, cfg.noise_dim, device)
        fake_images = G(real_features, z)

        # Discriminator step
        opt_d.zero_grad(set_to_none=True)
        real_scores = D(real_images)
        fake_scores = D(fake_images.detach())
        d_loss = hinge_d_loss(real_scores, fake_scores)
        d_loss.backward()
        opt_d.step()

        # Generator step
        opt_g.zero_grad(set_to_none=True)
        fake_scores = D(fake_images)
        g_adv_loss = hinge_g_loss(fake_scores)
        fake_features = E(fake_images)
        f_loss = feature_similarity_loss(real_features, fake_features, mode=cfg.feature_loss_mode)
        g_loss = g_adv_loss + cfg.feature_loss_weight * f_loss
        g_loss.backward()
        opt_g.step()

        stats["g_loss"] += g_loss.item()
        stats["d_loss"] += d_loss.item()
        stats["feature_loss"] += f_loss.item()

    num = min(total_steps, len(dataloader))
    for key in stats:
        stats[key] /= max(1, num)
    return stats
