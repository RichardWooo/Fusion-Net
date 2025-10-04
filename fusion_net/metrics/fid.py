from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import inception_v3


def _get_inception(device: torch.device) -> nn.Module:
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def _compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(
    real_loader: DataLoader,
    fake_loader: DataLoader,
    device: str = "cuda",
) -> float:
    """Compute Fréchet Inception Distance between two image sets."""

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    inception = _get_inception(dev)

    def gather_features(loader: DataLoader) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for batch in loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(dev)
                logits = inception(images)
                feats.append(logits.cpu().numpy())
        return np.concatenate(feats, axis=0)

    real_features = gather_features(real_loader)
    fake_features = gather_features(fake_loader)
    mu1, sigma1 = _compute_statistics(real_features)
    mu2, sigma2 = _compute_statistics(fake_features)
    return _frechet_distance(mu1, sigma1, mu2, sigma2)
