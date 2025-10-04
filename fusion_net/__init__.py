"""Fusion-Net package providing generator, discriminator, training utilities, and metrics."""

from .models.generator import FusionGenerator
from .models.discriminator import PatchGANDiscriminator
from .models.attention import CrossAttentionFusion, CrossAttentionConfig
from .losses.feature import feature_similarity_loss
from .metrics.fid import compute_fid
from .train import train_gan, TrainingConfig
from .visualization.grad_cam import grad_cam

__all__ = [
    "FusionGenerator",
    "PatchGANDiscriminator",
    "CrossAttentionFusion",
    "CrossAttentionConfig",
    "feature_similarity_loss",
    "compute_fid",
    "train_gan",
    "TrainingConfig",
    "grad_cam",
]
