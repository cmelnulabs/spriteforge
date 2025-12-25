"""
SpriteForge training utilities.

This package provides training infrastructure including loss functions,
the main Trainer class, and training configuration.
"""

from spriteforge.training.losses import (
    VAELoss,
    kl_divergence,
    reconstruction_loss,
    vae_loss,
)
from spriteforge.training.trainer import TrainConfig, Trainer, TrainMetrics

__all__ = [
    # Loss functions
    "reconstruction_loss",
    "kl_divergence",
    "vae_loss",
    "VAELoss",
    # Trainer
    "Trainer",
    "TrainConfig",
    "TrainMetrics",
]
