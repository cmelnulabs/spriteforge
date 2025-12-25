"""
SpriteForge model implementations.

This package contains all neural network architectures used in SpriteForge,
including the VAE and its encoder/decoder components.
"""

from spriteforge.models.base import BaseDecoder, BaseEncoder, BaseModel, BaseVAE
from spriteforge.models.encoder_decoder import ConvDecoder, ConvEncoder
from spriteforge.models.vae import SpriteVAE, VAEOutput

__all__ = [
    # Base classes
    "BaseModel",
    "BaseEncoder",
    "BaseDecoder",
    "BaseVAE",
    # Implementations
    "ConvEncoder",
    "ConvDecoder",
    "SpriteVAE",
    "VAEOutput",
]
