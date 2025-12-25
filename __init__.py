"""
SpriteForge: A Variational Autoencoder for 2D Pixel Art Sprite Generation.

SpriteForge is an educational deep learning project that uses VAE architecture
to learn latent representations of 2D sprites and generate new pixel art.

Example:
    >>> from spriteforge import SpriteVAE, SpriteDataset
    >>> model = SpriteVAE(image_size=32, latent_dim=128)
    >>> # Train with your sprite dataset
    >>> new_sprites = model.generate(num_samples=10)
"""

__version__ = "0.1.0"
__author__ = "cmelnulabs"

from spriteforge.models.vae import SpriteVAE
from spriteforge.data.dataset import SpriteDataset
from spriteforge.training.trainer import Trainer

__all__ = [
    "SpriteVAE",
    "SpriteDataset", 
    "Trainer",
    "__version__",
]
