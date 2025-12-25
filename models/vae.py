"""
Variational Autoencoder implementation for sprite generation.

This module contains the main SpriteVAE class that combines the encoder
and decoder networks with the VAE probabilistic framework.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from spriteforge.models.base import BaseVAE
from spriteforge.models.encoder_decoder import ConvDecoder, ConvEncoder


@dataclass
class VAEOutput:
    """
    Container for VAE forward pass outputs.
    
    Attributes:
        reconstruction: Reconstructed input image.
        mu: Mean of the latent distribution.
        log_var: Log variance of the latent distribution.
        z: Sampled latent vector.
    """
    
    reconstruction: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    z: torch.Tensor


class SpriteVAE(BaseVAE):
    """
    Variational Autoencoder for 2D pixel art sprite generation.
    
    This VAE learns a compressed latent representation of sprites and can
    both reconstruct existing sprites and generate new ones by sampling
    from the learned latent space.
    
    The architecture uses convolutional layers optimized for small pixel art
    images (32x32 or 64x64 RGBA sprites).
    
    Key Features:
        - Convolutional encoder/decoder for spatial feature preservation
        - Reparameterization trick for backpropagation through sampling
        - Configurable latent dimension for quality/compression tradeoff
        - Support for RGBA (4 channel) images
    
    Attributes:
        image_size: Size of input/output images (assumes square).
        in_channels: Number of image channels (4 for RGBA).
        latent_dim: Dimension of the latent space.
        encoder: The encoder network.
        decoder: The decoder network.
    
    Example:
        >>> # Create model
        >>> vae = SpriteVAE(image_size=32, latent_dim=128)
        >>> 
        >>> # Forward pass (for training)
        >>> x = torch.randn(16, 4, 32, 32)  # batch of sprites
        >>> output = vae(x)
        >>> print(output.reconstruction.shape)  # torch.Size([16, 4, 32, 32])
        >>> 
        >>> # Generate new sprites
        >>> new_sprites = vae.generate(num_samples=8)
        >>> print(new_sprites.shape)  # torch.Size([8, 4, 32, 32])
    """
    
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 4,
        latent_dim: int = 128,
        base_channels: int = 64,
    ) -> None:
        """
        Initialize the Sprite VAE.
        
        Args:
            image_size: Input/output image size (32 or 64 recommended).
            in_channels: Number of input channels (4 for RGBA sprites).
            latent_dim: Dimension of the latent space (128-256 recommended).
            base_channels: Base channel count for conv layers.
        
        Raises:
            ValueError: If image_size is not a power of 2 or < 16.
        """
        super().__init__()
        
        # Validate image size
        if image_size < 16 or (image_size & (image_size - 1)) != 0:
            raise ValueError(
                f"image_size must be a power of 2 >= 16, got {image_size}"
            )
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Create encoder and decoder
        self.encoder = ConvEncoder(
            image_size=image_size,
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
        
        self.decoder = ConvDecoder(
            image_size=image_size,
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
        )
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images to latent distribution parameters.
        
        Args:
            x: Input images of shape (batch, channels, height, width).
        
        Returns:
            Tuple of (mu, log_var):
                - mu: Mean of latent distribution, shape (batch, latent_dim).
                - log_var: Log variance, shape (batch, latent_dim).
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.
        
        Args:
            z: Latent vectors of shape (batch, latent_dim).
        
        Returns:
            Reconstructed images of shape (batch, channels, height, width).
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Apply the reparameterization trick for backpropagation.
        
        Instead of sampling z ~ N(mu, sigma), we sample epsilon ~ N(0, 1)
        and compute z = mu + sigma * epsilon. This allows gradients to flow
        through the sampling operation.
        
        Args:
            mu: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.
        
        Returns:
            Sampled latent vector z = mu + std * epsilon.
        """
        # Compute standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        
        # Sample epsilon from standard normal
        epsilon = torch.randn_like(std)
        
        # Reparameterization: z = mu + std * epsilon
        z = mu + std * epsilon
        
        return z
    
    def forward(self, x: torch.Tensor) -> VAEOutput:
        """
        Full forward pass through the VAE.
        
        Encodes input to latent distribution, samples using reparameterization,
        and decodes back to image space.
        
        Args:
            x: Input images of shape (batch, channels, height, width).
        
        Returns:
            VAEOutput containing reconstruction, mu, log_var, and z.
        """
        # Encode to distribution parameters
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode to image
        reconstruction = self.decode(z)
        
        return VAEOutput(
            reconstruction=reconstruction,
            mu=mu,
            log_var=log_var,
            z=z,
        )
    
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generate new sprite images by sampling from the prior.
        
        Samples latent vectors from a standard normal distribution N(0, I)
        and decodes them to generate new sprites.
        
        Args:
            num_samples: Number of sprites to generate.
        
        Returns:
            Generated images of shape (num_samples, channels, height, width).
            Values are in [0, 1] range.
        """
        # Sample from standard normal prior
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        
        # Decode to images
        with torch.no_grad():
            generated = self.decode(z)
        
        return generated
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Interpolate between two sprites in latent space.
        
        Encodes both sprites, linearly interpolates their latent representations,
        and decodes the interpolated points to create a smooth transition.
        
        Args:
            x1: First sprite of shape (1, channels, height, width).
            x2: Second sprite of shape (1, channels, height, width).
            num_steps: Number of interpolation steps (including endpoints).
        
        Returns:
            Interpolated images of shape (num_steps, channels, height, width).
        """
        with torch.no_grad():
            # Encode both sprites
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)
            
            # Create interpolation weights
            alphas = torch.linspace(0, 1, num_steps, device=self.device)
            
            # Interpolate in latent space
            interpolated_z = torch.stack([
                (1 - alpha) * mu1 + alpha * mu2 for alpha in alphas
            ]).squeeze(1)
            
            # Decode all interpolated points
            interpolated_images = self.decode(interpolated_z)
        
        return interpolated_images
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input sprites (encode then decode).
        
        This is useful for evaluating reconstruction quality without
        needing the full VAEOutput.
        
        Args:
            x: Input sprites of shape (batch, channels, height, width).
        
        Returns:
            Reconstructed sprites with same shape as input.
        """
        with torch.no_grad():
            output = self.forward(x)
        return output.reconstruction
