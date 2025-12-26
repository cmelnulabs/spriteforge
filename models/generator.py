"""
Generator network for text-to-sprite GAN.

This module implements the Generator that creates sprite images
from random noise and text embeddings.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from spriteforge.models.base import BaseGenerator


@dataclass
class GeneratorOutput:
    """
    Container for Generator outputs.
    
    Attributes:
        images: Generated sprite images.
        noise: Input noise used for generation.
        text_embedding: Text embedding used for conditioning.
    """
    
    images: torch.Tensor
    noise: torch.Tensor
    text_embedding: torch.Tensor


class SpriteGenerator(BaseGenerator):
    """
    Text-conditional Generator for sprite images.
    
    Generates sprites from random noise conditioned on text embeddings.
    Uses transposed convolutions to upsample from concatenated
    [noise + text_embedding] to the target sprite resolution.
    
    Architecture:
        [noise + text] -> FC -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x4
    
    Attributes:
        image_size: Output sprite size (32 or 64).
        noise_dim: Dimension of input noise vector.
        text_embedding_dim: Dimension of text embeddings.
        out_channels: Number of output channels (4 for RGBA).
    
    Example:
        >>> gen = SpriteGenerator(image_size=32, noise_dim=100, text_embedding_dim=256)
        >>> noise = torch.randn(8, 100)
        >>> text_emb = torch.randn(8, 256)
        >>> sprites = gen(noise, text_emb)
        >>> sprites.shape
        torch.Size([8, 4, 32, 32])
    """
    
    def __init__(
        self,
        image_size: int = 32,
        noise_dim: int = 100,
        text_embedding_dim: int = 256,
        out_channels: int = 4,
        base_channels: int = 64,
    ) -> None:
        """
        Initialize the Generator.
        
        Args:
            image_size: Output image size (32 or 64).
            noise_dim: Dimension of noise vector.
            text_embedding_dim: Dimension of text embeddings.
            out_channels: Number of output channels (4 for RGBA).
            base_channels: Base number of channels for conv layers.
        """
        super().__init__()
        
        self.image_size = image_size
        self.noise_dim = noise_dim
        self.text_embedding_dim = text_embedding_dim
        self.out_channels = out_channels
        
        # Calculate number of upsampling layers
        self.num_layers = 3 if image_size == 32 else 4
        
        # Input: concatenated [noise + text_embedding]
        self.input_dim = noise_dim + text_embedding_dim
        
        # Calculate initial spatial size and channels
        self.initial_spatial = image_size // (2 ** self.num_layers)  # 4 for 32x32
        self.initial_channels = base_channels * (2 ** self.num_layers)  # 512 for base=64
        
        # Projection from input to initial feature map
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.initial_channels * (self.initial_spatial ** 2)),
            nn.BatchNorm1d(self.initial_channels * (self.initial_spatial ** 2)),
            nn.ReLU(inplace=True),
        )
        
        # Build upsampling layers
        layers: list[nn.Module] = []
        current_channels = self.initial_channels
        
        for i in range(self.num_layers):
            # Last layer outputs to out_channels, others halve the channels
            if i == self.num_layers - 1:
                next_channels = out_channels
                use_tanh = True  # Final layer uses tanh for [-1, 1] output
                use_bn = False
            else:
                next_channels = current_channels // 2
                use_tanh = False
                use_bn = True
            
            layers.append(
                nn.ConvTranspose2d(
                    current_channels,
                    next_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False if use_bn else True,
                )
            )
            
            if use_bn:
                layers.append(nn.BatchNorm2d(next_channels))
            
            if use_tanh:
                layers.append(nn.Tanh())  # Output in [-1, 1]
            else:
                layers.append(nn.ReLU(inplace=True))
            
            current_channels = next_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using normal distribution."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, noise: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate sprite images from noise and text.
        
        Args:
            noise: Random noise of shape (batch, noise_dim).
            text_embedding: Text embedding of shape (batch, text_embedding_dim).
            
        Returns:
            Generated sprites of shape (batch, out_channels, image_size, image_size).
            Values in [-1, 1] range.
        """
        batch_size = noise.size(0)
        
        # Concatenate noise and text embedding
        x = torch.cat([noise, text_embedding], dim=1)  # (batch, noise_dim + text_dim)
        
        # Project to initial feature map
        x = self.fc(x)  # (batch, initial_channels * spatial^2)
        
        # Reshape to spatial dimensions
        x = x.view(batch_size, self.initial_channels, self.initial_spatial, self.initial_spatial)
        
        # Upsample to target resolution
        x = self.conv_layers(x)  # (batch, out_channels, image_size, image_size)
        
        return x
    
    def generate(
        self,
        text_embedding: torch.Tensor,
        num_samples: int = 1,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate sprites from text embeddings.
        
        Args:
            text_embedding: Text embedding of shape (1, embedding_dim) or (batch, embedding_dim).
            num_samples: Number of sprites to generate per text description.
            noise: Optional pre-generated noise. If None, samples random noise.
            
        Returns:
            Generated sprites of shape (batch*num_samples, out_channels, image_size, image_size).
        """
        batch_size = text_embedding.size(0)
        
        # Repeat text embeddings for multiple samples
        if num_samples > 1:
            text_embedding = text_embedding.repeat_interleave(num_samples, dim=0)
        
        total_samples = batch_size * num_samples
        
        # Generate or use provided noise
        if noise is None:
            noise = torch.randn(
                total_samples,
                self.noise_dim,
                device=text_embedding.device,
                dtype=text_embedding.dtype,
            )
        
        # Generate sprites
        with torch.no_grad():
            sprites = self.forward(noise, text_embedding)
        
        return sprites
    
    def interpolate(
        self,
        text_embedding1: torch.Tensor,
        text_embedding2: torch.Tensor,
        steps: int = 10,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate sprites interpolating between two text descriptions.
        
        Args:
            text_embedding1: First text embedding, shape (1, embedding_dim).
            text_embedding2: Second text embedding, shape (1, embedding_dim).
            steps: Number of interpolation steps.
            noise: Fixed noise for interpolation. If None, samples random noise.
            
        Returns:
            Interpolated sprites of shape (steps, out_channels, image_size, image_size).
        """
        if noise is None:
            noise = torch.randn(1, self.noise_dim, device=text_embedding1.device)
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, steps, device=text_embedding1.device)
        
        # Interpolate text embeddings
        text_embeddings = torch.stack([
            (1 - alpha) * text_embedding1[0] + alpha * text_embedding2[0]
            for alpha in alphas
        ])
        
        # Repeat noise for all steps
        noise = noise.repeat(steps, 1)
        
        # Generate interpolated sprites
        with torch.no_grad():
            sprites = self.forward(noise, text_embeddings)
        
        return sprites
