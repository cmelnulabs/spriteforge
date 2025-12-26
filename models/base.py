"""
Base model classes and interfaces for SpriteForge.

This module defines abstract base classes that all GAN models should inherit from,
ensuring a consistent interface across different architectures.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all SpriteForge models.
    
    All models should inherit from this class and implement
    the required abstract methods for a consistent interface.
    
    Attributes:
        device: The device (CPU/GPU) where the model resides.
    """
    
    def __init__(self) -> None:
        """Initialize the base model."""
        super().__init__()
        self._device: torch.device = torch.device("cpu")
    
    @property
    def device(self) -> torch.device:
        """Get the device where the model parameters reside."""
        return self._device
    
    def to(self, device: torch.device | str) -> "BaseModel":
        """
        Move the model to the specified device.
        
        Args:
            device: Target device (e.g., 'cuda', 'cpu', torch.device).
            
        Returns:
            Self reference for method chaining.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return super().to(device)
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass through the model.
        
        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
            
        Returns:
            Model output (implementation-specific).
        """
        pass
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str) -> None:
        """
        Save model weights to disk.
        
        Args:
            path: File path for the saved weights.
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, strict: bool = True) -> None:
        """
        Load model weights from disk.
        
        Args:
            path: File path to the saved weights.
            strict: Whether to strictly enforce state dict keys match.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict, strict=strict)


class BaseTextEncoder(BaseModel):
    """
    Abstract base class for text encoder networks.
    
    Text encoders convert text descriptions into embedding vectors
    that can be used to condition image generation.
    """
    
    @abstractmethod
    def forward(self, text: list[str]) -> torch.Tensor:
        """
        Encode text descriptions to embedding vectors.
        
        Args:
            text: List of text descriptions (batch).
            
        Returns:
            Text embeddings of shape (batch, embedding_dim).
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the text embeddings.
        
        Returns:
            Embedding dimension.
        """
        pass


class BaseGenerator(BaseModel):
    """
    Abstract base class for Generator networks in GANs.
    
    Generators create images from noise and conditioning information (text).
    """
    
    @abstractmethod
    def forward(self, noise: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate images from noise and text embeddings.
        
        Args:
            noise: Random noise vector of shape (batch, noise_dim).
            text_embedding: Text embedding of shape (batch, embedding_dim).
            
        Returns:
            Generated image tensor of shape (batch, channels, height, width).
        """
        pass
    
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
            num_samples: Number of sprites to generate per text.
            noise: Optional pre-generated noise. If None, will sample random noise.
            
        Returns:
            Generated sprites of shape (batch*num_samples, channels, height, width).
        """
        pass


class BaseDiscriminator(BaseModel):
    """
    Abstract base class for Discriminator networks in GANs.
    
    Discriminators evaluate whether images are real or fake,
    and optionally verify text-image correspondence.
    """
    
    @abstractmethod
    def forward(self, image: torch.Tensor, text_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminate real/fake images and check text-image matching.
        
        Args:
            image: Image tensor of shape (batch, channels, height, width).
            text_embedding: Text embedding of shape (batch, embedding_dim).
            
        Returns:
            Tuple of (realness_score, matching_score):
                - realness_score: Probability image is real, shape (batch, 1).
                - matching_score: Probability text matches image, shape (batch, 1).
        """
        pass
