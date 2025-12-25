"""
Base model classes and interfaces for SpriteForge.

This module defines abstract base classes that all models should inherit from,
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
    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor.
            
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


class BaseEncoder(BaseModel):
    """
    Abstract base class for encoder networks.
    
    Encoders compress input images into a latent representation.
    For VAEs, this outputs distribution parameters (mean and log-variance).
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            
        Returns:
            Tuple of (mean, log_variance) tensors for the latent distribution.
        """
        pass


class BaseDecoder(BaseModel):
    """
    Abstract base class for decoder networks.
    
    Decoders reconstruct images from latent representations.
    """
    
    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent tensor of shape (batch, latent_dim).
            
        Returns:
            Reconstructed image tensor of shape (batch, channels, height, width).
        """
        pass


class BaseVAE(BaseModel):
    """
    Abstract base class for Variational Autoencoders.
    
    VAEs combine an encoder and decoder with a probabilistic latent space,
    enabling both reconstruction and generation of new samples.
    """
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple of (mean, log_variance) for the latent distribution.
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output.
        
        Args:
            z: Latent tensor.
            
        Returns:
            Reconstructed output tensor.
        """
        pass
    
    @abstractmethod
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Apply the reparameterization trick.
        
        This enables backpropagation through the stochastic sampling step
        by expressing the random variable as a deterministic function.
        
        Args:
            mu: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.
            
        Returns:
            Sampled latent vector.
        """
        pass
    
    @abstractmethod
    def generate(self, num_samples: int) -> torch.Tensor:
        """
        Generate new samples from the prior distribution.
        
        Args:
            num_samples: Number of samples to generate.
            
        Returns:
            Generated output tensor.
        """
        pass
