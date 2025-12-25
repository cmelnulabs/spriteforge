"""
Data augmentation transforms for sprite images.

This module provides pixel-art-appropriate augmentation transforms
that preserve the discrete nature of pixel art while increasing
dataset diversity.
"""

import random
from typing import Callable

import torch


class Compose:
    """
    Compose multiple transforms together.
    
    Example:
        >>> transform = Compose([
        ...     RandomHorizontalFlip(),
        ...     RandomRotation90(),
        ... ])
        >>> augmented = transform(sprite)
    """
    
    def __init__(self, transforms: list[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        """
        Initialize the composition.
        
        Args:
            transforms: List of transforms to apply in sequence.
        """
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence."""
        for t in self.transforms:
            x = t(x)
        return x


class RandomHorizontalFlip:
    """
    Randomly flip the sprite horizontally.
    
    This is safe for most sprites as horizontal symmetry is common.
    Probability defaults to 0.5.
    """
    
    def __init__(self, p: float = 0.5) -> None:
        """
        Initialize the transform.
        
        Args:
            p: Probability of applying the flip.
        """
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply horizontal flip with probability p."""
        if random.random() < self.p:
            return torch.flip(x, dims=[-1])
        return x


class RandomVerticalFlip:
    """
    Randomly flip the sprite vertically.
    
    Use with caution as vertical flips may not make sense for
    all sprite types (e.g., characters with gravity).
    """
    
    def __init__(self, p: float = 0.5) -> None:
        """
        Initialize the transform.
        
        Args:
            p: Probability of applying the flip.
        """
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vertical flip with probability p."""
        if random.random() < self.p:
            return torch.flip(x, dims=[-2])
        return x


class RandomRotation90:
    """
    Randomly rotate the sprite by 90, 180, or 270 degrees.
    
    Only uses 90-degree rotations to preserve pixel alignment.
    Useful for objects that can appear in any orientation.
    """
    
    def __init__(self, p: float = 0.5) -> None:
        """
        Initialize the transform.
        
        Args:
            p: Probability of applying a rotation.
        """
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random 90-degree rotation with probability p."""
        if random.random() < self.p:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            return torch.rot90(x, k, dims=[-2, -1])
        return x


class ColorJitter:
    """
    Apply random color adjustments while preserving transparency.
    
    Adjusts brightness, contrast, and saturation of the RGB channels
    without affecting the alpha channel.
    """
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
    ) -> None:
        """
        Initialize the transform.
        
        Args:
            brightness: Maximum brightness adjustment factor.
            contrast: Maximum contrast adjustment factor.
            saturation: Maximum saturation adjustment factor.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random color adjustments."""
        # Separate RGB and alpha
        rgb = x[:3]
        alpha = x[3:4] if x.shape[0] == 4 else None
        
        # Random brightness
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            rgb = rgb * factor
        
        # Random contrast
        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = rgb.mean()
            rgb = (rgb - mean) * factor + mean
        
        # Random saturation (simple approach)
        if self.saturation > 0:
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            gray = rgb.mean(dim=0, keepdim=True)
            rgb = rgb * factor + gray * (1 - factor)
        
        # Clamp to valid range
        rgb = torch.clamp(rgb, 0, 1)
        
        # Recombine with alpha
        if alpha is not None:
            return torch.cat([rgb, alpha], dim=0)
        return rgb


class RandomPaletteSwap:
    """
    Randomly swap color palette by shuffling RGB channels.
    
    Creates color variations while maintaining the sprite's structure.
    Useful for generating diverse training data from limited sources.
    """
    
    def __init__(self, p: float = 0.3) -> None:
        """
        Initialize the transform.
        
        Args:
            p: Probability of applying the palette swap.
        """
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random channel shuffle with probability p."""
        if random.random() < self.p:
            # Separate RGB and alpha
            rgb = x[:3]
            alpha = x[3:4] if x.shape[0] == 4 else None
            
            # Shuffle RGB channels
            perm = torch.randperm(3)
            rgb = rgb[perm]
            
            # Recombine
            if alpha is not None:
                return torch.cat([rgb, alpha], dim=0)
            return rgb
        return x


class AddNoise:
    """
    Add small random noise to sprite colors.
    
    Helps the model generalize by adding small variations
    to pixel colors. Uses very low noise to not destroy pixel art.
    """
    
    def __init__(self, std: float = 0.02, p: float = 0.5) -> None:
        """
        Initialize the transform.
        
        Args:
            std: Standard deviation of the Gaussian noise.
            p: Probability of applying noise.
        """
        self.std = std
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with probability p."""
        if random.random() < self.p:
            # Only add noise to RGB, preserve alpha
            rgb = x[:3]
            alpha = x[3:4] if x.shape[0] == 4 else None
            
            noise = torch.randn_like(rgb) * self.std
            rgb = torch.clamp(rgb + noise, 0, 1)
            
            if alpha is not None:
                return torch.cat([rgb, alpha], dim=0)
            return rgb
        return x


def get_default_transforms(augment: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get the default transform pipeline for sprite training.
    
    Args:
        augment: Whether to include augmentation transforms.
    
    Returns:
        Composed transform function.
    """
    if augment:
        return Compose([
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            AddNoise(std=0.01, p=0.3),
        ])
    
    # No transforms for evaluation
    return lambda x: x
