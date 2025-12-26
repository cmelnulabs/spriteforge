"""
Dataset classes for loading sprite images with text captions.

This module provides PyTorch Dataset implementations for loading
sprite-text pairs for text-conditional GAN training.
"""

import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class TextSpriteDataset(Dataset):
    """
    PyTorch Dataset for loading sprite images with text descriptions.
    
    Loads sprite-text pairs from a directory with a captions file.
    The captions file should be a JSON with format:
    {
        "sprite_001.png": "red warrior with sword",
        "sprite_002.png": "blue potion",
        ...
    }
    
    Attributes:
        root_dir: Path to directory containing sprite images.
        captions_file: Path to JSON file with captions.
        image_size: Target size to resize sprites to.
        transform: Optional additional transforms.
    
    Example:
        >>> dataset = TextSpriteDataset("data/sprites", "captions.json", image_size=32)
        >>> sprite, text = dataset[0]
        >>> sprite.shape, text
        (torch.Size([4, 32, 32]), "red warrior")
    """
    
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    
    def __init__(
        self,
        root_dir: str | Path,
        captions_file: str | Path,
        image_size: int = 32,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize the text-sprite dataset.
        
        Args:
            root_dir: Path to directory containing sprite images.
            captions_file: Path to JSON file with {filename: caption} pairs.
            image_size: Target size for resizing (assumes square).
            transform: Optional transform to apply after loading.
        
        Raises:
            ValueError: If files don't exist or captions are invalid.
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        
        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")
        
        # Load captions
        captions_path = Path(captions_file)
        if not captions_path.exists():
            raise ValueError(f"Captions file does not exist: {captions_file}")
        
        with open(captions_path, "r") as f:
            self.captions = json.load(f)
        
        # Build list of valid (image_path, caption) pairs
        self.samples: list[tuple[Path, str]] = []
        for filename, caption in self.captions.items():
            img_path = self.root_dir / filename
            if img_path.exists():
                self.samples.append((img_path, caption))
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid sprite-caption pairs found")
        
        print(f"Loaded {len(self.samples)} sprite-caption pairs")
    
    def __len__(self) -> int:
        """Return the number of sprite-caption pairs."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Load a sprite and its text description.
        
        Args:
            idx: Index of the sample.
        
        Returns:
            Tuple of (sprite_tensor, text_description):
                - sprite: Tensor of shape (4, image_size, image_size), values in [-1, 1]
                - text: String description
        """
        image_path, text = self.samples[idx]
        
        # Load image with alpha channel
        image = Image.open(image_path).convert("RGBA")
        
        # Resize using nearest neighbor (preserves pixel art)
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )
        
        # Convert to tensor: (H, W, C) -> (C, H, W), normalized to [-1, 1]
        tensor = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0
        
        # Scale to [-1, 1] for GAN training
        tensor = tensor * 2.0 - 1.0
        
        # Apply optional transform
        if self.transform is not None:
            tensor = self.transform(tensor)
        
        return tensor, text


class SpriteDataset(Dataset):
    """
    Simple dataset for loading sprites without captions (for testing).
    
    Example:
        >>> dataset = SpriteDataset("data/sprites", image_size=32)
        >>> sprite = dataset[0]
        >>> sprite.shape
        torch.Size([4, 32, 32])
    """
    
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    
    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 32,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the sprite dataset."""
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        
        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")
        
        # Collect all image paths
        self.image_paths: list[Path] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            self.image_paths.extend(self.root_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.root_dir.glob(f"**/*{ext}"))
        
        self.image_paths = sorted(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess a sprite."""
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert("RGBA")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )
        
        tensor = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0
        
        # Scale to [-1, 1]
        tensor = tensor * 2.0 - 1.0
        
        if self.transform is not None:
            tensor = self.transform(tensor)
        
        return tensor
