"""
Dataset classes for loading and preprocessing sprite images.

This module provides PyTorch Dataset implementations for loading
sprite sheets and individual sprite images for training.
"""

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class SpriteDataset(Dataset):
    """
    PyTorch Dataset for loading sprite images.
    
    Loads individual sprite images from a directory. Supports PNG images
    with alpha channels (RGBA) for pixel art sprites.
    
    Attributes:
        root_dir: Path to the directory containing sprite images.
        image_size: Target size to resize sprites to.
        transform: Optional additional transforms to apply.
    
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
        """
        Initialize the sprite dataset.
        
        Args:
            root_dir: Path to directory containing sprite images.
            image_size: Target size for resizing (assumes square).
            transform: Optional transform to apply after loading.
        
        Raises:
            ValueError: If root_dir doesn't exist or contains no images.
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        
        if not self.root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")
        
        # Collect all image paths
        self.image_paths: list[Path] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            self.image_paths.extend(self.root_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.root_dir.glob(f"*{ext.upper()}"))
        
        # Also search subdirectories
        for ext in self.SUPPORTED_EXTENSIONS:
            self.image_paths.extend(self.root_dir.glob(f"**/*{ext}"))
            self.image_paths.extend(self.root_dir.glob(f"**/*{ext.upper()}"))
        
        # Remove duplicates and sort
        self.image_paths = sorted(set(self.image_paths))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
    
    def __len__(self) -> int:
        """Return the number of sprites in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess a sprite image.
        
        Args:
            idx: Index of the sprite to load.
        
        Returns:
            Tensor of shape (4, image_size, image_size) with values in [0, 1].
        """
        image_path = self.image_paths[idx]
        
        # Load image with alpha channel
        image = Image.open(image_path).convert("RGBA")
        
        # Resize to target size using nearest neighbor (preserves pixel art)
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST,
        )
        
        # Convert to tensor: (H, W, C) -> (C, H, W), normalized to [0, 1]
        tensor = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0
        
        # Apply optional transform
        if self.transform is not None:
            tensor = self.transform(tensor)
        
        return tensor


class SpriteSheetDataset(Dataset):
    """
    Dataset for loading sprites from sprite sheets.
    
    Extracts individual sprites from a larger sprite sheet image
    based on a grid layout.
    
    Attributes:
        sheet_path: Path to the sprite sheet image.
        sprite_size: Size of each sprite in the sheet.
        target_size: Target size to resize extracted sprites to.
    
    Example:
        >>> dataset = SpriteSheetDataset(
        ...     "data/spritesheet.png",
        ...     sprite_size=16,
        ...     target_size=32
        ... )
        >>> sprite = dataset[0]
        >>> sprite.shape
        torch.Size([4, 32, 32])
    """
    
    def __init__(
        self,
        sheet_path: str | Path,
        sprite_size: int,
        target_size: int = 32,
        padding: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize the sprite sheet dataset.
        
        Args:
            sheet_path: Path to the sprite sheet image.
            sprite_size: Size of each sprite in the sheet (assumes square).
            target_size: Target size to resize sprites to.
            padding: Padding between sprites in the sheet.
            transform: Optional transform to apply after extraction.
        
        Raises:
            ValueError: If sprite sheet doesn't exist.
        """
        self.sheet_path = Path(sheet_path)
        self.sprite_size = sprite_size
        self.target_size = target_size
        self.padding = padding
        self.transform = transform
        
        if not self.sheet_path.exists():
            raise ValueError(f"Sprite sheet does not exist: {sheet_path}")
        
        # Load the sprite sheet
        self.sheet = Image.open(self.sheet_path).convert("RGBA")
        self.sheet_width, self.sheet_height = self.sheet.size
        
        # Calculate grid dimensions
        effective_size = sprite_size + padding
        self.cols = self.sheet_width // effective_size
        self.rows = self.sheet_height // effective_size
        
        # Store sprite coordinates
        self.sprite_coords: list[tuple[int, int, int, int]] = []
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * effective_size
                y = row * effective_size
                self.sprite_coords.append((x, y, x + sprite_size, y + sprite_size))
    
    def __len__(self) -> int:
        """Return the number of sprites in the sheet."""
        return len(self.sprite_coords)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Extract and preprocess a sprite from the sheet.
        
        Args:
            idx: Index of the sprite to extract.
        
        Returns:
            Tensor of shape (4, target_size, target_size) with values in [0, 1].
        """
        coords = self.sprite_coords[idx]
        
        # Extract sprite from sheet
        sprite = self.sheet.crop(coords)
        
        # Resize if needed
        if self.sprite_size != self.target_size:
            sprite = sprite.resize(
                (self.target_size, self.target_size),
                Image.Resampling.NEAREST,
            )
        
        # Convert to tensor
        tensor = torch.from_numpy(
            __import__("numpy").array(sprite)
        ).permute(2, 0, 1).float() / 255.0
        
        # Apply optional transform
        if self.transform is not None:
            tensor = self.transform(tensor)
        
        return tensor
    
    def filter_empty(self, alpha_threshold: float = 0.1) -> "SpriteSheetDataset":
        """
        Filter out empty or mostly transparent sprites.
        
        Creates a new dataset excluding sprites with average alpha
        below the threshold.
        
        Args:
            alpha_threshold: Minimum average alpha to keep sprite.
        
        Returns:
            New SpriteSheetDataset with filtered coordinates.
        """
        filtered_coords: list[tuple[int, int, int, int]] = []
        
        for coords in self.sprite_coords:
            sprite = self.sheet.crop(coords)
            alpha = sprite.split()[-1]  # Get alpha channel
            avg_alpha = sum(alpha.getdata()) / (self.sprite_size ** 2) / 255.0
            
            if avg_alpha >= alpha_threshold:
                filtered_coords.append(coords)
        
        # Create new instance with filtered coords
        new_dataset = SpriteSheetDataset.__new__(SpriteSheetDataset)
        new_dataset.sheet_path = self.sheet_path
        new_dataset.sprite_size = self.sprite_size
        new_dataset.target_size = self.target_size
        new_dataset.padding = self.padding
        new_dataset.transform = self.transform
        new_dataset.sheet = self.sheet
        new_dataset.sheet_width = self.sheet_width
        new_dataset.sheet_height = self.sheet_height
        new_dataset.cols = self.cols
        new_dataset.rows = self.rows
        new_dataset.sprite_coords = filtered_coords
        
        return new_dataset
