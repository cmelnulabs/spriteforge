"""
Utility functions for SpriteForge.

This module contains helper functions for visualization, file I/O,
and other common operations.
"""

from pathlib import Path
from typing import Sequence

import torch


def save_sprite_grid(
    sprites: torch.Tensor,
    path: str | Path,
    ncols: int = 8,
    padding: int = 2,
    scale: int = 1,
) -> None:
    """
    Save a grid of sprites as a single image.
    
    Args:
        sprites: Tensor of sprites, shape (N, C, H, W).
        path: Output file path.
        ncols: Number of columns in the grid.
        padding: Padding between sprites in pixels.
        scale: Upscale factor for the output.
    """
    from PIL import Image
    import numpy as np
    
    n, c, h, w = sprites.shape
    nrows = (n + ncols - 1) // ncols
    
    # Calculate grid dimensions
    grid_h = nrows * h + (nrows - 1) * padding
    grid_w = ncols * w + (ncols - 1) * padding
    
    # Create grid array with transparent background
    if c == 4:
        grid = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
    else:
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Place sprites in grid
    for i, sprite in enumerate(sprites):
        row = i // ncols
        col = i % ncols
        
        y = row * (h + padding)
        x = col * (w + padding)
        
        # Convert tensor to numpy
        img = (sprite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        grid[y:y + h, x:x + w] = img
    
    # Create image
    mode = "RGBA" if c == 4 else "RGB"
    img = Image.fromarray(grid, mode=mode)
    
    # Upscale if requested
    if scale > 1:
        new_size = (img.width * scale, img.height * scale)
        img = img.resize(new_size, Image.Resampling.NEAREST)
    
    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def tensor_to_pil(tensor: torch.Tensor) -> "Image.Image":
    """
    Convert a sprite tensor to a PIL Image.
    
    Args:
        tensor: Sprite tensor of shape (C, H, W) with values in [0, 1].
    
    Returns:
        PIL Image in RGBA mode.
    """
    from PIL import Image
    import numpy as np
    
    # Ensure tensor is on CPU and detached
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.cpu()
    
    # Convert to numpy
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Create PIL image
    mode = "RGBA" if tensor.shape[0] == 4 else "RGB"
    return Image.fromarray(img_array, mode=mode)


def pil_to_tensor(image: "Image.Image") -> torch.Tensor:
    """
    Convert a PIL Image to a sprite tensor.
    
    Args:
        image: PIL Image (will be converted to RGBA).
    
    Returns:
        Tensor of shape (4, H, W) with values in [0, 1].
    """
    import numpy as np
    
    # Convert to RGBA
    image = image.convert("RGBA")
    
    # Convert to tensor
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    return tensor


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device for the best available hardware.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """
    Count model parameters by component.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Dictionary with parameter counts by module name.
    """
    counts = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        counts[name] = params
    
    counts["total"] = sum(counts.values())
    counts["trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    return counts
