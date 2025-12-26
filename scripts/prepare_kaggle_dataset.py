"""
Convert Kaggle pixel-art dataset to text-sprite pairs for training.

Downloads sprites from numpy arrays, converts categories to text descriptions,
and creates the captions.json file needed for training.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# Category to text description mapping
CATEGORY_DESCRIPTIONS = {
    0: [
        "character sprite",
        "pixel character",
        "game character",
        "character icon",
    ],
    1: [
        "item sprite",
        "game item",
        "collectible item",
        "pixel item",
    ],
    2: [
        "effect sprite",
        "magic effect",
        "particle effect",
        "visual effect",
    ],
    3: [
        "terrain sprite",
        "ground tile",
        "environment tile",
        "terrain element",
    ],
    4: [
        "enemy sprite",
        "monster sprite",
        "hostile creature",
        "enemy character",
    ],
}


def prepare_dataset(
    data_path: str,
    output_dir: str,
    max_sprites: int | None = None,
    target_size: int = 32,
    add_variation: bool = True,
):
    """
    Prepare the Kaggle pixel-art dataset for text-conditional training.
    
    Args:
        data_path: Path to the downloaded Kaggle dataset.
        output_dir: Output directory for processed sprites.
        max_sprites: Maximum number of sprites to process (None = all).
        target_size: Target size to resize sprites to (16 -> 32 or 64).
        add_variation: Add variations to category descriptions.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    sprites = np.load(data_path / "sprites.npy")
    labels = np.load(data_path / "sprites_labels.npy")
    
    print(f"Loaded {len(sprites):,} sprites")
    
    # Limit dataset size if requested
    if max_sprites and max_sprites < len(sprites):
        indices = np.random.choice(len(sprites), max_sprites, replace=False)
        sprites = sprites[indices]
        labels = labels[indices]
        print(f"Using {max_sprites:,} random sprites")
    
    # Convert sprites and create captions
    captions = {}
    
    print(f"Processing sprites...")
    for idx in tqdm(range(len(sprites))):
        sprite = sprites[idx]
        label_onehot = labels[idx]
        
        # Get category index
        category = int(np.argmax(label_onehot))
        
        # Generate text description
        if add_variation:
            # Randomly select from category descriptions
            description = random.choice(CATEGORY_DESCRIPTIONS[category])
        else:
            # Use first (canonical) description
            description = CATEGORY_DESCRIPTIONS[category][0]
        
        # Convert to PIL Image
        img = Image.fromarray(sprite)
        
        # Resize if needed (16x16 -> 32x32 using nearest neighbor)
        if target_size != sprite.shape[0]:
            img = img.resize((target_size, target_size), Image.Resampling.NEAREST)
        
        # Convert RGB to RGBA (add alpha channel)
        if img.mode == "RGB":
            # Add full opacity alpha channel
            img = img.convert("RGBA")
        
        # Save sprite
        filename = f"sprite_{idx:06d}.png"
        img.save(output_dir / filename)
        
        # Add to captions
        captions[filename] = description
    
    # Save captions file
    captions_file = output_dir / "captions.json"
    with open(captions_file, "w") as f:
        json.dump(captions, f, indent=2)
    
    print(f"\n✓ Processed {len(sprites):,} sprites")
    print(f"✓ Saved to: {output_dir}")
    print(f"✓ Captions file: {captions_file}")
    
    # Print statistics
    print("\nDataset statistics:")
    category_counts = {}
    for caption in captions.values():
        base_category = caption.split()[0] if caption else "unknown"
        category_counts[base_category] = category_counts.get(base_category, 0) + 1
    
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count:,}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Kaggle pixel-art dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/melendez/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/1",
        help="Path to Kaggle dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sprites_processed",
        help="Output directory",
    )
    parser.add_argument(
        "--max-sprites",
        type=int,
        default=None,
        help="Maximum number of sprites (default: all)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=32,
        help="Target sprite size (16, 32, or 64)",
    )
    parser.add_argument(
        "--no-variation",
        action="store_true",
        help="Use only canonical category names",
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.data_path,
        args.output,
        args.max_sprites,
        args.size,
        add_variation=not args.no_variation,
    )
