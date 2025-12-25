#!/usr/bin/env python3
"""
Preprocess raw sprite images for training.

Handles:
- Resizing to target dimensions
- Filtering empty/transparent sprites
- Deduplication via perceptual hashing
- Extracting sprites from spritesheets
"""

import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

from PIL import Image
import numpy as np


def compute_phash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute perceptual hash for deduplication."""
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "".join(str(int(b)) for b in diff.flatten())


def get_alpha_coverage(img: Image.Image) -> float:
    """Calculate percentage of non-transparent pixels."""
    if img.mode != "RGBA":
        return 1.0
    alpha = np.array(img)[:, :, 3]
    return np.count_nonzero(alpha > 10) / alpha.size


def extract_sprites_from_sheet(
    sheet_path: Path,
    sprite_size: int,
    output_dir: Path,
    padding: int = 0,
) -> int:
    """Extract individual sprites from a spritesheet."""
    sheet = Image.open(sheet_path).convert("RGBA")
    w, h = sheet.size
    
    effective = sprite_size + padding
    cols = w // effective
    rows = h // effective
    
    count = 0
    for row in range(rows):
        for col in range(cols):
            x = col * effective
            y = row * effective
            sprite = sheet.crop((x, y, x + sprite_size, y + sprite_size))
            
            if get_alpha_coverage(sprite) > 0.05:
                name = f"{sheet_path.stem}_{row}_{col}.png"
                sprite.save(output_dir / name)
                count += 1
    
    return count


def process_single_image(
    img_path: Path,
    output_dir: Path,
    target_size: int,
    min_alpha: float,
) -> tuple[bool, str | None]:
    """Process a single image file."""
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception:
        return False, None
    
    # Check alpha coverage
    if get_alpha_coverage(img) < min_alpha:
        return False, None
    
    # Resize using nearest neighbor (preserves pixel art)
    img = img.resize((target_size, target_size), Image.Resampling.NEAREST)
    
    # Compute hash for deduplication
    phash = compute_phash(img)
    
    # Save
    output_path = output_dir / f"{img_path.stem}.png"
    img.save(output_path)
    
    return True, phash


def preprocess(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 32,
    min_alpha: float = 0.1,
    deduplicate: bool = True,
    extract_sheets: bool = False,
    sheet_sprite_size: int = 16,
    max_samples: int | None = None,
) -> dict:
    """
    Preprocess a directory of sprite images.
    
    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    image_files = [
        f for f in input_dir.rglob("*") 
        if f.suffix.lower() in extensions
    ]
    
    print(f"Found {len(image_files)} image files")
    
    stats = {
        "total_found": len(image_files),
        "processed": 0,
        "filtered_alpha": 0,
        "filtered_duplicate": 0,
        "extracted_from_sheets": 0,
        "errors": 0,
    }
    
    seen_hashes: set[str] = set()
    
    for i, img_path in enumerate(image_files):
        if max_samples and stats["processed"] >= max_samples:
            break
        
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(image_files)}...")
        
        # Check if it might be a spritesheet
        if extract_sheets:
            try:
                img = Image.open(img_path)
                w, h = img.size
                if w > target_size * 2 and h > target_size * 2:
                    count = extract_sprites_from_sheet(
                        img_path, sheet_sprite_size, output_dir
                    )
                    stats["extracted_from_sheets"] += count
                    continue
            except Exception:
                pass
        
        success, phash = process_single_image(
            img_path, output_dir, target_size, min_alpha
        )
        
        if not success:
            if phash is None:
                stats["filtered_alpha"] += 1
            else:
                stats["errors"] += 1
            continue
        
        if deduplicate and phash:
            if phash in seen_hashes:
                # Remove duplicate
                (output_dir / f"{img_path.stem}.png").unlink()
                stats["filtered_duplicate"] += 1
                continue
            seen_hashes.add(phash)
        
        stats["processed"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess sprites for training")
    parser.add_argument("input", help="Input directory with raw sprites")
    parser.add_argument("output", help="Output directory for processed sprites")
    parser.add_argument("--size", type=int, default=32, help="Target size (default: 32)")
    parser.add_argument("--min-alpha", type=float, default=0.1, help="Min alpha coverage")
    parser.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")
    parser.add_argument("--extract-sheets", action="store_true", help="Extract from spritesheets")
    parser.add_argument("--sheet-size", type=int, default=16, help="Sprite size in sheets")
    parser.add_argument("--max", type=int, help="Max samples to process")
    
    args = parser.parse_args()
    
    stats = preprocess(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        target_size=args.size,
        min_alpha=args.min_alpha,
        deduplicate=not args.no_dedupe,
        extract_sheets=args.extract_sheets,
        sheet_sprite_size=args.sheet_size,
        max_samples=args.max,
    )
    
    print("\n--- Preprocessing Complete ---")
    print(f"Total found:      {stats['total_found']}")
    print(f"Processed:        {stats['processed']}")
    print(f"From sheets:      {stats['extracted_from_sheets']}")
    print(f"Filtered (alpha): {stats['filtered_alpha']}")
    print(f"Filtered (dupe):  {stats['filtered_duplicate']}")
    print(f"Errors:           {stats['errors']}")


if __name__ == "__main__":
    main()
