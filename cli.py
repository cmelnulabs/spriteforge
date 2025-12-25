"""
Command-line interface for SpriteForge.

Provides commands for training, generating, and managing sprite VAE models.
"""

import argparse
import sys
from pathlib import Path

import torch


def train(args: argparse.Namespace) -> int:
    """
    Train a SpriteVAE model.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Exit code (0 for success).
    """
    from torch.utils.data import DataLoader, random_split

    from spriteforge.data import SpriteDataset, SpriteSheetDataset, get_default_transforms
    from spriteforge.models import SpriteVAE
    from spriteforge.training import TrainConfig, Trainer

    print(f"Loading dataset from: {args.data}")
    
    # Load dataset based on type
    if args.spritesheet:
        dataset = SpriteSheetDataset(
            args.data,
            sprite_size=args.sprite_size,
            target_size=args.image_size,
            transform=get_default_transforms(augment=True),
        )
        if args.filter_empty:
            dataset = dataset.filter_empty(alpha_threshold=0.1)
    else:
        dataset = SpriteDataset(
            args.data,
            image_size=args.image_size,
            transform=get_default_transforms(augment=True),
        )
    
    print(f"Dataset size: {len(dataset)} sprites")
    
    # Split into train/val
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    # Create model
    model = SpriteVAE(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        in_channels=4,
    )
    
    # Create trainer
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        beta=args.beta,
        beta_warmup_epochs=args.beta_warmup,
        checkpoint_dir=args.output,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
    )
    
    trainer = Trainer(model, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Setup TensorBoard if requested
    if args.tensorboard:
        trainer.setup_tensorboard(args.output + "/runs")
    
    # Train
    trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
    
    return 0


def generate(args: argparse.Namespace) -> int:
    """
    Generate sprites using a trained model.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Exit code (0 for success).
    """
    from PIL import Image
    import numpy as np

    from spriteforge.models import SpriteVAE

    print(f"Loading model from: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    
    # Create model
    model = SpriteVAE(
        image_size=config["image_size"],
        latent_dim=config["latent_dim"],
        in_channels=config.get("in_channels", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Generate sprites
    print(f"Generating {args.num} sprites...")
    with torch.no_grad():
        sprites = model.generate(args.num)
    
    # Convert to images and save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sprite in enumerate(sprites):
        # Convert tensor to numpy array
        img_array = (sprite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(img_array, mode="RGBA")
        
        # Optionally upscale
        if args.scale > 1:
            new_size = (img.width * args.scale, img.height * args.scale)
            img = img.resize(new_size, Image.Resampling.NEAREST)
        
        # Save
        img.save(output_dir / f"sprite_{i:04d}.png")
    
    print(f"Saved {args.num} sprites to {output_dir}")
    
    return 0


def download(args: argparse.Namespace) -> int:
    """
    Download a sprite dataset from Kaggle.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Exit code (0 for success).
    """
    import subprocess
    import zipfile
    from pathlib import Path
    
    datasets = {
        "pixel-art": "ebrahimelgazar/pixel-art",          # 89k sprites, 16x16
        "pixel-characters": "volodymyrpivoshenko/pixel-characters-dataset",  # 3.6k, 64x64
        "pokemon": "hlrhegemony/pokemon-image-dataset",   # Pokemon sprites
    }
    
    if args.dataset not in datasets:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {', '.join(datasets.keys())}")
        return 1
    
    kaggle_id = datasets[args.dataset]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {args.dataset} from Kaggle...")
    print("(Requires kaggle CLI: pip install kaggle)")
    print("(And API key at ~/.kaggle/kaggle.json)")
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(output_dir)],
            check=True,
        )
    except FileNotFoundError:
        print("\nError: kaggle CLI not found. Install with: pip install kaggle")
        return 1
    except subprocess.CalledProcessError:
        print("\nError: Download failed. Check your Kaggle API key.")
        return 1
    
    # Unzip
    zip_name = kaggle_id.split("/")[-1] + ".zip"
    zip_path = output_dir / zip_name
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        zip_path.unlink()
    
    print(f"\nDataset downloaded to {output_dir}")
    print(f"Next: spriteforge preprocess {output_dir} data/processed/ --size 32")
    return 0


def preprocess(args: argparse.Namespace) -> int:
    """
    Preprocess raw sprite images for training.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Exit code (0 for success).
    """
    from pathlib import Path
    import hashlib
    
    from PIL import Image
    import numpy as np
    
    def compute_phash(img: Image.Image, hash_size: int = 8) -> str:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return "".join(str(int(b)) for b in diff.flatten())
    
    def get_alpha_coverage(img: Image.Image) -> float:
        if img.mode != "RGBA":
            return 1.0
        alpha = np.array(img)[:, :, 3]
        return np.count_nonzero(alpha > 10) / alpha.size
    
    def extract_from_sheet(sheet_path, sprite_size, output_dir):
        sheet = Image.open(sheet_path).convert("RGBA")
        w, h = sheet.size
        count = 0
        for row in range(h // sprite_size):
            for col in range(w // sprite_size):
                x, y = col * sprite_size, row * sprite_size
                sprite = sheet.crop((x, y, x + sprite_size, y + sprite_size))
                if get_alpha_coverage(sprite) > 0.05:
                    sprite.save(output_dir / f"{sheet_path.stem}_{row}_{col}.png")
                    count += 1
        return count
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in extensions]
    
    print(f"Found {len(files)} images")
    
    seen, processed, filtered = set(), 0, 0
    
    for i, path in enumerate(files):
        if args.max and processed >= args.max:
            break
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(files)}...")
        
        try:
            img = Image.open(path).convert("RGBA")
        except Exception:
            continue
        
        w, h = img.size
        if args.extract_sheets and w > args.size * 2 and h > args.size * 2:
            processed += extract_from_sheet(path, args.sheet_size, output_dir)
            continue
        
        if get_alpha_coverage(img) < args.min_alpha:
            filtered += 1
            continue
        
        img = img.resize((args.size, args.size), Image.Resampling.NEAREST)
        phash = compute_phash(img)
        
        if not args.no_dedupe and phash in seen:
            filtered += 1
            continue
        seen.add(phash)
        
        img.save(output_dir / f"{path.stem}.png")
        processed += 1
    
    print(f"\nDone: {processed} sprites saved, {filtered} filtered")
    return 0


def interpolate(args: argparse.Namespace) -> int:
    """
    Interpolate between two sprites.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Exit code (0 for success).
    """
    from PIL import Image
    import numpy as np

    from spriteforge.models import SpriteVAE

    print(f"Loading model from: {args.model}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    
    # Create model
    model = SpriteVAE(
        image_size=config["image_size"],
        latent_dim=config["latent_dim"],
        in_channels=config.get("in_channels", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load input sprites
    def load_sprite(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGBA")
        img = img.resize((config["image_size"], config["image_size"]), Image.Resampling.NEAREST)
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)
    
    sprite1 = load_sprite(args.sprite1)
    sprite2 = load_sprite(args.sprite2)
    
    # Interpolate
    print(f"Interpolating with {args.steps} steps...")
    interpolated = model.interpolate(sprite1, sprite2, num_steps=args.steps)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sprite in enumerate(interpolated):
        img_array = (sprite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="RGBA")
        
        if args.scale > 1:
            new_size = (img.width * args.scale, img.height * args.scale)
            img = img.resize(new_size, Image.Resampling.NEAREST)
        
        img.save(output_dir / f"interp_{i:02d}.png")
    
    print(f"Saved {args.steps} interpolation steps to {output_dir}")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="spriteforge",
        description="SpriteForge: VAE for 2D pixel art sprite generation",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a SpriteVAE model")
    train_parser.add_argument("data", help="Path to sprite data (directory or spritesheet)")
    train_parser.add_argument("-o", "--output", default="output", help="Output directory")
    train_parser.add_argument("--image-size", type=int, default=32, help="Image size (default: 32)")
    train_parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--beta", type=float, default=1.0, help="KL divergence weight")
    train_parser.add_argument("--beta-warmup", type=int, default=10, help="Beta warmup epochs")
    train_parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    train_parser.add_argument("--spritesheet", action="store_true", help="Input is a spritesheet")
    train_parser.add_argument("--sprite-size", type=int, default=16, help="Sprite size in sheet")
    train_parser.add_argument("--filter-empty", action="store_true", help="Filter empty sprites")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    train_parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard")
    train_parser.add_argument("--resume", help="Resume from checkpoint path")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate sprites from trained model")
    gen_parser.add_argument("model", help="Path to trained model checkpoint")
    gen_parser.add_argument("-o", "--output", default="generated", help="Output directory")
    gen_parser.add_argument("-n", "--num", type=int, default=16, help="Number to generate")
    gen_parser.add_argument("--scale", type=int, default=1, help="Upscale factor")
    
    # Interpolate command
    interp_parser = subparsers.add_parser("interpolate", help="Interpolate between sprites")
    interp_parser.add_argument("model", help="Path to trained model checkpoint")
    interp_parser.add_argument("sprite1", help="First sprite image")
    interp_parser.add_argument("sprite2", help="Second sprite image")
    interp_parser.add_argument("-o", "--output", default="interpolated", help="Output directory")
    interp_parser.add_argument("--steps", type=int, default=10, help="Interpolation steps")
    interp_parser.add_argument("--scale", type=int, default=1, help="Upscale factor")
    
    # Preprocess command
    prep_parser = subparsers.add_parser("preprocess", help="Preprocess raw sprites")
    prep_parser.add_argument("input", help="Input directory with raw sprites")
    prep_parser.add_argument("output", help="Output directory for processed sprites")
    prep_parser.add_argument("--size", type=int, default=32, help="Target size")
    prep_parser.add_argument("--min-alpha", type=float, default=0.1, help="Min alpha coverage")
    prep_parser.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")
    prep_parser.add_argument("--extract-sheets", action="store_true", help="Extract from spritesheets")
    prep_parser.add_argument("--sheet-size", type=int, default=16, help="Sprite size in sheets")
    prep_parser.add_argument("--max", type=int, help="Max samples to process")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download dataset from Kaggle")
    dl_parser.add_argument("dataset", choices=["pixel-art", "pixel-characters", "pokemon"],
                          help="Dataset to download")
    dl_parser.add_argument("-o", "--output", default="data/raw", help="Output directory")
    
    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    commands = {
        "train": train,
        "generate": generate,
        "interpolate": interpolate,
        "preprocess": preprocess,
        "download": download,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
