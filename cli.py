"""
Command-line interface for SpriteForge text-to-sprite GAN.

Provides commands for training and generating sprites from text descriptions.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def train(args: argparse.Namespace) -> int:
    """
    Train a text-to-sprite GAN.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Exit code (0 for success).
    """
    from torch.utils.data import DataLoader
    
    from spriteforge.data.dataset import TextSpriteDataset
    from spriteforge.models.discriminator import SpriteDiscriminator
    from spriteforge.models.generator import SpriteGenerator
    from spriteforge.models.text_encoder import SimpleTextEncoder
    from spriteforge.training.trainer import GANTrainConfig, GANTrainer
    
    print(f"Loading dataset from: {args.data}")
    print(f"Using captions: {args.captions}")
    
    # Load dataset
    dataset = TextSpriteDataset(
        root_dir=args.data,
        captions_file=args.captions,
        image_size=args.image_size,
    )
    
    print(f"Dataset size: {len(dataset)} sprite-text pairs")
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    # Create models
    print("Creating models...")
    generator = SpriteGenerator(
        image_size=args.image_size,
        noise_dim=args.noise_dim,
        text_embedding_dim=args.text_dim,
        out_channels=4,
    )
    
    discriminator = SpriteDiscriminator(
        image_size=args.image_size,
        in_channels=4,
        text_embedding_dim=args.text_dim,
    )
    
    text_encoder = SimpleTextEncoder(
        vocab_size=args.vocab_size,
        embedding_dim=args.text_dim,
    )
    
    # Create trainer
    config = GANTrainConfig(
        epochs=args.epochs,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        text_embedding_dim=args.text_dim,
        vocab_size=args.vocab_size,
        n_critic=args.n_critic,
        loss_mode=args.loss_mode,
        matching_weight=args.matching_weight,
        checkpoint_dir=args.output,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu",
    )
    
    trainer = GANTrainer(generator, discriminator, text_encoder, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Train
    trainer.fit(train_loader, start_epoch=start_epoch)
    
    return 0


def generate(args: argparse.Namespace) -> int:
    """
    Generate sprites from text descriptions.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Exit code (0 for success).
    """
    from spriteforge.models.discriminator import SpriteDiscriminator
    from spriteforge.models.generator import SpriteGenerator
    from spriteforge.models.text_encoder import SimpleTextEncoder
    from spriteforge.training.trainer import SimpleTokenizer
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    
    # Create models
    generator = SpriteGenerator(
        image_size=config["image_size"],
        noise_dim=config["noise_dim"],
        text_embedding_dim=config["text_embedding_dim"],
        out_channels=4,
    )
    
    text_encoder = SimpleTextEncoder(
        vocab_size=config["vocab_size"],
        embedding_dim=config["text_embedding_dim"],
    )
    
    # Load weights
    generator.load_state_dict(checkpoint["generator_state_dict"])
    text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
    generator.eval()
    text_encoder.eval()
    
    # Load tokenizer
    tokenizer = SimpleTokenizer(config["vocab_size"])
    tokenizer.word2idx = checkpoint["tokenizer_vocab"]
    
    # Prepare text descriptions
    if args.text:
        texts = [args.text]
    elif args.text_file:
        with open(args.text_file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Provide --text or --text-file")
        return 1
    
    print(f"Generating {args.num} sprites per text...")
    print(f"Texts: {texts}")
    
    # Generate sprites
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    generator.to(device)
    text_encoder.to(device)
    
    with torch.no_grad():
        # Encode text
        text_tokens = tokenizer.encode(texts).to(device)
        text_embeddings = text_encoder(text_tokens)
        
        # Repeat for multiple samples
        if args.num > 1:
            text_embeddings = text_embeddings.repeat_interleave(args.num, dim=0)
        
        # Generate
        batch_size = len(texts) * args.num
        noise = torch.randn(batch_size, config["noise_dim"], device=device)
        sprites = generator(noise, text_embeddings)
        
        # Convert from [-1, 1] to [0, 1]
        sprites = (sprites + 1.0) / 2.0
        sprites = sprites.clamp(0, 1)
    
    # Save sprites
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    idx = 0
    for text_idx, text in enumerate(texts):
        for sample_idx in range(args.num):
            sprite = sprites[idx].cpu()
            
            # Convert tensor to numpy array
            img_array = (sprite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Create PIL image
            img = Image.fromarray(img_array, mode="RGBA")
            
            # Optionally upscale
            if args.scale > 1:
                new_size = (img.width * args.scale, img.height * args.scale)
                img = img.resize(new_size, Image.Resampling.NEAREST)
            
            # Save
            safe_text = text.replace(" ", "_")[:20]
            filename = f"{safe_text}_{sample_idx:03d}.png"
            img.save(output_dir / filename)
            idx += 1
    
    print(f"Saved {idx} sprites to {output_dir}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SpriteForge - Text-to-sprite GAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a text-to-sprite GAN")
    train_parser.add_argument(
        "data",
        type=str,
        help="Path to sprite directory",
    )
    train_parser.add_argument(
        "--captions",
        type=str,
        required=True,
        help="Path to captions.json file",
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Image size (default: 32)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    train_parser.add_argument(
        "--noise-dim",
        type=int,
        default=100,
        help="Noise vector dimension (default: 100)",
    )
    train_parser.add_argument(
        "--text-dim",
        type=int,
        default=256,
        help="Text embedding dimension (default: 256)",
    )
    train_parser.add_argument(
        "--vocab-size",
        type=int,
        default=5000,
        help="Vocabulary size (default: 5000)",
    )
    train_parser.add_argument(
        "--g-lr",
        type=float,
        default=2e-4,
        help="Generator learning rate (default: 2e-4)",
    )
    train_parser.add_argument(
        "--d-lr",
        type=float,
        default=2e-4,
        help="Discriminator learning rate (default: 2e-4)",
    )
    train_parser.add_argument(
        "--n-critic",
        type=int,
        default=1,
        help="Discriminator updates per generator update (default: 1)",
    )
    train_parser.add_argument(
        "--loss-mode",
        type=str,
        default="bce",
        choices=["bce", "wgan", "wgan-gp"],
        help="Loss function (default: bce)",
    )
    train_parser.add_argument(
        "--matching-weight",
        type=float,
        default=1.0,
        help="Weight for text-matching loss (default: 1.0)",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints (default: checkpoints)",
    )
    train_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    train_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate sprites from text")
    gen_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file",
    )
    gen_parser.add_argument(
        "--text",
        type=str,
        help="Text description (e.g., 'red warrior')",
    )
    gen_parser.add_argument(
        "--text-file",
        type=str,
        help="File with text descriptions (one per line)",
    )
    gen_parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of sprites per text (default: 1)",
    )
    gen_parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Upscale factor (default: 1)",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default="generated",
        help="Output directory (default: generated)",
    )
    gen_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU generation",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run command
    if args.command == "train":
        return train(args)
    elif args.command == "generate":
        return generate(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
