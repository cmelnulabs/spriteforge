"""
Training loop for text-to-sprite GAN.

This module provides the GANTrainer class that handles GAN training
with alternating Generator/Discriminator optimization.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from spriteforge.models.discriminator import SpriteDiscriminator
from spriteforge.models.generator import SpriteGenerator
from spriteforge.models.text_encoder import SimpleTextEncoder
from spriteforge.training.losses import GANLoss


@dataclass
class GANTrainConfig:
    """Configuration for GAN training."""
    
    epochs: int = 100
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    batch_size: int = 32
    noise_dim: int = 100
    text_embedding_dim: int = 256
    vocab_size: int = 5000
    n_critic: int = 1  # D updates per G update
    loss_mode: str = "bce"
    matching_weight: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    save_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GANMetrics:
    """Metrics collected during GAN training."""
    
    epoch: int = 0
    g_loss: float = 0.0
    d_loss: float = 0.0
    d_real: float = 0.0
    d_fake: float = 0.0


class SimpleTokenizer:
    """Simple word-based tokenizer for sprite descriptions."""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.next_idx = 2
    
    def build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary from texts."""
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx and self.next_idx < self.vocab_size:
                    self.word2idx[word] = self.next_idx
                    self.idx2word[self.next_idx] = word
                    self.next_idx += 1
    
    def encode(self, texts: list[str], max_length: int = 32) -> torch.Tensor:
        """Encode texts to token indices."""
        batch_tokens = []
        for text in texts:
            words = text.lower().split()
            tokens = [self.word2idx.get(w, 1) for w in words]  # 1 = <UNK>
            # Pad or truncate
            if len(tokens) < max_length:
                tokens += [0] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
            batch_tokens.append(tokens)
        return torch.tensor(batch_tokens, dtype=torch.long)


class GANTrainer:
    """Trainer for text-conditional sprite GAN."""
    
    def __init__(
        self,
        generator: SpriteGenerator,
        discriminator: SpriteDiscriminator,
        text_encoder: SimpleTextEncoder,
        config: GANTrainConfig,
        tokenizer: SimpleTokenizer | None = None,
    ) -> None:
        """Initialize the GAN trainer."""
        self.generator = generator
        self.discriminator = discriminator
        self.text_encoder = text_encoder
        self.config = config
        self.device = torch.device(config.device)
        
        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.text_encoder.to(self.device)
        
        # Setup optimizers
        self.g_optimizer = Adam(
            self.generator.parameters(),
            lr=config.g_lr,
            betas=(0.5, 0.999),
        )
        self.d_optimizer = Adam(
            self.discriminator.parameters(),
            lr=config.d_lr,
            betas=(0.5, 0.999),
        )
        
        # Setup loss
        self.criterion = GANLoss(
            mode=config.loss_mode,
            matching_weight=config.matching_weight,
        )
        
        # Tokenizer
        self.tokenizer = tokenizer or SimpleTokenizer(config.vocab_size)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history: list[GANMetrics] = []
    
    def train_discriminator(
        self,
        real_images: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[float, float, float]:
        """Train discriminator for one step."""
        batch_size = real_images.size(0)
        
        # Generate fake images
        noise = torch.randn(batch_size, self.config.noise_dim, device=self.device)
        fake_images = self.generator(noise, text_embeddings).detach()
        
        # Forward pass
        real_realness, real_matching = self.discriminator(real_images, text_embeddings)
        fake_realness, fake_matching = self.discriminator(fake_images, text_embeddings)
        
        # Compute loss
        d_loss, _, _, _ = self.criterion.discriminator_loss(
            real_realness, fake_realness, real_matching, fake_matching
        )
        
        # Backward pass
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item(), real_realness.mean().item(), fake_realness.mean().item()
    
    def train_generator(
        self,
        batch_size: int,
        text_embeddings: torch.Tensor,
    ) -> float:
        """Train generator for one step."""
        # Generate fake images
        noise = torch.randn(batch_size, self.config.noise_dim, device=self.device)
        fake_images = self.generator(noise, text_embeddings)
        
        # Forward pass through discriminator
        fake_realness, fake_matching = self.discriminator(fake_images, text_embeddings)
        
        # Compute loss
        g_loss, _, _ = self.criterion.generator_loss(fake_realness, fake_matching)
        
        # Backward pass
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def train_epoch(self, train_loader: DataLoader) -> GANMetrics:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        self.text_encoder.eval()  # Text encoder stays fixed
        
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        d_real_sum = 0.0
        d_fake_sum = 0.0
        num_batches = 0
        
        progress = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, texts) in enumerate(progress):
            images = images.to(self.device)
            batch_size = images.size(0)
            
            # Tokenize and encode text
            text_tokens = self.tokenizer.encode(texts).to(self.device)
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_tokens)
            
            # Train Discriminator
            d_loss, d_real, d_fake = self.train_discriminator(images, text_embeddings)
            
            # Train Generator (every n_critic steps)
            if batch_idx % self.config.n_critic == 0:
                g_loss = self.train_generator(batch_size, text_embeddings)
            else:
                g_loss = 0.0
            
            # Accumulate metrics
            g_loss_sum += g_loss
            d_loss_sum += d_loss
            d_real_sum += d_real
            d_fake_sum += d_fake
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                progress.set_postfix({
                    "G": f"{g_loss:.4f}",
                    "D": f"{d_loss:.4f}",
                    "D(real)": f"{d_real:.3f}",
                    "D(fake)": f"{d_fake:.3f}",
                })
        
        # Compute averages
        return GANMetrics(
            g_loss=g_loss_sum / num_batches,
            d_loss=d_loss_sum / num_batches,
            d_real=d_real_sum / num_batches,
            d_fake=d_fake_sum / num_batches,
        )
    
    def fit(self, train_loader: DataLoader, start_epoch: int = 0) -> list[GANMetrics]:
        """Train the GAN for configured number of epochs."""
        print(f"Training on {self.device}")
        print(f"Generator parameters: {self.generator.count_parameters():,}")
        print(f"Discriminator parameters: {self.discriminator.count_parameters():,}")
        
        # Build vocabulary from first epoch
        if start_epoch == 0:
            print("Building vocabulary...")
            all_texts = []
            for _, texts in train_loader:
                all_texts.extend(texts)
            self.tokenizer.build_vocab(all_texts)
            print(f"Vocabulary size: {len(self.tokenizer.word2idx)}")
        
        for epoch in range(start_epoch, self.config.epochs):
            # Train
            metrics = self.train_epoch(train_loader)
            metrics.epoch = epoch
            
            # Log metrics
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"G Loss: {metrics.g_loss:.4f} | "
                f"D Loss: {metrics.d_loss:.4f} | "
                f"D(real): {metrics.d_real:.3f} | "
                f"D(fake): {metrics.d_fake:.3f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, metrics)
            
            self.history.append(metrics)
        
        print("Training complete!")
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: GANMetrics) -> None:
        """Save a training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "text_encoder_state_dict": self.text_encoder.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
            "d_optimizer_state_dict": self.d_optimizer.state_dict(),
            "tokenizer_vocab": self.tokenizer.word2idx,
            "config": {
                "image_size": self.generator.image_size,
                "noise_dim": self.config.noise_dim,
                "text_embedding_dim": self.config.text_embedding_dim,
                "vocab_size": self.config.vocab_size,
            },
            "metrics": {
                "g_loss": metrics.g_loss,
                "d_loss": metrics.d_loss,
            },
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        self.d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        self.tokenizer.word2idx = checkpoint["tokenizer_vocab"]
        
        return checkpoint["epoch"]
    
    @torch.no_grad()
    def generate_samples(self, texts: list[str], num_samples: int = 1) -> torch.Tensor:
        """Generate sprites from text descriptions."""
        self.generator.eval()
        self.text_encoder.eval()
        
        # Encode text
        text_tokens = self.tokenizer.encode(texts).to(self.device)
        text_embeddings = self.text_encoder(text_tokens)
        
        # Repeat for multiple samples
        if num_samples > 1:
            text_embeddings = text_embeddings.repeat_interleave(num_samples, dim=0)
        
        # Generate
        batch_size = len(texts) * num_samples
        noise = torch.randn(batch_size, self.config.noise_dim, device=self.device)
        sprites = self.generator(noise, text_embeddings)
        
        return sprites
