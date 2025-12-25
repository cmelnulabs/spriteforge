"""
Training loop and utilities for SpriteForge models.

This module provides the main Trainer class that handles the complete
training workflow including optimization, logging, and checkpointing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from spriteforge.models.vae import SpriteVAE, VAEOutput
from spriteforge.training.losses import VAELoss


@dataclass
class TrainConfig:
    """
    Configuration for training.
    
    Attributes:
        epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        batch_size: Training batch size.
        beta: KL divergence weight for VAE loss.
        beta_warmup_epochs: Epochs to warm up beta from 0.
        weight_decay: L2 regularization strength.
        checkpoint_dir: Directory to save checkpoints.
        log_interval: Batches between logging.
        save_interval: Epochs between checkpoint saves.
        device: Device to train on ('cuda' or 'cpu').
    """
    
    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 32
    beta: float = 1.0
    beta_warmup_epochs: int = 10
    weight_decay: float = 1e-5
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    save_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainMetrics:
    """
    Metrics collected during training.
    
    Attributes:
        epoch: Current epoch number.
        total_loss: Average total VAE loss.
        recon_loss: Average reconstruction loss.
        kl_loss: Average KL divergence loss.
        learning_rate: Current learning rate.
    """
    
    epoch: int = 0
    total_loss: float = 0.0
    recon_loss: float = 0.0
    kl_loss: float = 0.0
    learning_rate: float = 0.0


class Trainer:
    """
    Trainer for SpriteVAE models.
    
    Handles the complete training loop including:
    - Optimization with AdamW
    - Learning rate scheduling
    - Loss computation with beta warmup
    - Checkpointing and logging
    - TensorBoard integration (optional)
    
    Example:
        >>> model = SpriteVAE(image_size=32, latent_dim=128)
        >>> config = TrainConfig(epochs=100, learning_rate=1e-4)
        >>> trainer = Trainer(model, config)
        >>> trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: SpriteVAE,
        config: TrainConfig,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: The SpriteVAE model to train.
            config: Training configuration.
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup learning rate scheduler
        self.scheduler: LRScheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )
        
        # Setup loss function
        self.criterion = VAELoss(
            beta=config.beta,
            beta_warmup_epochs=config.beta_warmup_epochs,
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history: list[TrainMetrics] = []
        self.best_loss = float("inf")
        
        # TensorBoard writer (optional)
        self.writer: Any = None
    
    def setup_tensorboard(self, log_dir: str = "runs") -> None:
        """
        Setup TensorBoard logging.
        
        Args:
            log_dir: Directory for TensorBoard logs.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def train_epoch(self, train_loader: DataLoader) -> TrainMetrics:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data.
        
        Returns:
            Training metrics for the epoch.
        """
        self.model.train()
        
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0
        
        progress = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress):
            # Handle both tensor and tuple returns from dataloader
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output: VAEOutput = self.model(x)
            
            # Compute loss
            total, recon, kl = self.criterion(
                output.reconstruction, x, output.mu, output.log_var
            )
            
            # Backward pass
            total.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss_sum += total.item()
            recon_loss_sum += recon.item()
            kl_loss_sum += kl.item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                progress.set_postfix({
                    "loss": f"{total.item():.4f}",
                    "recon": f"{recon.item():.4f}",
                    "kl": f"{kl.item():.4f}",
                })
        
        # Compute averages
        return TrainMetrics(
            total_loss=total_loss_sum / num_batches,
            recon_loss=recon_loss_sum / num_batches,
            kl_loss=kl_loss_sum / num_batches,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> TrainMetrics:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data.
        
        Returns:
            Validation metrics.
        """
        self.model.eval()
        
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0
        
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.to(self.device)
            
            output: VAEOutput = self.model(x)
            total, recon, kl = self.criterion(
                output.reconstruction, x, output.mu, output.log_var
            )
            
            total_loss_sum += total.item()
            recon_loss_sum += recon.item()
            kl_loss_sum += kl.item()
            num_batches += 1
        
        return TrainMetrics(
            total_loss=total_loss_sum / num_batches,
            recon_loss=recon_loss_sum / num_batches,
            kl_loss=kl_loss_sum / num_batches,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        start_epoch: int = 0,
    ) -> list[TrainMetrics]:
        """
        Train the model for the configured number of epochs.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
            start_epoch: Epoch to start from (for resuming).
        
        Returns:
            List of training metrics for each epoch.
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(start_epoch, self.config.epochs):
            # Update beta warmup
            self.criterion.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            train_metrics.epoch = epoch
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics.total_loss
            else:
                val_loss = train_metrics.total_loss
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Loss: {train_metrics.total_loss:.4f} | "
                f"Recon: {train_metrics.recon_loss:.4f} | "
                f"KL: {train_metrics.kl_loss:.4f} | "
                f"Beta: {self.criterion.current_beta:.4f} | "
                f"LR: {train_metrics.learning_rate:.6f}"
            )
            
            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar("Loss/total", train_metrics.total_loss, epoch)
                self.writer.add_scalar("Loss/reconstruction", train_metrics.recon_loss, epoch)
                self.writer.add_scalar("Loss/kl", train_metrics.kl_loss, epoch)
                self.writer.add_scalar("Params/beta", self.criterion.current_beta, epoch)
                self.writer.add_scalar("Params/lr", train_metrics.learning_rate, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, train_metrics)
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, train_metrics, is_best=True)
            
            self.history.append(train_metrics)
        
        print("Training complete!")
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: TrainMetrics,
        is_best: bool = False,
    ) -> None:
        """
        Save a training checkpoint.
        
        Args:
            epoch: Current epoch number.
            metrics: Current training metrics.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": {
                "total_loss": metrics.total_loss,
                "recon_loss": metrics.recon_loss,
                "kl_loss": metrics.kl_loss,
            },
            "config": {
                "image_size": self.model.image_size,
                "latent_dim": self.model.latent_dim,
                "in_channels": self.model.in_channels,
            },
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (loss: {metrics.total_loss:.4f})")
    
    def load_checkpoint(self, path: str | Path) -> int:
        """
        Load a training checkpoint.
        
        Args:
            path: Path to the checkpoint file.
        
        Returns:
            The epoch number of the loaded checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint["epoch"]
