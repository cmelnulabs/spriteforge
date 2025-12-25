"""
Loss functions for VAE training.

This module implements the VAE loss function which combines
reconstruction loss with KL divergence regularization.
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute pixel-wise reconstruction loss.
    
    Uses binary cross-entropy for images with values in [0, 1].
    BCE is preferred over MSE for pixel art as it handles the
    discrete nature of pixel colors better.
    
    Args:
        x_recon: Reconstructed images, shape (B, C, H, W).
        x_target: Target images, shape (B, C, H, W).
        reduction: 'mean', 'sum', or 'none'.
    
    Returns:
        Reconstruction loss value.
    """
    # Use BCE loss for [0, 1] normalized images
    loss = F.binary_cross_entropy(x_recon, x_target, reduction=reduction)
    
    return loss


def kl_divergence(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence from latent distribution to standard normal.
    
    KL(q(z|x) || p(z)) where q is the encoder distribution N(mu, sigma)
    and p is the prior N(0, I).
    
    The closed-form solution for Gaussian distributions:
    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    Args:
        mu: Mean of the latent distribution, shape (B, latent_dim).
        log_var: Log variance of the latent distribution, shape (B, latent_dim).
        reduction: 'mean', 'sum', or 'none'.
    
    Returns:
        KL divergence value.
    """
    # KL divergence formula
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl


def vae_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the full VAE loss (ELBO).
    
    Loss = Reconstruction Loss + beta * KL Divergence
    
    The beta parameter controls the weight of the KL term:
    - beta = 1: Standard VAE
    - beta < 1: Less regularization, better reconstruction
    - beta > 1: More regularization, smoother latent space (beta-VAE)
    
    Args:
        x_recon: Reconstructed images from decoder.
        x_target: Original input images.
        mu: Mean from encoder.
        log_var: Log variance from encoder.
        beta: Weight for KL divergence term.
    
    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    
    Example:
        >>> output = vae(x)
        >>> total, recon, kl = vae_loss(
        ...     output.reconstruction, x, output.mu, output.log_var
        ... )
        >>> total.backward()
    """
    recon = reconstruction_loss(x_recon, x_target)
    kl = kl_divergence(mu, log_var)
    
    total = recon + beta * kl
    
    return total, recon, kl


class VAELoss:
    """
    Callable VAE loss class with configurable parameters.
    
    Provides a cleaner interface for loss computation during training.
    
    Attributes:
        beta: KL divergence weight.
        beta_warmup_epochs: Number of epochs to warm up beta from 0.
    
    Example:
        >>> criterion = VAELoss(beta=1.0, beta_warmup_epochs=10)
        >>> for epoch in range(100):
        ...     criterion.set_epoch(epoch)
        ...     loss, recon, kl = criterion(output, target)
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        beta_warmup_epochs: int = 0,
    ) -> None:
        """
        Initialize the VAE loss.
        
        Args:
            beta: Target KL divergence weight.
            beta_warmup_epochs: Epochs to linearly warm up beta.
        """
        self.beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self._current_epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for beta warmup.
        
        Args:
            epoch: Current training epoch (0-indexed).
        """
        self._current_epoch = epoch
    
    @property
    def current_beta(self) -> float:
        """Get the current beta value considering warmup."""
        if self.beta_warmup_epochs <= 0:
            return self.beta
        
        warmup_factor = min(1.0, self._current_epoch / self.beta_warmup_epochs)
        return self.beta * warmup_factor
    
    def __call__(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            x_recon: Reconstructed images.
            x_target: Target images.
            mu: Latent mean.
            log_var: Latent log variance.
        
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss).
        """
        return vae_loss(x_recon, x_target, mu, log_var, beta=self.current_beta)


def perceptual_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    feature_extractor: torch.nn.Module,
) -> torch.Tensor:
    """
    Compute perceptual loss using feature extractor.
    
    Measures similarity in feature space rather than pixel space,
    which can lead to perceptually better reconstructions.
    
    Note: Requires a pre-trained feature extractor (e.g., VGG).
    
    Args:
        x_recon: Reconstructed images.
        x_target: Target images.
        feature_extractor: Pre-trained network for feature extraction.
    
    Returns:
        Perceptual loss value.
    """
    # Extract features (assuming 3-channel input for VGG)
    # For RGBA, we'd need to drop alpha or use a different extractor
    features_recon = feature_extractor(x_recon[:, :3])
    features_target = feature_extractor(x_target[:, :3])
    
    # MSE in feature space
    loss = F.mse_loss(features_recon, features_target)
    
    return loss
