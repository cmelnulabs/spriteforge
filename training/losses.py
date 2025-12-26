"""
Loss functions for text-to-sprite GAN training.

This module implements loss functions for training GANs including
adversarial losses and auxiliary text-matching losses.
"""

import torch
import torch.nn.functional as F


def generator_loss_bce(
    fake_realness: torch.Tensor,
    fake_matching: torch.Tensor,
    matching_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Generator loss using Binary Cross-Entropy.
    
    Generator wants:
    1. Discriminator to think fake images are real (realness = 1)
    2. Discriminator to think fake images match the text (matching = 1)
    
    Args:
        fake_realness: Discriminator's realness score for fake images, shape (batch, 1).
        fake_matching: Discriminator's matching score for fake images, shape (batch, 1).
        matching_weight: Weight for the matching loss term.
    
    Returns:
        Tuple of (total_loss, realness_loss, matching_loss).
    
    Example:
        >>> fake_realness = torch.rand(16, 1)  # Discriminator output for fakes
        >>> fake_matching = torch.rand(16, 1)
        >>> total, real_loss, match_loss = generator_loss_bce(fake_realness, fake_matching)
    """
    # Generator wants discriminator to output 1 (real) for fake images
    target_real = torch.ones_like(fake_realness)
    realness_loss = F.binary_cross_entropy(fake_realness, target_real)
    
    # Generator wants matching score to be 1 (text matches image)
    target_match = torch.ones_like(fake_matching)
    matching_loss = F.binary_cross_entropy(fake_matching, target_match)
    
    total = realness_loss + matching_weight * matching_loss
    
    return total, realness_loss, matching_loss


def discriminator_loss_bce(
    real_realness: torch.Tensor,
    fake_realness: torch.Tensor,
    real_matching: torch.Tensor,
    fake_matching: torch.Tensor,
    wrong_matching: torch.Tensor | None = None,
    matching_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Discriminator loss using Binary Cross-Entropy.
    
    Discriminator learns to:
    1. Classify real images as real (realness = 1)
    2. Classify fake images as fake (realness = 0)
    3. Match correct text-image pairs (matching = 1)
    4. Reject wrong text-image pairs (matching = 0)
    
    Args:
        real_realness: Realness score for real images, shape (batch, 1).
        fake_realness: Realness score for fake images, shape (batch, 1).
        real_matching: Matching score for correct text-image pairs, shape (batch, 1).
        fake_matching: Matching score for fake images with correct text, shape (batch, 1).
        wrong_matching: Optional matching score for real images with wrong text, shape (batch, 1).
        matching_weight: Weight for matching loss terms.
    
    Returns:
        Tuple of (total_loss, realness_loss, matching_loss, wrong_loss).
    
    Example:
        >>> real_realness = torch.rand(16, 1)
        >>> fake_realness = torch.rand(16, 1)
        >>> real_matching = torch.rand(16, 1)
        >>> fake_matching = torch.rand(16, 1)
        >>> total, real_loss, match_loss, wrong_loss = discriminator_loss_bce(
        ...     real_realness, fake_realness, real_matching, fake_matching
        ... )
    """
    # Realness loss: real=1, fake=0
    target_real = torch.ones_like(real_realness)
    target_fake = torch.zeros_like(fake_realness)
    
    realness_loss = (
        F.binary_cross_entropy(real_realness, target_real) +
        F.binary_cross_entropy(fake_realness, target_fake)
    ) / 2
    
    # Matching loss: correct pairs=1, incorrect pairs=0
    target_match = torch.ones_like(real_matching)
    target_no_match = torch.zeros_like(fake_matching)
    
    matching_loss = (
        F.binary_cross_entropy(real_matching, target_match) +
        F.binary_cross_entropy(fake_matching, target_no_match)
    ) / 2
    
    # Optional: penalize real images with wrong text
    wrong_loss = torch.tensor(0.0, device=real_realness.device)
    if wrong_matching is not None:
        wrong_loss = F.binary_cross_entropy(wrong_matching, target_no_match)
        matching_loss = (matching_loss * 2 + wrong_loss) / 3
    
    total = realness_loss + matching_weight * matching_loss
    
    return total, realness_loss, matching_loss, wrong_loss


def generator_loss_wgan(
    fake_realness: torch.Tensor,
    fake_matching: torch.Tensor,
    matching_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Generator loss using Wasserstein GAN objective.
    
    Wasserstein distance provides better training stability and
    meaningful loss curves compared to BCE.
    
    Args:
        fake_realness: Discriminator output for fake images (not sigmoid-activated).
        fake_matching: Matching score for fake images (sigmoid-activated).
        matching_weight: Weight for matching loss.
    
    Returns:
        Tuple of (total_loss, realness_loss, matching_loss).
    """
    # Generator wants to maximize discriminator output for fakes
    # Equivalent to minimizing the negative
    realness_loss = -fake_realness.mean()
    
    # Matching loss remains BCE-based
    target_match = torch.ones_like(fake_matching)
    matching_loss = F.binary_cross_entropy(fake_matching, target_match)
    
    total = realness_loss + matching_weight * matching_loss
    
    return total, realness_loss, matching_loss


def discriminator_loss_wgan(
    real_realness: torch.Tensor,
    fake_realness: torch.Tensor,
    real_matching: torch.Tensor,
    fake_matching: torch.Tensor,
    matching_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Discriminator loss using Wasserstein GAN objective.
    
    Args:
        real_realness: Discriminator output for real images (not sigmoid-activated).
        fake_realness: Discriminator output for fake images (not sigmoid-activated).
        real_matching: Matching score for correct pairs (sigmoid-activated).
        fake_matching: Matching score for incorrect pairs (sigmoid-activated).
        matching_weight: Weight for matching loss.
    
    Returns:
        Tuple of (total_loss, realness_loss, matching_loss).
    """
    # Wasserstein distance: maximize D(real) - D(fake)
    # Equivalent to minimizing -(D(real) - D(fake)) = D(fake) - D(real)
    realness_loss = fake_realness.mean() - real_realness.mean()
    
    # Matching loss (BCE-based)
    target_match = torch.ones_like(real_matching)
    target_no_match = torch.zeros_like(fake_matching)
    
    matching_loss = (
        F.binary_cross_entropy(real_matching, target_match) +
        F.binary_cross_entropy(fake_matching, target_no_match)
    ) / 2
    
    total = realness_loss + matching_weight * matching_loss
    
    return total, realness_loss, matching_loss


def gradient_penalty(
    discriminator: torch.nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    text_embedding: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.
    
    Enforces 1-Lipschitz constraint by penalizing gradient norm deviation from 1.
    
    Args:
        discriminator: The discriminator network.
        real_images: Real sprite images.
        fake_images: Generated sprite images.
        text_embedding: Text embeddings.
        lambda_gp: Gradient penalty coefficient.
    
    Returns:
        Gradient penalty loss.
    """
    batch_size = real_images.size(0)
    device = real_images.device
    
    # Random interpolation weight
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake images
    interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    
    # Get discriminator output for interpolated images
    d_interpolated, _ = discriminator(interpolated, text_embedding)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Compute gradient norm
    gradient_norm = gradients.norm(2, dim=1)
    
    # Penalty for deviation from norm=1
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty


class GANLoss:
    """
    Configurable GAN loss wrapper.
    
    Supports multiple GAN objectives:
    - BCE (vanilla GAN)
    - WGAN (Wasserstein GAN)
    - WGAN-GP (Wasserstein GAN with Gradient Penalty)
    
    Attributes:
        mode: Loss mode ('bce', 'wgan', 'wgan-gp').
        matching_weight: Weight for text-matching loss.
        lambda_gp: Gradient penalty weight (for WGAN-GP).
    
    Example:
        >>> criterion = GANLoss(mode='bce', matching_weight=1.0)
        >>> # For discriminator
        >>> d_loss, *_ = criterion.discriminator_loss(
        ...     real_realness, fake_realness, real_matching, fake_matching
        ... )
        >>> # For generator  
        >>> g_loss, *_ = criterion.generator_loss(fake_realness, fake_matching)
    """
    
    def __init__(
        self,
        mode: str = "bce",
        matching_weight: float = 1.0,
        lambda_gp: float = 10.0,
    ) -> None:
        """
        Initialize GAN loss.
        
        Args:
            mode: Loss mode ('bce', 'wgan', 'wgan-gp').
            matching_weight: Weight for matching loss component.
            lambda_gp: Gradient penalty weight (only for wgan-gp).
        """
        assert mode in ["bce", "wgan", "wgan-gp"], f"Invalid mode: {mode}"
        
        self.mode = mode
        self.matching_weight = matching_weight
        self.lambda_gp = lambda_gp
    
    def generator_loss(
        self,
        fake_realness: torch.Tensor,
        fake_matching: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute generator loss based on mode."""
        if self.mode == "bce":
            return generator_loss_bce(fake_realness, fake_matching, self.matching_weight)
        else:  # wgan or wgan-gp (same for generator)
            return generator_loss_wgan(fake_realness, fake_matching, self.matching_weight)
    
    def discriminator_loss(
        self,
        real_realness: torch.Tensor,
        fake_realness: torch.Tensor,
        real_matching: torch.Tensor,
        fake_matching: torch.Tensor,
        wrong_matching: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Compute discriminator loss based on mode."""
        if self.mode == "bce":
            return discriminator_loss_bce(
                real_realness,
                fake_realness,
                real_matching,
                fake_matching,
                wrong_matching,
                self.matching_weight,
            )
        else:  # wgan or wgan-gp
            return discriminator_loss_wgan(
                real_realness,
                fake_realness,
                real_matching,
                fake_matching,
                self.matching_weight,
            )

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
