"""
Discriminator network for text-to-sprite GAN.

This module implements the Discriminator that evaluates whether
sprites are real or fake, and whether they match the text description.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from spriteforge.models.base import BaseDiscriminator


@dataclass
class DiscriminatorOutput:
    """
    Container for Discriminator outputs.
    
    Attributes:
        realness_score: Probability that image is real (vs fake).
        matching_score: Probability that text matches the image.
    """
    
    realness_score: torch.Tensor
    matching_score: torch.Tensor


class SpriteDiscriminator(BaseDiscriminator):
    """
    Text-conditional Discriminator for sprite images.
    
    Evaluates sprites on two criteria:
    1. Real vs Fake: Is this a real sprite or generated?
    2. Text Matching: Does the sprite match the text description?
    
    This dual-task design helps the generator produce sprites that
    are both realistic and semantically aligned with the text.
    
    Architecture:
        Image: 32x32x4 -> 16x16x64 -> 8x8x128 -> 4x4x256 -> flatten
        Text: embedding_dim -> 256
        Combined: [image_features + text_features] -> realness + matching scores
    
    Attributes:
        image_size: Input sprite size.
        in_channels: Number of input channels (4 for RGBA).
        text_embedding_dim: Dimension of text embeddings.
    
    Example:
        >>> disc = SpriteDiscriminator(image_size=32, text_embedding_dim=256)
        >>> images = torch.randn(8, 4, 32, 32)
        >>> text_emb = torch.randn(8, 256)
        >>> realness, matching = disc(images, text_emb)
        >>> realness.shape, matching.shape
        (torch.Size([8, 1]), torch.Size([8, 1]))
    """
    
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 4,
        text_embedding_dim: int = 256,
        base_channels: int = 64,
    ) -> None:
        """
        Initialize the Discriminator.
        
        Args:
            image_size: Input image size (32 or 64).
            in_channels: Number of input channels (4 for RGBA).
            text_embedding_dim: Dimension of text embeddings.
            base_channels: Base number of channels for conv layers.
        """
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.text_embedding_dim = text_embedding_dim
        
        # Calculate number of downsampling layers
        self.num_layers = 3 if image_size == 32 else 4
        
        # Image processing path
        layers: list[nn.Module] = []
        current_channels = in_channels
        
        for i in range(self.num_layers):
            out_channels = base_channels * (2 ** i)
            
            # First layer doesn't use batch norm (PatchGAN style)
            use_bn = i > 0
            
            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_bn,
                )
            )
            
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = out_channels
        
        self.image_encoder = nn.Sequential(*layers)
        
        # Calculate flattened image feature size
        self.final_spatial = image_size // (2 ** self.num_layers)
        self.final_channels = base_channels * (2 ** (self.num_layers - 1))
        self.image_feature_dim = self.final_channels * (self.final_spatial ** 2)
        
        # Text processing path
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.text_feature_dim = 256
        
        # Combined feature dimension
        self.combined_dim = self.image_feature_dim + self.text_feature_dim
        
        # Realness classifier (real vs fake)
        self.realness_head = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        
        # Matching classifier (text matches image)
        self.matching_head = nn.Sequential(
            nn.Linear(self.combined_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using normal distribution."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        image: torch.Tensor, 
        text_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminate images and check text-image matching.
        
        Args:
            image: Input sprite of shape (batch, in_channels, image_size, image_size).
                   Values should be in [-1, 1] range.
            text_embedding: Text embedding of shape (batch, text_embedding_dim).
            
        Returns:
            Tuple of (realness_score, matching_score):
                - realness_score: Probability image is real, shape (batch, 1).
                - matching_score: Probability text matches image, shape (batch, 1).
        """
        batch_size = image.size(0)
        
        # Encode image
        image_features = self.image_encoder(image)
        image_features = image_features.view(batch_size, -1)  # Flatten
        
        # Encode text
        text_features = self.text_encoder(text_embedding)
        
        # Combine features
        combined = torch.cat([image_features, text_features], dim=1)
        
        # Compute scores
        realness = self.realness_head(combined)
        matching = self.matching_head(combined)
        
        return realness, matching
    
    def discriminate_realness(self, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Only compute realness score (for training efficiency).
        
        Args:
            image: Input sprite.
            text_embedding: Text embedding.
            
        Returns:
            Realness score of shape (batch, 1).
        """
        realness, _ = self.forward(image, text_embedding)
        return realness
    
    def discriminate_matching(self, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Only compute matching score (for training efficiency).
        
        Args:
            image: Input sprite.
            text_embedding: Text embedding.
            
        Returns:
            Matching score of shape (batch, 1).
        """
        _, matching = self.forward(image, text_embedding)
        return matching


class PatchDiscriminator(BaseDiscriminator):
    """
    PatchGAN-style Discriminator for sprite images.
    
    Instead of outputting a single score, outputs a grid of scores
    (one per patch). This encourages sharper, more detailed sprites.
    
    More suitable for higher resolution sprites (64x64+).
    
    Example:
        >>> disc = PatchDiscriminator(image_size=64, text_embedding_dim=256)
        >>> images = torch.randn(8, 4, 64, 64)
        >>> text_emb = torch.randn(8, 256)
        >>> realness, matching = disc(images, text_emb)
        >>> realness.shape  # e.g., (8, 1, 4, 4) - score per patch
    """
    
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 4,
        text_embedding_dim: int = 256,
        base_channels: int = 64,
    ) -> None:
        """Initialize the Patch Discriminator."""
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.text_embedding_dim = text_embedding_dim
        
        # Similar to regular discriminator but doesn't fully flatten
        # Uses conv layers until we have a small spatial grid (e.g., 4x4)
        
        # Image encoder (keeps spatial structure)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Text conditioning (broadcast to spatial dimensions)
        self.text_projection = nn.Linear(text_embedding_dim, base_channels * 4)
        
        # Output layers (per-patch scores)
        self.realness_conv = nn.Conv2d(base_channels * 4, 1, 3, 1, 1)
        self.matching_conv = nn.Conv2d(base_channels * 4, 1, 3, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        image: torch.Tensor, 
        text_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminate with per-patch scores.
        
        Returns:
            Tuple of (realness_map, matching_map) with spatial dimensions.
        """
        batch_size = image.size(0)
        
        # Encode image (preserve spatial structure)
        img_features = self.image_encoder(image)  # (batch, C, H, W)
        _, C, H, W = img_features.shape
        
        # Project text and broadcast to spatial dimensions
        text_features = self.text_projection(text_embedding)  # (batch, C)
        text_features = text_features.view(batch_size, C, 1, 1)
        text_features = text_features.expand(-1, -1, H, W)  # (batch, C, H, W)
        
        # Combine with element-wise multiplication (gating)
        combined = img_features * text_features
        
        # Per-patch scores
        realness = torch.sigmoid(self.realness_conv(combined))
        matching = torch.sigmoid(self.matching_conv(combined))
        
        # Average over spatial dimensions for final score
        realness = realness.mean(dim=[2, 3], keepdim=True)
        matching = matching.mean(dim=[2, 3], keepdim=True)
        
        return realness, matching
