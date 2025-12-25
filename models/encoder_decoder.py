"""
Encoder and Decoder network implementations for SpriteForge VAE.

This module contains convolutional neural network architectures optimized
for encoding and decoding small pixel art sprites (32x32 or 64x64).
"""

import torch
import torch.nn as nn

from spriteforge.models.base import BaseDecoder, BaseEncoder


class ConvEncoder(BaseEncoder):
    """
    Convolutional encoder for sprite images.
    
    Uses a series of strided convolutions to progressively downsample
    the input image while increasing channel depth, then projects
    to latent distribution parameters (mean and log-variance).
    
    Architecture (for 32x32 input):
        32x32x4 -> 16x16x64 -> 8x8x128 -> 4x4x256 -> flatten -> latent
    
    Attributes:
        image_size: Expected input image size (assumes square images).
        in_channels: Number of input channels (4 for RGBA).
        latent_dim: Dimension of the latent space.
    
    Example:
        >>> encoder = ConvEncoder(image_size=32, latent_dim=128)
        >>> x = torch.randn(16, 4, 32, 32)  # batch of 16 RGBA sprites
        >>> mu, log_var = encoder(x)
        >>> mu.shape
        torch.Size([16, 128])
    """
    
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 4,
        latent_dim: int = 128,
        base_channels: int = 64,
    ) -> None:
        """
        Initialize the convolutional encoder.
        
        Args:
            image_size: Input image size (32 or 64 supported).
            in_channels: Number of input channels (default 4 for RGBA).
            latent_dim: Dimension of the latent space.
            base_channels: Base number of channels (doubled each layer).
        """
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Determine number of downsampling layers based on image size
        # 32x32 -> 4x4 needs 3 layers, 64x64 -> 4x4 needs 4 layers
        self.num_layers = 3 if image_size == 32 else 4
        
        # Build convolutional layers
        layers: list[nn.Module] = []
        current_channels = in_channels
        
        for i in range(self.num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            current_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        # After num_layers of stride-2 convs: image_size / (2^num_layers)
        self.final_spatial = image_size // (2 ** self.num_layers)
        self.final_channels = base_channels * (2 ** (self.num_layers - 1))
        self.flatten_size = self.final_channels * (self.final_spatial ** 2)
        
        # Project to latent distribution parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_size, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input images to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
               Expected shape: (B, 4, image_size, image_size) for RGBA sprites.
        
        Returns:
            Tuple of (mu, log_var):
                - mu: Mean of latent distribution, shape (B, latent_dim).
                - log_var: Log variance of latent distribution, shape (B, latent_dim).
        """
        # Pass through convolutional layers
        h = self.conv_layers(x)
        
        # Flatten
        h = h.view(h.size(0), -1)
        
        # Project to distribution parameters
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        return mu, log_var


class ConvDecoder(BaseDecoder):
    """
    Convolutional decoder for sprite images.
    
    Uses transposed convolutions to progressively upsample from a latent
    vector back to the original image dimensions.
    
    Architecture (for 32x32 output):
        latent -> 4x4x256 -> 8x8x128 -> 16x16x64 -> 32x32x4
    
    Attributes:
        image_size: Output image size (assumes square images).
        out_channels: Number of output channels (4 for RGBA).
        latent_dim: Dimension of the latent space.
    
    Example:
        >>> decoder = ConvDecoder(image_size=32, latent_dim=128)
        >>> z = torch.randn(16, 128)  # batch of 16 latent vectors
        >>> x_recon = decoder(z)
        >>> x_recon.shape
        torch.Size([16, 4, 32, 32])
    """
    
    def __init__(
        self,
        image_size: int = 32,
        out_channels: int = 4,
        latent_dim: int = 128,
        base_channels: int = 64,
    ) -> None:
        """
        Initialize the convolutional decoder.
        
        Args:
            image_size: Output image size (32 or 64 supported).
            out_channels: Number of output channels (default 4 for RGBA).
            latent_dim: Dimension of the latent space.
            base_channels: Base number of channels.
        """
        super().__init__()
        
        self.image_size = image_size
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        # Match encoder's number of layers
        self.num_layers = 3 if image_size == 32 else 4
        
        # Calculate initial spatial size and channels
        self.initial_spatial = image_size // (2 ** self.num_layers)
        self.initial_channels = base_channels * (2 ** (self.num_layers - 1))
        self.unflatten_size = self.initial_channels * (self.initial_spatial ** 2)
        
        # Project from latent to initial feature map
        self.fc = nn.Linear(latent_dim, self.unflatten_size)
        
        # Build transposed convolutional layers
        layers: list[nn.Module] = []
        current_channels = self.initial_channels
        
        for i in range(self.num_layers - 1, -1, -1):
            # Last layer outputs to out_channels, others halve the channels
            if i == 0:
                next_channels = out_channels
                layers.extend([
                    nn.ConvTranspose2d(
                        current_channels,
                        next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    ),
                    nn.Sigmoid(),  # Output in [0, 1] range
                ])
            else:
                next_channels = base_channels * (2 ** (i - 1))
                layers.extend([
                    nn.ConvTranspose2d(
                        current_channels,
                        next_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ])
            current_channels = next_channels
        
        self.deconv_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.
        
        Args:
            z: Latent tensor of shape (batch, latent_dim).
        
        Returns:
            Reconstructed images of shape (batch, out_channels, image_size, image_size).
            Values are in [0, 1] range.
        """
        # Project and reshape
        h = self.fc(z)
        h = h.view(-1, self.initial_channels, self.initial_spatial, self.initial_spatial)
        
        # Pass through transposed convolutions
        x_recon = self.deconv_layers(h)
        
        return x_recon
