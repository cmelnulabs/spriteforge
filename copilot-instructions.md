# SpriteForge - Copilot Instructions

> Guidelines for AI-assisted development and clean code practices.

---

## 🎯 Project Overview

**SpriteForge** is a Variational Autoencoder (VAE) for generating 2D pixel art sprites.

- **Language**: Python 3.10+
- **Framework**: PyTorch
- **Style**: Clean, documented, educational

---

## 📁 Project Structure

```
spriteforge/
├── spriteforge/              # Main package
│   ├── __init__.py
│   ├── models/               # Neural network architectures
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base classes
│   │   ├── vae.py            # Vanilla VAE
│   │   ├── conv_vae.py       # Convolutional VAE
│   │   └── encoder_decoder.py
│   ├── data/                 # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py        # PyTorch Dataset classes
│   │   ├── transforms.py     # Image transformations
│   │   └── utils.py
│   ├── training/             # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop
│   │   ├── losses.py         # Loss functions
│   │   └── metrics.py
│   ├── utils/                # General utilities
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   ├── logging.py        # Logging setup
│   │   └── visualization.py
│   └── cli/                  # Command-line interface
│       ├── __init__.py
│       ├── train.py
│       └── generate.py
├── tests/                    # Unit and integration tests
├── configs/                  # YAML configuration files
├── data/                     # Dataset storage (gitignored)
├── outputs/                  # Training outputs (gitignored)
├── notebooks/                # Jupyter notebooks
├── docs/                     # Documentation
├── ROADMAP.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 🧹 Code Style Rules

### General Principles

1. **Clarity over cleverness** - Code should be readable by beginners
2. **Explicit over implicit** - Avoid magic; document assumptions
3. **Single responsibility** - Each function/class does one thing well
4. **DRY but not at the cost of clarity** - Some duplication is acceptable for readability

### Python Style

```python
# ✅ GOOD: Clear, documented, type-hinted
def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode input image to latent space parameters.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        
    Returns:
        Tuple of (mu, log_var) tensors, each of shape (batch_size, latent_dim)
    """
    h = self.encoder(x)
    mu = self.fc_mu(h)
    log_var = self.fc_log_var(h)
    return mu, log_var

# ❌ BAD: Unclear, no documentation
def enc(self, x):
    h = self.e(x)
    return self.m(h), self.v(h)
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `ConvolutionalEncoder` |
| Functions | snake_case | `compute_kl_divergence` |
| Variables | snake_case | `latent_dim` |
| Constants | UPPER_SNAKE | `DEFAULT_LATENT_DIM` |
| Private | _prefix | `_validate_input` |
| Type vars | Single uppercase | `T`, `Tensor` |

### Imports

```python
# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local
from spriteforge.models.base import BaseVAE
from spriteforge.utils.config import Config
```

### Documentation

Every public class and function MUST have a docstring:

```python
class VAE(BaseVAE):
    """
    Variational Autoencoder for sprite generation.
    
    This implementation follows the original VAE paper by Kingma & Welling (2013).
    The encoder maps input images to a distribution in latent space (parameterized
    by mean and log-variance), and the decoder reconstructs images from samples
    of this distribution.
    
    Architecture:
        Encoder: Input → Conv layers → Flatten → FC → (μ, log σ²)
        Decoder: z → FC → Reshape → ConvTranspose layers → Output
    
    Attributes:
        latent_dim: Dimension of the latent space.
        encoder: Encoder network.
        decoder: Decoder network.
        
    Example:
        >>> model = VAE(input_channels=4, latent_dim=128)
        >>> x = torch.randn(16, 4, 64, 64)  # Batch of RGBA sprites
        >>> reconstruction, mu, log_var = model(x)
        >>> loss = model.loss_function(reconstruction, x, mu, log_var)
    """
```

---

## 🔧 PyTorch Best Practices

### Model Definition

```python
class Encoder(nn.Module):
    """Encoder network for VAE."""
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dims: List[int] = [32, 64, 128, 256],
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        
        # Build encoder layers
        layers = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Latent space projection
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
```

### Device Handling

```python
# ✅ GOOD: Let user control device
def train(model: nn.Module, device: torch.device) -> None:
    model = model.to(device)
    
# ✅ GOOD: Auto-detect in CLI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ❌ BAD: Hardcoded device
model = model.cuda()  # Fails without GPU
```

### Reproducibility

```python
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/test_vae.py

import pytest
import torch
from spriteforge.models.vae import VAE


class TestVAE:
    """Tests for VAE model."""
    
    @pytest.fixture
    def model(self) -> VAE:
        """Create a VAE instance for testing."""
        return VAE(input_channels=4, latent_dim=32)
    
    @pytest.fixture
    def sample_batch(self) -> torch.Tensor:
        """Create a sample batch of sprites."""
        return torch.randn(4, 4, 64, 64)
    
    def test_forward_shape(self, model: VAE, sample_batch: torch.Tensor) -> None:
        """Test that forward pass returns correct shapes."""
        recon, mu, log_var = model(sample_batch)
        
        assert recon.shape == sample_batch.shape
        assert mu.shape == (4, 32)
        assert log_var.shape == (4, 32)
    
    def test_encode_decode_consistency(self, model: VAE) -> None:
        """Test that encode → decode pipeline works."""
        x = torch.randn(1, 4, 64, 64)
        mu, log_var = model.encode(x)
        z = model.reparameterize(mu, log_var)
        recon = model.decode(z)
        
        assert recon.shape == x.shape
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=spriteforge --cov-report=html
```

---

## 📊 Logging and Metrics

### Use Structured Logging

```python
import logging

logger = logging.getLogger(__name__)

# ✅ GOOD: Structured, informative
logger.info(
    "Training epoch completed",
    extra={
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
)

# ❌ BAD: Unstructured string formatting
print(f"Epoch {epoch}, loss: {loss}")
```

---

## 🚫 Anti-Patterns to Avoid

```python
# ❌ Global mutable state
GLOBAL_MODEL = None

# ❌ Catching all exceptions
try:
    train()
except:
    pass

# ❌ Magic numbers
x = x * 0.017453292519943295  # What is this?

# ✅ Use constants
DEGREES_TO_RADIANS = math.pi / 180
x = x * DEGREES_TO_RADIANS

# ❌ Nested conditionals > 3 levels
if a:
    if b:
        if c:
            if d:
                do_something()

# ✅ Early returns
if not a:
    return
if not b:
    return
if not c:
    return
if d:
    do_something()
```

---

## 📝 Commit Messages

Follow conventional commits:

```
feat: add convolutional encoder with residual connections
fix: correct KL divergence computation for batch processing
docs: add architecture diagram to README
test: add unit tests for reparameterization trick
refactor: extract loss functions to separate module
```

---

## 🔍 Code Review Checklist

Before submitting code:

- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] No hardcoded magic numbers
- [ ] Tests pass (`pytest tests/`)
- [ ] Code is formatted (`black .` and `isort .`)
- [ ] No linting errors (`ruff check .`)
- [ ] Documentation is updated if needed
