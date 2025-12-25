# SpriteForge Roadmap

> A Variational Autoencoder (VAE) specialized in 2D pixel art sprite generation.

---

## 🎯 Project Vision

SpriteForge aims to be an educational and practical tool for generating 2D pixel art sprites using deep learning. The project prioritizes clean code, extensive documentation, and a modular architecture that allows experimentation with different VAE variants.

---

## Phase 1: Foundation (Core VAE)

**Goal**: Implement a working vanilla VAE that can encode and decode simple sprites.

### Features
- [ ] Project structure and development environment setup
- [ ] Data pipeline for loading and preprocessing sprite datasets
- [ ] Basic VAE architecture (Encoder → Latent Space → Decoder)
- [ ] Training loop with reconstruction loss + KL divergence
- [ ] Model checkpointing and basic logging
- [ ] Simple CLI for training and inference
- [ ] Unit tests for core components

### Technical Details
- Input: 32x32 or 64x64 RGB/RGBA sprites
- Latent space: 128-256 dimensions
- Framework: PyTorch
- Dataset: Start with simple shapes, then Kenney assets

### Deliverables
- `spriteforge/models/vae.py` - Core VAE implementation
- `spriteforge/data/` - Data loading utilities
- `spriteforge/train.py` - Training script
- `spriteforge/generate.py` - Generation script
- Documentation for all modules

---

## Phase 2: Enhanced Architecture

**Goal**: Improve generation quality with architectural enhancements.

### Features
- [ ] Convolutional VAE (Conv2d layers instead of fully connected)
- [ ] Residual connections in encoder/decoder
- [ ] Batch normalization and proper weight initialization
- [ ] Learning rate scheduling
- [ ] Early stopping based on validation loss
- [ ] TensorBoard/Weights & Biases integration
- [ ] Hyperparameter configuration via YAML files

### Technical Details
- Encoder: Conv2d → BatchNorm → LeakyReLU → ... → Flatten → μ, σ
- Decoder: Linear → Reshape → ConvTranspose2d → ... → Sigmoid
- Loss: BCE/MSE reconstruction + β-weighted KL divergence

### Deliverables
- `spriteforge/models/conv_vae.py` - Convolutional VAE
- `spriteforge/config/` - Configuration management
- Enhanced training metrics and visualization

---

## Phase 3: Conditional Generation

**Goal**: Generate sprites conditioned on labels/categories.

### Features
- [ ] Conditional VAE (CVAE) implementation
- [ ] Label embedding layer
- [ ] Support for multi-label conditioning
- [ ] Category-based generation (e.g., "character", "item", "tile")
- [ ] Style conditioning (e.g., "medieval", "sci-fi", "fantasy")
- [ ] Interpolation in latent space between categories

### Technical Details
- Condition injection: Concatenate label embedding with latent vector
- One-hot encoding or learned embeddings for categories
- Modified loss function for conditional generation

### Deliverables
- `spriteforge/models/cvae.py` - Conditional VAE
- Dataset annotation tools
- Conditional generation CLI/API

---

## Phase 4: Advanced Features

**Goal**: Production-ready features and quality improvements.

### Features
- [ ] β-VAE for disentangled representations
- [ ] VQ-VAE (Vector Quantized VAE) variant
- [ ] Sprite sheet generation (multiple frames)
- [ ] Animation sequence generation
- [ ] Palette control and color quantization
- [ ] Upscaling post-processor (optional super-resolution)
- [ ] REST API for generation service

### Technical Details
- VQ-VAE: Discrete latent space with codebook
- Animation: Temporal consistency in latent space
- Palette: Post-processing or palette-aware loss

### Deliverables
- `spriteforge/models/beta_vae.py`
- `spriteforge/models/vq_vae.py`
- `spriteforge/api/` - REST API service
- `spriteforge/postprocess/` - Post-processing utilities

---

## Phase 5: User Experience

**Goal**: Make SpriteForge accessible to non-ML users.

### Features
- [ ] Web UI for sprite generation (Gradio/Streamlit)
- [ ] Pre-trained model weights download
- [ ] Docker containerization
- [ ] Google Colab notebook
- [ ] Comprehensive tutorials and examples
- [ ] Plugin for game engines (Godot/Unity export)

### Deliverables
- `spriteforge/ui/` - Web interface
- `Dockerfile` and `docker-compose.yml`
- `notebooks/` - Jupyter notebooks for tutorials
- Pre-trained models on Hugging Face Hub

---

## Phase 6: Research Extensions (Optional)

**Goal**: Experimental features for research and exploration.

### Features
- [ ] Hierarchical VAE for multi-scale sprites
- [ ] Adversarial training (VAE-GAN hybrid)
- [ ] Diffusion-based decoder
- [ ] Few-shot learning for custom styles
- [ ] Latent space arithmetic (sprite1 + sprite2 = sprite3)

---

## 📊 Success Metrics

| Metric | Target |
|--------|--------|
| Reconstruction Loss (MSE) | < 0.01 |
| FID Score | < 50 |
| Generation Time | < 100ms per sprite |
| Model Size | < 50MB |
| Code Coverage | > 80% |

---

## 🗓️ Estimated Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1 | 2-3 weeks | 🚧 In Progress |
| Phase 2 | 2 weeks | ⏳ Planned |
| Phase 3 | 2 weeks | ⏳ Planned |
| Phase 4 | 3-4 weeks | ⏳ Planned |
| Phase 5 | 2 weeks | ⏳ Planned |
| Phase 6 | Ongoing | 💡 Research |

---

## 📚 References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling, 2013
- [β-VAE: Learning Basic Visual Concepts](https://openreview.net/forum?id=Sy2fzU9gl) - Higgins et al., 2017
- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937) - van den Oord et al., 2017
- [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)
