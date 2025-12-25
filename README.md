# SpriteForge

A Variational Autoencoder for generating 2D pixel art sprites.

## Installation

```bash
pip install -e .
```

## Usage

Train a model:
```bash
spriteforge train ./sprites/ --epochs 100
```

Generate sprites:
```bash
spriteforge generate model.pt --num 16
```

Interpolate between sprites:
```bash
spriteforge interpolate model.pt sprite1.png sprite2.png
```

## License

GPL-3.0
