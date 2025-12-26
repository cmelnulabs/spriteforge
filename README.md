# SpriteForge

A text-to-sprite GAN for generating 2D pixel art sprites from text descriptions.

## Installation

```bash
pip install -e .
```

## Usage

Train a model:
```bash
spriteforge train ./sprites/ --captions captions.json --epochs 100
```

Generate sprites from text:
```bash
spriteforge generate model_G.pt model_T.pt --text "red warrior with sword"
```

Generate multiple variations:
```bash
spriteforge generate model_G.pt model_T.pt --text "blue potion" --num 16
```

## License

GPL-3.0
