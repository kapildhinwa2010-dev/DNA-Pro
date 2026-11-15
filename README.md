# DNA-Pro

[![CI](https://github.com/kapildhinwa2010-dev/DNA-Pro/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Pro/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/DNA-Pro/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Pro/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**DNA-Pro** is a production-ready implementation of DNA (Distributed Neural Architecture) for generating neural network weights with extreme compression capabilities. It's optimized for inference and production deployment.

### Key Features

- ✅ **Extreme Compression**: Generate 1B+ parameters from ~257K parameters
- ✅ **Production-Ready**: Optimized for inference and deployment
- ✅ **Memory Efficient**: Chunked processing for large models
- ✅ **GPU Accelerated**: Full CUDA support
- ✅ **PyTorch Integration**: Seamless integration with PyTorch models
- ✅ **Inference Mode**: Frozen parameters for stable inference

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/DNA-Pro.git
cd DNA-Pro
pip install -e .
```

## Usage

### Basic Weight Generation

```python
from dna_pro import DNAPro, create_dna_pro

# Create DNA-Pro instance
dna = create_dna_pro(num_freqs=64, hidden_size=128)

# Generate a weight matrix
W = dna.generate_weights(1000, 1000)
print(f"Generated matrix: {W.shape}")
```

### Generate for Multiple Layers

```python
# Generate weights for multiple layers
layer_config = {
    'fc1': (256, 128),
    'fc2': (128, 256),
    'fc3': (64, 128),
}
weights = dna.generate_layer_weights(layer_config)

for name, W in weights.items():
    print(f"{name}: {W.shape}")
```

### Generate for Entire Model

```python
import torch.nn as nn
from dna_pro import DNAProWrapper

# Create a model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

# Generate weights for the model
wrapper = DNAProWrapper()
weights = wrapper.generate_for_model(MyModel())

# Load into model
model = MyModel()
model.load_state_dict(weights)
```

### Inference Mode

```python
# Create in inference mode (frozen parameters)
dna = create_dna_pro(inference_mode=True)

# All parameters are frozen
for param in dna.parameters():
    assert not param.requires_grad
```

## Architecture

### Fourier Feature Generation

DNA-Pro uses random Fourier features to map 2D coordinate grids into a high-dimensional space:

```
Coordinates → Fourier Projection → Features → DNA Network → Weights
```

### DNA Network

The DNA network consists of:
- Input: Fourier features (num_freqs * 2 dimensions)
- Hidden layers: 3 layers with SiLU activation (128 hidden size by default)
- Output: Scalar weight value
- Trainable scale parameter

### Memory Efficiency

For large weight matrices, DNA-Pro uses chunked processing:
- Configurable chunk size (default: 16384 elements)
- Automatic chunking for matrices larger than chunk size
- Seamless reconstruction of full matrix

### Xavier Scaling

All generated weights are automatically scaled using Xavier initialization to maintain proper variance.

## Compression Ratios

| Output Size | DNA Parameters | Compression Ratio |
|-------------|----------------|-------------------|
| 100K        | ~257K          | ~0.4x             |
| 1M          | ~257K          | ~3.9x             |
| 10M         | ~257K          | ~39x              |
| 100M        | ~257K          | ~391x             |
| 1B          | ~257K          | ~3913x            |
| 10B         | ~257K          | ~39,130x          |

*Note: For models smaller than ~650K parameters, DNA-Pro may not compress. The compression ratio increases with model size.*

## API Reference

### DNAPro

```python
DNAPro(
    num_freqs: int = 64,
    hidden_size: int = 128,
    chunk_size: int = 16384,
    inference_mode: bool = True
)
```

**Methods:**
- `generate_weights(out_dim: int, in_dim: int) -> Tensor`: Generate a weight matrix
- `generate_layer_weights(layer_config: Dict) -> Dict`: Generate weights for multiple layers
- `generate_model_weights(model: nn.Module) -> Dict`: Generate weights for a model
- `count_parameters() -> int`: Count DNA network parameters
- `compression_ratio(original_params: int) -> float`: Calculate compression ratio

### DNAProWrapper

High-level wrapper for easier use:

```python
wrapper = DNAProWrapper(config=DNAProConfig())
```

**Methods:**
- `generate_weights(shapes: Dict) -> Dict`: Generate weights for layers
- `generate_for_model(model: nn.Module) -> Dict`: Generate weights for model

### create_dna_pro()

Factory function for quick DNA-Pro creation:

```python
dna = create_dna_pro(
    num_freqs=64,
    hidden_size=128,
    inference_mode=True,
    device='cuda'
)
```

## Production Deployment

### Export for Inference

```python
# Create DNA-Pro
dna = create_dna_pro(inference_mode=True)

# Generate weights
weights = dna.generate_model_weights(your_model)

# Save weights
torch.save(weights, 'dna_weights.pt')
```

### Load for Inference

```python
# Load weights
weights = torch.load('dna_weights.pt')

# Create model and load weights
model = YourModel()
model.load_state_dict(weights)
model.eval()

# Run inference
with torch.no_grad():
    output = model(input)
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Fourier feature networks
- Neural network compression
- Implicit neural representations
- Parameter-efficient learning
- Production machine learning
