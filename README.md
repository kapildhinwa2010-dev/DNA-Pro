# DNA-Pro

[![CI](https://github.com/kapildhinwa2010-dev/DNA-Pro/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Pro/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/DNA-Pro/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Pro/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Deepest member of the DNA weight-generator family: 3 hidden layers, inference-mode freezing.

## What it does

DNAPro maps a 2D coordinate grid through Fourier features and a 3-layer
SiLU MLP (~257K parameters by default) to scalar weight values, with
chunked generation for large shapes and an inference_mode that freezes
all parameters.

## Installation

    pip install torch numpy pytest

## Usage

    from dna_pro import create_dna_pro, DNAProWrapper

    dna = create_dna_pro(num_freqs=64, hidden_size=128)
    W = dna.generate_weights(1000, 1000)

    # Frozen-parameter inference
    dna = create_dna_pro(inference_mode=True)

    wrapper = DNAProWrapper()
    weights = wrapper.generate_for_model(model)

## API

- DNAPro(num_freqs=64, hidden_size=128, chunk_size=16384,
  inference_mode=True): generate_weights, generate_layer_weights,
  generate_model_weights, count_parameters, compression_ratio
- DNAProWrapper / DNAProConfig: generate_weights(shapes),
  generate_for_model(model)
- create_dna_pro(...) factory

## Honest results and limitations

- The ratio table (up to 39,130x) is parameter arithmetic: output
  elements divided by ~257K generator params. For models under ~650K
  parameters the generator is bigger than the model - no compression.
- "Production-ready" is not a claim we can back: no deployment
  benchmarks, no accuracy evaluation, and no reconstruction-quality
  measurement are included.
- An untrained generator outputs a smooth random field, not useful
  weights. Fitting it per matrix is the real cost and is not included.
- Generation of billion-parameter shapes is impractical on CPU
  (minutes to hours) and memory-bound everywhere.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that generated-weight
  reconstructions of trained models were not usable without per-layer
  fitting, which erases most claimed savings, and that the related
  carrier-mode recall failed about half of noise trials. The robust
  design kept in the toolkit is the Reed-Solomon slot lattice
  (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
