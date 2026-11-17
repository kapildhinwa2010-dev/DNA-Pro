# DNA-Pro

DNA-Pro is a small PyTorch module that maps two-dimensional coordinates through Fourier features and a multilayer perceptron to generate tensors shaped like neural-network weight matrices. **Status: prototype.** The repository contains a generator implementation and tests, but no training or fitting procedure that learns useful weights from an existing model.

## Implemented functionality

- `DNAPro` creates a fixed random Fourier projection and a four-linear-layer MLP with three SiLU hidden layers. Its `generate_weights(out_dim, in_dim)` method returns a tensor with shape `(out_dim, in_dim)`.
- Large matrices are processed in row chunks according to `chunk_size`.
- `generate_layer_weights` generates tensors for a mapping of layer names to `(out_dim, in_dim)` shapes. `generate_model_weights` finds `torch.nn.Linear` modules and returns generated weight tensors and, where present, bias tensors shaped `(1, out_features)`.
- `DNAProConfig`, `DNAProWrapper`, and `create_dna_pro` provide configuration, wrapper, and factory interfaces.
- Parameters are frozen by default (`inference_mode=True`); construct with `inference_mode=False` to leave them trainable. The repository does not provide a training loop or learned checkpoint.
- `count_parameters` counts the generator's trainable parameters. `compression_ratio` reports a parameter-count ratio only; it does not measure storage savings for a fitted model or preserve model accuracy.

The generated values come from the generator's initialization unless a caller trains the network. They are not recovered or pretrained weights, and the repository does not provide a procedure for fitting them to a model or applying the returned tensors back into one.

## Requirements and running from source

The project metadata specifies Python 3.10 or newer, PyTorch 2.0 or newer, and NumPy 1.24 or newer. From a checkout, install the listed dependencies and run the included demonstration:

```bash
git clone https://github.com/kapildhinwa2010-dev/DNA-Pro.git
cd DNA-Pro
python -m pip install -r requirements.txt
python dna_pro.py
```

The demo creates sample tensors and prints their shapes. To call the API from the repository root:

```python
from dna_pro import create_dna_pro

generator = create_dna_pro(num_freqs=8, hidden_size=16)
weights = generator.generate_weights(out_dim=4, in_dim=3)
print(weights.shape)  # torch.Size([4, 3])
```

`generate_model_weights` only handles `torch.nn.Linear` layers; it returns generated tensors rather than modifying the supplied model.

## Tests and CI

At commit `736d62ed74107b3a6e630ab30a00ead0c1a5e565`, the test suite was run locally with Python 3.12.3 using `python3 -m pytest -q`: **23 passed**. The `python3 dna_pro.py` demonstration also ran successfully. These are local results; remote GitHub Actions run statuses were not checked. The CI workflow defines test jobs for Python 3.10, 3.11, and 3.12, and the repository also defines a CodeQL workflow.

To run tests from a checkout:

```bash
python -m pytest -q
```

## Limitations

- There is no data-fitting/training pipeline, pretrained generator, reconstruction-quality or accuracy evaluation, or evidence that generated tensors can replace trained model weights.
- The reported compression ratio is arithmetic based on parameter counts, not a benchmark or demonstrated end-to-end compression result.
- The generator handles `nn.Linear` layers only in its model traversal helper; it does not load the generated tensors into a model.
- A wheel built from the current `pyproject.toml` did not contain `dna_pro.py` (it contained distribution metadata only). The source-tree commands above work, but installing the built wheel as an importable package is not currently verified and should not be relied on until packaging is corrected.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for its terms.
