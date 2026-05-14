# Icon

A framework for measuring information flow in representation-bearing systems.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-alpha%20(v0.1.0)-orange.svg)](CHANGELOG.md)

## What this is

Icon is the reference implementation of the framework specified in *Icon: A Framework for Measuring Information Flow* (Park, 2026). The framework defines five measurements taken at each internal representation of a system, a statistical check that tells you when to trust the measurements, and a settings discipline that makes measurements reproducible across laboratories.

This repository contains:

- The five measurements: `F_in`, `F_task`, `F_self`, `F_layer`, `ρ` (Section 2 of the spec)
- The Trust τ validity-classification system (Section 7)
- The nine-step measurement pipeline (Section 9)
- The four aggregation primitives: static, spatial, temporal, shift (Section 10)
- The PROTOCOL declaration and manifest serialization (Sections 12–14)
- The two interface contracts — `AdapterBase`, `DataLoaderBase` (Section 16)
- Example adapters for an MLP and ResNet18

The framework specification, the empirical companion (Park, 2026a), and the conceptual companion (Park, 2026b) are submitted together as three companion papers; this repository accompanies the framework specification.

## Installation

```bash
pip install git+https://github.com/joonhai-official/Icon.git
```

For development:

```bash
git clone https://github.com/joonhai-official/Icon.git
cd Icon
pip install -e ".[dev]"
```

For examples (adds torchvision):

```bash
pip install -e ".[examples]"
```

## Quick start

```python
import icon

# 1. Wrap your system in the AdapterBase contract.
class MyAdapter(icon.AdapterBase):
    def layer_names(self):
        return ["conv1", "conv2", "fc1"]

    def forward_with_taps(self, x, names):
        # Return {name: tensor of shape [N, d_L]} for each requested layer.
        # The reduction from higher-rank activations to [N, d_L] is your choice;
        # document it. (See examples/cifar10_resnet.py for one convention.)
        ...

    def layer_dim(self, name):
        ...

# 2. Wrap your data in the DataLoaderBase contract.
class MyLoader(icon.DataLoaderBase):
    def train_batch(self, batch_size, seed): ...
    def val_batch(self, batch_size, seed): ...
    def num_classes(self): return 10

# 3. Declare a PROTOCOL. Defaults follow Section 12's reference values.
protocol = icon.PROTOCOL()

# 4. Measure.
profile = icon.measure(
    adapter=MyAdapter(model),
    loader=MyLoader(data),
    layer_name="conv2",
    protocol=protocol,
    master_seed=42,
)

# 5. Inspect.
print(profile.components)        # (F_in, F_task, F_self, F_layer, ρ)
print(profile.eta_t)             # canonical ratio F_task / F_in
print(profile.trust)             # per-component Trust τ
print(profile.trust_aggregate)   # profile-level Trust τ
profile.manifest.save("./run.json")  # full settings + seeds + environment
```

## Examples

Two examples are included:

- **[`examples/mnist_mlp.py`](examples/mnist_mlp.py)** — minimal MLP on MNIST. The shortest path from spec to working measurement.
- **[`examples/cifar10_resnet.py`](examples/cifar10_resnet.py)** — torchvision ResNet18 on CIFAR-10. Demonstrates a forward-hook adapter and the tap convention for convolutional layers (global average pooling to reduce feature maps to `[N, d_L]`).

Run:

```bash
python examples/mnist_mlp.py
python examples/cifar10_resnet.py
```

Both fall back to random data if torchvision is unavailable, so they always demonstrate the pipeline even without dataset downloads.

## Tests

```bash
pytest                  # all tests
pytest -m "not slow"    # quick subset (no InfoNCE training)
```

The test suite covers:

- Unit tests for each measurement (noise injection, RMS, Trust τ, ρ, η_t, PROTOCOL hashing, manifest schema).
- Property tests for invariants: η_t's `d_L` cancellation at machine precision, InfoNCE's `log B` ceiling.
- Integration tests for the full pipeline including determinism (same master seed → identical Profile).

## Specification mapping

This repository follows the framework specification section-by-section. The short version:

| Specification section | Module |
|---|---|
| §1–2 (The five measurements) | `icon/measurements/` |
| §4 (Why noise is required) | `icon/core/noise.py` |
| §7 (Trust τ) | `icon/core/trust.py` |
| §9 (Measurement pipeline) | `icon/pipeline/measure.py` |
| §10 (Aggregations) | `icon/pipeline/aggregate.py` |
| §11 (Cross-system comparison) | `icon/pipeline/compare.py` |
| §12 (PROTOCOL) | `icon/io/protocol.py` |
| §13 (Manifest schema) | `icon/io/manifest.py` |
| §14 (Versioning) | `icon/io/manifest.py` |
| §16 (Interface contracts) | `icon/contracts/` |
| §17 (Public surface) | `icon/__init__.py` |
| §A.4 (InfoNCE estimator) | `icon/core/infonce.py` |
| §A.7 (η_t identity) | `icon/measurements/eta_t.py` |

For the complete mapping including test coverage, see [`docs/spec_mapping.md`](docs/spec_mapping.md).

## Status

This is an **alpha release (v0.1.0)** alongside the framework specification's initial submission. The reference implementation covers all five components defined in the specification and verifies the key structural properties (determinism, saturation ceiling, η_t identity at machine precision).

Cross-architectural validation of `F_self`, `F_layer`, and `ρ` is described in the specification as follow-up work — the implementation itself is structurally complete, and these components are well-defined operationally and derived structurally in Section 3.

## Citation

This repository accompanies three companion papers submitted together. Citations:

```bibtex
@article{park2026icon,
  title={Icon: A Framework for Measuring Information Flow},
  author={Park, JoonHa},
  year={2026},
  note={Companion submission}
}

@article{park2026capacity,
  title={Information Capacity in Neural Networks: An Empirical Study of $\kappa$ Scaling},
  author={Park, JoonHa},
  year={2026},
  note={Companion submission}
}

@article{park2026substrates,
  title={Information Flow and Self-Reference Across Substrates: On Self-Reference, Substrate, and the Need for Measurement},
  author={Park, JoonHa},
  year={2026},
  note={Companion submission}
}
```

After arXiv submission, `note` will be replaced with the corresponding arXiv IDs.

## Contributing

The framework is designed to grow by community contribution. Per Section 20 of the specification:

- Adapter packages for specific model frameworks
- Data loader packages for specific dataset formats
- Domain extension packages mapping the framework to specific fields
- Lookup table contributions — published `(configuration → profile)` measurements

Open an issue to discuss before substantial changes. See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidance.

## License

Apache 2.0. See [LICENSE](LICENSE).
