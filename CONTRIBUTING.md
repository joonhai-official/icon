# Contributing to Icon

Thanks for your interest. The framework is designed to grow by community contribution; this document describes how.

## What kinds of contributions

Per Section 20 of the framework specification, four kinds of contributions are welcomed:

1. **Adapter packages** for specific model frameworks (PyTorch CNN, JAX transformer, etc.). These implement `AdapterBase` for a class of systems.
2. **Data loader packages** for specific dataset formats. These implement `DataLoaderBase`.
3. **Domain extension packages** mapping the framework to specific fields (neuroscience recordings, signal-processing pipelines, etc.). These typically extend the PROTOCOL with domain-specific reference values.
4. **Lookup table contributions** — published `(configuration → profile)` measurements that enrich the framework's empirical anchors.

Plus the obvious: bug reports, documentation improvements, test additions, replication studies, and falsification attempts.

## How to contribute

### Bug reports

Open an issue. Include:
- A minimal reproducible example
- The Icon version (`import icon; icon.__version__`)
- Python and PyTorch versions
- The full traceback

### Pull requests

For small fixes (typos, docstrings, small bugs), open a PR directly.

For substantial changes (new features, behavior changes), please open an issue first to discuss. The framework's stability depends on the kernel staying small — additions to the public surface or the interface contracts (`AdapterBase`, `DataLoaderBase`) go through review.

### Adapter / loader packages

These typically live in their own repositories, not in this one. To make your package discoverable, please:
- Use the prefix `icon-` (e.g., `icon-pytorch-cnn`, `icon-jax-transformer`)
- Document your tap convention in the README (where in the underlying system you attach taps, how you reduce higher-rank activations to `[N, d_L]`)
- Release under a license compatible with Apache 2.0
- Open an issue here to link your package so it can be referenced

### Domain extensions

Same as adapters: their own repositories, `icon-` prefix, documented mapping from the framework's measurements to the domain's quantities, with explicit reference to specification section numbers.

## Code style

- Python 3.10+
- Format with `ruff format`
- Lint with `ruff check`
- Type hints required for public surface (`icon.__init__.py`)
- Docstrings: NumPy style
- Tests required for new behavior

## Tests

Run:
```bash
pip install -e ".[dev]"
pytest
```

Tests are organized:
- `tests/unit/` — single-module behavior
- `tests/property/` — invariants (e.g., InfoNCE ≤ log B, η_t's d_L cancellation)
- `tests/integration/` — end-to-end pipeline behavior

## Specification authority

When implementation behavior conflicts with the specification, the specification wins. If you believe the specification itself is wrong, please open an issue describing the conflict; specification changes follow the versioning rules in Section 14 (MAJOR / MINOR / PATCH semantic versioning, 12-month advance notice for MAJOR changes).

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
