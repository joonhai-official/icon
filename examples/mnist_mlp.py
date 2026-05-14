# examples/mnist_mlp.py
"""Minimal Icon usage example: MLP on MNIST.

Shows the full workflow:
    1. Wrap a PyTorch MLP in AdapterBase.
    2. Wrap an MNIST split in DataLoaderBase.
    3. Measure one layer.
    4. Inspect the resulting Profile.
    5. Save the manifest to disk for reproducibility.

Run:
    pip install -e ".[examples]"
    python examples/mnist_mlp.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

import icon


# -----------------------------------------------------------------------------
# 1. Define a small MLP. No training — we measure a fresh random initialization
#    to keep the example self-contained. Replace with your trained model.
# -----------------------------------------------------------------------------

class SmallMLP(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: tuple[int, ...] = (128, 64), n_classes: int = 10):
        super().__init__()
        self.h1 = nn.Linear(in_dim, hidden[0])
        self.h2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], n_classes)

    def forward(self, x):
        # Returned for completeness; not used by Icon directly.
        h1 = torch.relu(self.h1(x))
        h2 = torch.relu(self.h2(h1))
        return self.out(h2)


# -----------------------------------------------------------------------------
# 2. Adapter — wraps the MLP for measurement.
#
#    Tap convention: post-activation (after ReLU) at each named hidden layer.
#    Layers are returned in forward order; F_layer pairs adjacent entries.
# -----------------------------------------------------------------------------

class MLPAdapter(icon.AdapterBase):
    def __init__(self, model: SmallMLP):
        self.model = model
        self.model.eval()
        # Read dimensions from the model so the adapter stays in sync.
        self._dims = {"h1": model.h1.out_features, "h2": model.h2.out_features}

    def layer_names(self) -> list[str]:
        return ["h1", "h2"]

    def forward_with_taps(self, x: torch.Tensor, layer_names: list[str]) -> dict[str, torch.Tensor]:
        # x is [N, 784] (flattened MNIST). Flattening is the loader's job.
        with torch.no_grad():
            h1 = torch.relu(self.model.h1(x))
            h2 = torch.relu(self.model.h2(h1))
        result: dict[str, torch.Tensor] = {}
        if "h1" in layer_names: result["h1"] = h1
        if "h2" in layer_names: result["h2"] = h2
        return result

    def layer_dim(self, layer_name: str) -> int:
        return self._dims[layer_name]


# -----------------------------------------------------------------------------
# 3. Loader — wraps an MNIST split. To keep the example self-contained
#    (no torchvision dependency required at import time), the loader can
#    operate on either real MNIST or random fallback data.
# -----------------------------------------------------------------------------

class MNISTLoader(icon.DataLoaderBase):
    """Wraps MNIST. Falls back to random data if torchvision is unavailable."""

    def __init__(self):
        self._X, self._Y = self._load_or_fallback()

    def _load_or_fallback(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from torchvision import datasets, transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
            # Load all 10k validation samples into memory.
            X = torch.stack([ds[i][0].flatten() for i in range(len(ds))])
            Y = torch.tensor([ds[i][1] for i in range(len(ds))])
            print(f"  loaded MNIST validation: X={tuple(X.shape)}, Y={tuple(Y.shape)}")
            return X, Y
        except Exception as e:
            print(f"  torchvision unavailable ({type(e).__name__}); using random fallback.")
            torch.manual_seed(0)
            X = torch.randn(2048, 784)
            Y = torch.randint(0, 10, (2048,))
            return X, Y

    def train_batch(self, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        # For this example, train and val draw from the same pool with
        # different seeds. In a real workflow, use proper splits.
        return self._sample(batch_size, seed)

    def val_batch(self, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sample(batch_size, seed + 10_000_000)

    def _sample(self, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = self._X.shape[0]
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:batch_size]
        return self._X[idx], self._Y[idx]

    def num_classes(self) -> int:
        return 10


# -----------------------------------------------------------------------------
# 4. Run the measurement.
# -----------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Icon — MNIST/MLP example")
    print("=" * 60)

    # Build the model (no training in this example).
    print("\n[1] Building model and loader...")
    torch.manual_seed(0)
    model = SmallMLP()
    adapter = MLPAdapter(model)
    loader = MNISTLoader()
    print(f"  adapter layers: {adapter.layer_names()}")
    print(f"  layer dims: h1={adapter.layer_dim('h1')}, h2={adapter.layer_dim('h2')}")

    # Declare the PROTOCOL. Defaults are appropriate for most cases.
    # Reduced batch size and step count for fast example execution.
    protocol = icon.PROTOCOL()
    protocol.infonce.batch_size = 128
    protocol.infonce.n_steps = 500
    protocol.pool.sample_count = 1024
    print(f"  protocol_hash: {protocol.hash()[:16]}...")

    # Measure both layers.
    print("\n[2] Measuring all layers...")
    profiles = icon.measure_layers(adapter, loader, protocol=protocol, master_seed=42)

    print("\n[3] Results")
    print(f"  {'layer':<6} {'F_in':>8} {'F_task':>8} {'F_self':>8} "
          f"{'F_layer':>8} {'ρ':>6} {'η_t':>6} {'trust':<10}")
    for name, p in zip(adapter.layer_names(), profiles):
        fl = f"{p.f_layer:.3f}" if p.f_layer is not None else "  —"
        print(f"  {name:<6} {p.f_in:>8.4f} {p.f_task:>8.4f} {p.f_self:>8.4f} "
              f"{fl:>8} {p.rho:>6.3f} {p.eta_t:>6.3f} {p.trust_aggregate.value}")

    # Spatial view across layers.
    sv = icon.spatial(profiles)
    print(f"\n[4] Spatial view: {len(sv.profiles)} profiles")

    # Save manifest for reproducibility.
    import os
    os.makedirs("./manifests", exist_ok=True)
    out_path = "./manifests/mnist_mlp_h1.json"
    profiles[0].manifest.save(out_path)
    print(f"\n[5] Saved manifest to {out_path}")
    print(f"  PROTOCOL identity (SHA-256): {profiles[0].manifest.protocol_hash()[:16]}...")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
