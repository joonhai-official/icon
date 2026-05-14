# examples/cifar10_resnet.py
"""Realistic Icon example: ResNet on CIFAR-10.

Demonstrates the framework's handling of:
    - A convolutional architecture (higher-rank activations).
    - The tap convention for reducing feature maps to [N, d_L].
    - Multiple-layer measurement across a non-trivial network.

The adapter uses PyTorch forward hooks to capture intermediate activations
and applies global average pooling to reduce feature maps to a per-channel
vector. This is one valid choice of tap convention; document yours.

Run (requires torchvision):
    pip install -e ".[examples]"
    python examples/cifar10_resnet.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

import icon


# -----------------------------------------------------------------------------
# Adapter for torchvision ResNet18.
#
# Tap convention:
#     - Attach forward hooks to the four residual blocks (layer1..layer4).
#     - Reduce feature maps [N, C, H, W] to [N, C] by global average pooling
#       over spatial dimensions. d_L = number of output channels.
#
# Document this convention in your adapter — it determines what "Z_L" means.
# -----------------------------------------------------------------------------

class ResNetAdapter(icon.AdapterBase):
    LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]

    def __init__(self, resnet: nn.Module):
        self.model = resnet
        self.model.eval()
        # Capture handles so we can remove hooks if needed.
        self._captures: dict[str, torch.Tensor] = {}
        self._hooks: list = []
        for name in self.LAYER_NAMES:
            module = getattr(self.model, name)
            handle = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(handle)

        # Probe dims by running one tiny forward.
        with torch.no_grad():
            self.model(torch.zeros(1, 3, 32, 32))
        self._dims = {name: self._reduce(self._captures[name]).shape[1]
                      for name in self.LAYER_NAMES}
        self._captures.clear()

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._captures[name] = output.detach()
        return hook

    @staticmethod
    def _reduce(feature_map: torch.Tensor) -> torch.Tensor:
        """Global average pool to [N, C]. Tap convention for this adapter."""
        if feature_map.dim() == 4:  # [N, C, H, W]
            return feature_map.mean(dim=(2, 3))
        return feature_map.flatten(start_dim=1)

    def layer_names(self) -> list[str]:
        return list(self.LAYER_NAMES)

    def forward_with_taps(self, x: torch.Tensor, layer_names: list[str]) -> dict[str, torch.Tensor]:
        self._captures.clear()
        with torch.no_grad():
            self.model(x)
        return {name: self._reduce(self._captures[name]) for name in layer_names}

    def layer_dim(self, layer_name: str) -> int:
        return self._dims[layer_name]


# -----------------------------------------------------------------------------
# Loader for CIFAR-10.
# -----------------------------------------------------------------------------

class CIFAR10Loader(icon.DataLoaderBase):
    def __init__(self):
        self._X, self._Y = self._load_or_fallback()

    def _load_or_fallback(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from torchvision import datasets, transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
            ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
            print(f"  loading CIFAR-10 validation ({len(ds)} samples)...")
            X = torch.stack([ds[i][0] for i in range(len(ds))])
            Y = torch.tensor([ds[i][1] for i in range(len(ds))])
            return X, Y
        except Exception as e:
            print(f"  torchvision unavailable ({type(e).__name__}); using random fallback.")
            torch.manual_seed(0)
            X = torch.randn(1024, 3, 32, 32)
            Y = torch.randint(0, 10, (1024,))
            return X, Y

    def train_batch(self, batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def main():
    print("=" * 60)
    print("Icon — CIFAR-10/ResNet18 example")
    print("=" * 60)

    print("\n[1] Building model...")
    try:
        from torchvision.models import resnet18
        # Random initialization. In real use, load trained weights.
        model = resnet18(num_classes=10)
    except ImportError:
        print("  torchvision required for this example.")
        return

    adapter = ResNetAdapter(model)
    loader = CIFAR10Loader()
    print(f"  adapter layers: {adapter.layer_names()}")
    for name in adapter.layer_names():
        print(f"    {name}: d_L = {adapter.layer_dim(name)}")

    # Smaller batch for example speed. Production: B=512.
    protocol = icon.PROTOCOL()
    protocol.infonce.batch_size = 128
    protocol.infonce.n_steps = 500
    protocol.pool.sample_count = 1024
    print(f"  protocol_hash: {protocol.hash()[:16]}...")

    print("\n[2] Measuring all layers (this takes a few minutes)...")
    profiles = icon.measure_layers(adapter, loader, protocol=protocol, master_seed=42)

    print("\n[3] Results")
    print(f"  {'layer':<8} {'F_in':>7} {'F_task':>7} {'F_self':>7} "
          f"{'F_layer':>8} {'ρ':>6} {'η_t':>6} {'trust':<10}")
    for name, p in zip(adapter.layer_names(), profiles):
        fl = f"{p.f_layer:.3f}" if p.f_layer is not None else "  —"
        print(f"  {name:<8} {p.f_in:>7.4f} {p.f_task:>7.4f} {p.f_self:>7.4f} "
              f"{fl:>8} {p.rho:>6.3f} {p.eta_t:>6.3f} {p.trust_aggregate.value}")

    # Save all manifests.
    import os
    os.makedirs("./manifests", exist_ok=True)
    for name, p in zip(adapter.layer_names(), profiles):
        p.manifest.save(f"./manifests/cifar10_resnet18_{name}.json")
    print(f"\n[4] Saved {len(profiles)} manifests to ./manifests/")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
