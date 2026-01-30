"""
ICON-Primitive Models: Frozen Stems (E0, S0)
"""
import torch
import torch.nn as nn
import torch.nn.init as init

DATASET_INPUT_DIMS = {"mnist": 784, "cifar10": 3072, "imagenet_subset": 12288}
DATASET_IN_CHANNELS = {"mnist": 1, "cifar10": 3, "imagenet_subset": 3}

def orthogonal_init(layer: nn.Module, seed: int):
    """Orthogonal init with unit-variance pre-activation scaling.

    Important: do not pollute the global RNG state.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))

        if isinstance(layer, nn.Linear):
            init.orthogonal_(layer.weight)
            # Scale so that pre-activation variance is ~1 under unit-variance input.
            layer.weight.data *= (1.0 / (layer.weight.shape[1] ** 0.5))
            if layer.bias is not None:
                layer.bias.data.zero_()
            return

        if isinstance(layer, nn.Conv2d):
            w = layer.weight.view(layer.weight.shape[0], -1)
            init.orthogonal_(w)
            layer.weight.data = w.view_as(layer.weight)
            fan_in = layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3]
            layer.weight.data *= (1.0 / (fan_in ** 0.5))
            if layer.bias is not None:
                layer.bias.data.zero_()
            return

        raise TypeError(f"Unsupported layer for orthogonal init: {type(layer)}")

class VectorStem(nn.Module):
    """E0: raw input → d-dim vector"""
    def __init__(self, input_dim: int, output_dim: int = 256, seed: int = 123):
        super().__init__()
        self.input_dim, self.output_dim, self.seed = input_dim, output_dim, seed
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        orthogonal_init(self.linear, seed)
    
    def forward(self, x):
        if x.dim() > 2: x = x.view(x.shape[0], -1)
        x = x.to(dtype=self.linear.weight.dtype)
        return self.linear(x)
    
    def get_stem_id(self): return f"E0_vector_{self.output_dim}d_seed{self.seed}"

class SpatialStem(nn.Module):
    """S0: raw image → feature map (C×H×W)"""
    def __init__(self, in_channels: int = 3, out_channels: int = 16, seed: int = 123):
        super().__init__()
        self.out_channels, self.seed = out_channels, seed
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.relu = nn.ReLU()
        orthogonal_init(self.conv1, seed)
        orthogonal_init(self.conv2, seed + 1)
    
    def forward(self, x): return self.relu(self.conv2(self.relu(self.conv1(x))))
    def get_stem_id(self): return f"S0_spatial_{self.out_channels}x8x8_seed{self.seed}"

def create_vector_stem(dataset: str, output_dim: int = 256, seed: int = 123) -> VectorStem:
    return VectorStem(DATASET_INPUT_DIMS[dataset], output_dim, seed)

def create_spatial_stem(dataset: str, seed: int = 123, out_channels: int = 16) -> SpatialStem:
    return SpatialStem(DATASET_IN_CHANNELS[dataset], out_channels, seed)
