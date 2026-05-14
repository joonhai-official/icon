# icon/core/infonce.py
"""InfoNCE estimator (Appendix A.4).

Estimates mutual information from finite samples using a learnable
contrastive critic (van den Oord, Li, & Vinyals, 2018). The framework's
reference critic is a separable cosine critic with learnable scale:

    f(a, b) = exp(s · cos(g(a), h(b)))

where g, h are small MLPs and s is a learnable scalar.

Saturation ceiling: Î_NCE ≤ log B (Poole et al., 2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InfoNCEConfig:
    batch_size: int = 512
    n_steps: int = 2000
    critic_hidden: tuple[int, ...] = (256, 256)
    critic_proj_dim: int = 64
    learning_rate: float = 1e-3


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Module:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SeparableCosineCritic(nn.Module):
    """f(a, b) = exp(s · cos(g(a), h(b)))."""

    def __init__(self, dim_a: int, dim_b: int, config: InfoNCEConfig):
        super().__init__()
        self.g = _mlp(dim_a, config.critic_hidden, config.critic_proj_dim)
        self.h = _mlp(dim_b, config.critic_hidden, config.critic_proj_dim)
        # Learnable scale, initialized to log(1/τ) ≈ ln(20) like CLIP.
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Returns [N_a, N_b] cosine-similarity logits with learnable scale.
        g = F.normalize(self.g(a), dim=-1)
        h = F.normalize(self.h(b), dim=-1)
        return self.log_scale.exp() * g @ h.T


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    # Y may arrive as [N] integer labels. Treat as one-hot for the critic.
    if x.dim() == 1:
        # Float embedding via one-hot. Integer labels handled here.
        if x.dtype in (torch.int64, torch.int32, torch.long):
            n_classes = int(x.max().item()) + 1
            return F.one_hot(x.long(), num_classes=n_classes).float()
        return x.unsqueeze(-1).float()
    return x.float()


def estimate_mi(
    a: torch.Tensor,
    b: torch.Tensor,
    config: InfoNCEConfig,
    seed: int,
) -> float:
    """Return Î_NCE(A; B), in nats. Bounded above by log(config.batch_size)."""
    a = _to_2d(a)
    b = _to_2d(b)
    n = a.shape[0]

    torch.manual_seed(seed)
    critic = SeparableCosineCritic(a.shape[1], b.shape[1], config)
    optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    B = min(config.batch_size, n)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(config.n_steps):
        # Sample a contrastive batch of size B from the N pairs.
        idx = torch.randperm(n, generator=generator)[:B]
        a_batch = a[idx]
        b_batch = b[idx]

        logits = critic.logits(a_batch, b_batch)  # [B, B]
        # Positives are on the diagonal; negatives are off-diagonal.
        targets = torch.arange(B, device=logits.device)
        loss = F.cross_entropy(logits, targets)
        # Symmetric InfoNCE: contrast in both directions.
        loss = 0.5 * (loss + F.cross_entropy(logits.T, targets))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final estimate from the trained critic.
    # Î_NCE = log B - E[cross_entropy], converted from nats per cross-entropy.
    critic.eval()
    with torch.no_grad():
        idx = torch.randperm(n, generator=generator)[:B]
        logits = critic.logits(a[idx], b[idx])
        targets = torch.arange(B, device=logits.device)
        ce = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
        mi = math.log(B) - ce.item()
    return max(mi, 0.0)


def permutation_null(
    a: torch.Tensor,
    b: torch.Tensor,
    config: InfoNCEConfig,
    seed: int,
) -> float:
    """Run estimate_mi with shuffled pairs; returns the finite-sample bias floor."""
    a = _to_2d(a)
    b = _to_2d(b)
    n = b.shape[0]

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    return estimate_mi(a, b[perm], config, seed)
