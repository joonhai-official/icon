"""MI estimators for Icon_primitive.

Implements three protocol-locked estimators:
- InfoNCE (Primary): contrastive lower bound.
- MINE (Secondary): neural lower bound.
- KSG (Tertiary): nonparametric kNN estimator (with fixed projection to 32-d).

API:
    est = get_estimator(name, config, device)
    est.fit(X_train, Z_train)  # for neural estimators
    out = est.estimate(X_test, Z_test)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import digamma
from sklearn.neighbors import KDTree


class BaseMIEstimator(ABC):
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device

    def fit(self, X: torch.Tensor, Z: torch.Tensor) -> None:
        """Optional training step (default: no-op)."""
        return None

    @abstractmethod
    def estimate(self, X: torch.Tensor, Z: torch.Tensor) -> Dict[str, float]:
        """Estimate MI on the provided samples."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return config for receipts."""


def _act(name: str) -> nn.Module:
    return nn.ReLU() if name.lower() == "relu" else nn.GELU()


class EncoderMLP(nn.Module):
    """MLP encoder used by InfoNCE."""

    def __init__(self, d_in: int, hidden_dims: Tuple[int, ...], d_out: int, activation: str = "relu"):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(_act(activation))
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticMLP(nn.Module):
    """Joint critic used by MINE (concat(x,z) -> scalar)."""

    def __init__(self, d_x: int, d_z: int, hidden_dims: Tuple[int, ...], activation: str = "relu"):
        super().__init__()
        layers = []
        prev = d_x + d_z
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(_act(activation))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=-1)).squeeze(-1)


class InfoNCE(BaseMIEstimator):
    """InfoNCE estimator (contrastive bound).

    Protocol-locked defaults:
      steps=2000, batch_size=512, lr=1e-4, temperature=0.1

    Implementation note:
      We use two MLP encoders and a dot-product score matrix for efficiency.
    """

    DEFAULT_CONFIG = {
        "hidden_dims": (512, 256),
        "encoder_out_dim": 128,
        "steps": 2000,
        "batch_size": 512,
        "lr": 1e-4,
        "temperature": 0.1,
        "saturation_margin": 0.1,
        "activation": "relu",
    }

    def __init__(self, config: Dict[str, Any], device: torch.device):
        merged = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged, device)
        self._gx: Optional[EncoderMLP] = None
        self._hz: Optional[EncoderMLP] = None

    def fit(self, X: torch.Tensor, Z: torch.Tensor) -> None:
        X = X.to(self.device)
        Z = Z.to(self.device)
        n, d_x = X.shape
        n2, d_z = Z.shape
        if n2 != n:
            raise ValueError("X and Z must have the same number of samples")

        out_dim = int(self.config["encoder_out_dim"])
        hidden = tuple(int(h) for h in self.config["hidden_dims"])
        act = str(self.config.get("activation", "relu"))
        self._gx = EncoderMLP(d_x, hidden, out_dim, activation=act).to(self.device)
        self._hz = EncoderMLP(d_z, hidden, out_dim, activation=act).to(self.device)

        opt = torch.optim.Adam(list(self._gx.parameters()) + list(self._hz.parameters()), lr=float(self.config["lr"]))
        batch_size = int(min(self.config["batch_size"], n))
        temp = float(self.config["temperature"])

        self._gx.train()
        self._hz.train()

        for _ in range(int(self.config["steps"])):
            idx = torch.randperm(n, device=self.device)[:batch_size]
            x_b = X[idx]
            z_b = Z[idx]

            gx = F.normalize(self._gx(x_b.float()), dim=-1)
            hz = F.normalize(self._hz(z_b.float()), dim=-1)
            scores = (gx @ hz.T) / temp
            # InfoNCE loss: -diag + logsumexp(row)
            loss = (-torch.diagonal(scores) + torch.logsumexp(scores, dim=1)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def estimate(self, X: torch.Tensor, Z: torch.Tensor) -> Dict[str, float]:
        if self._gx is None or self._hz is None:
            raise RuntimeError("InfoNCE.estimate called before fit")

        X = X.to(self.device)
        Z = Z.to(self.device)
        n = X.shape[0]
        batch_size = int(min(self.config["batch_size"], n))
        temp = float(self.config["temperature"])

        self._gx.eval()
        self._hz.eval()
        losses = []
        with torch.no_grad():
            # Evaluate in batches to cap memory
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                x_b = X[idx]
                z_b = Z[idx]
                gx = F.normalize(self._gx(x_b.float()), dim=-1)
                hz = F.normalize(self._hz(z_b.float()), dim=-1)
                scores = (gx @ hz.T) / temp
                loss = (-torch.diagonal(scores) + torch.logsumexp(scores, dim=1)).mean()
                losses.append(float(loss.item()))

        final_loss = float(np.mean(losses))
        mi = float(np.log(batch_size) - final_loss)
        sat_margin = float(np.log(batch_size) - mi)
        return {"mi": mi, "loss": final_loss, "log_batch": float(np.log(batch_size)), "saturation_margin": sat_margin}

    def get_config(self) -> Dict[str, Any]:
        return {"name": "infonce", "config": dict(self.config)}


class MINE(BaseMIEstimator):
    """MINE estimator (neural lower bound)."""

    DEFAULT_CONFIG = {
        "hidden_dims": (512, 256),
        "steps": 2000,
        "batch_size": 512,
        "lr": 1e-4,
        "ema_decay": 0.99,
        "activation": "relu",
    }

    def __init__(self, config: Dict[str, Any], device: torch.device):
        merged = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged, device)
        self._critic: Optional[CriticMLP] = None

    def fit(self, X: torch.Tensor, Z: torch.Tensor) -> None:
        X = X.to(self.device)
        Z = Z.to(self.device)
        n, d_x = X.shape
        n2, d_z = Z.shape
        if n2 != n:
            raise ValueError("X and Z must have the same number of samples")
        hidden = tuple(int(h) for h in self.config["hidden_dims"])
        act = str(self.config.get("activation", "relu"))
        self._critic = CriticMLP(d_x, d_z, hidden, activation=act).to(self.device)

        opt = torch.optim.Adam(self._critic.parameters(), lr=float(self.config["lr"]))
        batch_size = int(min(self.config["batch_size"], n))
        ema_decay = float(self.config["ema_decay"])
        ema: Optional[torch.Tensor] = None

        self._critic.train()
        for _ in range(int(self.config["steps"])):
            idx = torch.randperm(n, device=self.device)[:batch_size]
            x_j = X[idx]
            z_j = Z[idx]
            z_m = Z[torch.randperm(n, device=self.device)[:batch_size]]

            t_joint = self._critic(x_j, z_j)
            t_marg = self._critic(x_j, z_m)

            joint_term = t_joint.mean()
            # IMPORTANT: EMA must be detached.
            # Otherwise ema carries a computation graph across iterations and can
            # cause "backward through graph a second time" errors (and a memory leak).
            exp_marg = torch.exp(t_marg).mean()
            exp_marg_detached = exp_marg.detach()
            ema = exp_marg_detached if ema is None else ema_decay * ema + (1.0 - ema_decay) * exp_marg_detached

            mi_est = joint_term - torch.log(ema + 1e-10)
            loss = -mi_est

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def estimate(self, X: torch.Tensor, Z: torch.Tensor) -> Dict[str, float]:
        if self._critic is None:
            raise RuntimeError("MINE.estimate called before fit")
        X = X.to(self.device)
        Z = Z.to(self.device)
        n = X.shape[0]
        batch_size = int(min(self.config["batch_size"], n))
        ema_decay = float(self.config["ema_decay"])
        ema: Optional[torch.Tensor] = None
        vals = []
        self._critic.eval()
        with torch.no_grad():
            for _ in range(64):
                idx = torch.randperm(n, device=self.device)[:batch_size]
                x_j = X[idx]
                z_j = Z[idx]
                z_m = Z[torch.randperm(n, device=self.device)[:batch_size]]
                t_joint = self._critic(x_j, z_j)
                t_marg = self._critic(x_j, z_m)
                joint_term = t_joint.mean()
                exp_marg = torch.exp(t_marg).mean()
                ema = exp_marg if ema is None else ema_decay * ema + (1.0 - ema_decay) * exp_marg
                vals.append(float((joint_term - torch.log(ema + 1e-10)).item()))
        return {"mi": float(np.mean(vals))}

    def get_config(self) -> Dict[str, Any]:
        return {"name": "mine", "config": dict(self.config)}


class KSG(BaseMIEstimator):
    """KSG estimator with fixed orthogonal projection and KDTree counting."""

    DEFAULT_CONFIG = {
        "projection_dim": 32,
        "projection_seed": 42,
        "k": 5,
        "metric": "chebyshev",
    }

    def __init__(self, config: Dict[str, Any], device: torch.device):
        merged = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged, device)

    def _orth_proj(self, d_in: int, d_out: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        a = rng.standard_normal(size=(d_in, d_out)).astype(np.float64)
        q, _ = np.linalg.qr(a)
        return q[:, : min(d_out, q.shape[1])]

    def _project(self, arr: np.ndarray, d_out: int, seed: int) -> np.ndarray:
        if arr.shape[1] <= d_out:
            return arr
        p = self._orth_proj(arr.shape[1], d_out, seed)
        return arr @ p

    def estimate(self, X: torch.Tensor, Z: torch.Tensor) -> Dict[str, float]:
        X_np = X.detach().cpu().numpy().astype(np.float64)
        Z_np = Z.detach().cpu().numpy().astype(np.float64)
        n, d_x = X_np.shape
        _, d_z = Z_np.shape

        proj_dim = int(self.config["projection_dim"])
        proj_seed = int(self.config["projection_seed"])
        k = int(self.config["k"])

        Xp = self._project(X_np, proj_dim, proj_seed)
        Zp = self._project(Z_np, proj_dim, proj_seed + 1)
        XZ = np.concatenate([Xp, Zp], axis=1)

        # Joint distances to k-th neighbor (Chebyshev / max-norm).
        tree_joint = KDTree(XZ, metric=str(self.config.get("metric", "chebyshev")))
        dists, _ = tree_joint.query(XZ, k=k + 1)
        eps = dists[:, k]
        # Use slightly smaller radius to match "< eps" convention.
        radius = np.nextafter(eps, 0)

        tree_x = KDTree(Xp, metric=str(self.config.get("metric", "chebyshev")))
        tree_z = KDTree(Zp, metric=str(self.config.get("metric", "chebyshev")))

        n_x = tree_x.query_radius(Xp, r=radius, count_only=True) - 1
        n_z = tree_z.query_radius(Zp, r=radius, count_only=True) - 1
        n_x = np.maximum(0, n_x)
        n_z = np.maximum(0, n_z)

        mi = float(digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_z + 1)) + digamma(n))
        return {"mi": max(0.0, mi), "projection_dim": proj_dim, "k": k}

    def get_config(self) -> Dict[str, Any]:
        return {"name": "ksg", "config": dict(self.config)}


def get_estimator(name: str, config: Dict[str, Any], device: torch.device) -> BaseMIEstimator:
    name_l = name.lower()
    if name_l in {"infonce", "primary"}:
        return InfoNCE(config, device)
    if name_l in {"mine", "secondary"}:
        return MINE(config, device)
    if name_l in {"ksg", "ksg32", "tertiary"}:
        return KSG(config, device)
    raise ValueError(f"Unknown MI estimator: {name}")
