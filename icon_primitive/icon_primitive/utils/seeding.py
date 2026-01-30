"""Seed Management for Reproducibility"""
import random
from typing import Dict, Optional

import numpy as np
import torch

from .determinism import apply_determinism

def set_seed(model_seed: int = 0, data_seed: int = 0, mi_seed: int = 0, determinism_cfg: Optional[Dict] = None):
    """Set model/data/MI seeds and (optionally) apply determinism flags."""
    torch.manual_seed(int(model_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(model_seed))

    np.random.seed(int(data_seed))
    random.seed(int(data_seed))

    if determinism_cfg is not None:
        apply_determinism(determinism_cfg)

def set_mi_seed(mi_seed: int = 0):
    """MI estimator용 seed 설정"""
    torch.manual_seed(int(mi_seed))
    np.random.seed(int(mi_seed))

def get_seed_config(model_seed: int, data_seed: int = 0, mi_seed: int = 0) -> Dict:
    return {"model_seed": model_seed, "data_seed": data_seed, "mi_seed": mi_seed}
