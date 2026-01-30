"""Learning-rate schedules (protocol-locked).

v1.1 uses warmup + cosine decay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupCosineConfig:
    warmup_epochs: int
    total_epochs: int

    def lr_multiplier(self, epoch: int) -> float:
        """Return multiplicative factor for base LR at a given 0-indexed epoch."""
        if self.total_epochs <= 0:
            return 1.0
        if epoch < self.warmup_epochs:
            # Linear warmup from 0 to 1.
            return float(epoch + 1) / float(max(1, self.warmup_epochs))
        # Cosine decay from 1 to 0.
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
