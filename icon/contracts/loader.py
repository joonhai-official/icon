# icon/contracts/loader.py
"""DataLoaderBase — Section 16 of the framework specification.

A data loader wraps a dataset for the measurement pipeline. The contract
specifies three operations and nothing more. Users implement this abstract
base class for their own dataset.

See Section 16 of the specification for the contract's full text and rationale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataLoaderBase(ABC):
    """Wraps a dataset for the measurement pipeline.

    Subclasses must implement exactly three operations: train_batch,
    val_batch, and num_classes. The framework does not require any
    additional methods.

    Determinism is required: the same (batch_size, seed) pair must always
    return the same batch. The framework's reproducibility guarantee
    depends on this property.
    """

    @abstractmethod
    def train_batch(
        self,
        batch_size: int,
        seed: int,
    ) -> tuple[Any, Any]:
        """Return one batch from the training split.

        Parameters
        ----------
        batch_size : int
            The number of samples to return.
        seed : int
            A deterministic seed. The same (batch_size, seed) must always
            return the same batch on the same hardware.

        Returns
        -------
        tuple
            (X, Y) where X is the input batch and Y is the corresponding
            target batch. For unsupervised settings, Y may be None.
        """
        raise NotImplementedError

    @abstractmethod
    def val_batch(
        self,
        batch_size: int,
        seed: int,
    ) -> tuple[Any, Any]:
        """Return one batch from the validation split.

        Same determinism requirement as `train_batch`.

        Parameters
        ----------
        batch_size : int
        seed : int

        Returns
        -------
        tuple
            (X, Y) batch from the validation split.
        """
        raise NotImplementedError

    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of classes, or 0 for unsupervised settings.

        Returns
        -------
        int
            Number of classes for classification tasks. Return 0 to indicate
            that the loader is unsupervised (the framework will skip F_task
            computation for such loaders).
        """
        raise NotImplementedError
