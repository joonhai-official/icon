# icon/contracts/adapter.py
"""AdapterBase — Section 16 of the framework specification.

An adapter wraps a system under measurement. The contract specifies three
operations and nothing more. Users implement this abstract base class for
their own model (PyTorch, JAX, NumPy, or any other framework).

See Section 16 of the specification for the contract's full text and rationale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AdapterBase(ABC):
    """Wraps a system under measurement.

    Subclasses must implement exactly three operations: layer_names,
    forward_with_taps, and layer_dim. The framework does not require
    any additional methods, and additional methods are not used by
    the measurement pipeline.

    The tap convention — where in the underlying system the adapter
    attaches taps, and how higher-rank activations are reduced to
    [N, d_L] — is the adapter's choice and must be documented in
    the subclass's docstring.
    """

    @abstractmethod
    def layer_names(self) -> list[str]:
        """Return the ordered list of layer identifiers available for measurement.

        The order defines the spatial ordering used by Spatial aggregation.
        Adjacent identifiers define the layer pairs used by F_layer.

        Returns
        -------
        list of str
            The ordered identifiers of all layers exposed by this adapter.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_with_taps(
        self,
        x: Any,
        layer_names: list[str],
    ) -> dict[str, Any]:
        """Run a forward pass and return activations at the requested layers.

        Parameters
        ----------
        x : array-like
            An input batch. The shape and type are the adapter's choice and
            must match what the wrapped system expects.
        layer_names : list of str
            The layer identifiers at which to expose activations. Each must
            be a member of `self.layer_names()`.

        Returns
        -------
        dict
            Maps each requested layer identifier to its activation tensor of
            shape [N, d_L]. Higher-rank activations (for example, feature maps
            from a convolutional layer) must be reduced to [N, d_L] by the
            adapter; the reduction convention is the adapter's choice and
            must be documented.
        """
        raise NotImplementedError

    @abstractmethod
    def layer_dim(self, layer_name: str) -> int:
        """Return the post-reduction dimension of the named layer.

        Parameters
        ----------
        layer_name : str
            A layer identifier from `self.layer_names()`.

        Returns
        -------
        int
            The dimension d_L of the activation tensor returned by
            `forward_with_taps` for this layer.
        """
        raise NotImplementedError
