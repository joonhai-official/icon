# icon/contracts/__init__.py
"""Interface contracts (Section 16 of the specification).

A user of the framework implements two contracts:
- AdapterBase: wraps a system under measurement (three operations)
- DataLoaderBase: wraps a dataset for the measurement pipeline (three operations)
"""

from icon.contracts.adapter import AdapterBase
from icon.contracts.loader import DataLoaderBase

__all__ = ["AdapterBase", "DataLoaderBase"]
