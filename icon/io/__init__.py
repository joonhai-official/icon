# icon/io/__init__.py
"""Serialization and reproducibility infrastructure (Part 4 of the specification).

This package contains the PROTOCOL declaration (settings discipline) and
the manifest schema (serialized record of one measurement). Together they
make measurements reproducible across laboratories.
"""

from icon.io.protocol import PROTOCOL, protocols_match
from icon.io.manifest import Manifest, check_manifest_compat

__all__ = ["PROTOCOL", "protocols_match", "Manifest", "check_manifest_compat"]
