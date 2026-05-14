# icon/core/__init__.py
"""Core mathematical operations (Parts 1-2 of the specification).

This package contains the building blocks: noise injection (§4), the
InfoNCE estimator (§A.4), Trust τ classification (§7), and deterministic
seed derivation.

These are pure functions where possible. They do not depend on adapters,
loaders, or PROTOCOL — those compositions live in `icon.pipeline`.
"""
