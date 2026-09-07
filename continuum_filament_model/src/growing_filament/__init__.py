"""Prototype of a physically explicit growing-filament model.

This package is intentionally independent from the legacy implementations in
``growing_natural_length_model/`` and ``triangular_lattice/``.
"""

from .model import (
    FilamentState,
    ModelParameters,
    OverdampedGrowingFilament,
    remesh,
)

__all__ = [
    "FilamentState",
    "ModelParameters",
    "OverdampedGrowingFilament",
    "remesh",
]
