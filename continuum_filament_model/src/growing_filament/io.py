"""Simple, non-pickle trajectory serialization for the prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from .model import FilamentState, ModelParameters

SCHEMA_VERSION = "continuum-filament-0.1"


def save_trajectory(
    path: str | Path,
    trajectory: Iterable[FilamentState],
    parameters: ModelParameters,
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    """Save variable-node trajectories as primitive arrays plus JSON metadata."""

    states = list(trajectory)
    if not states:
        raise ValueError("trajectory must not be empty")
    positions = np.concatenate([s.positions for s in states], axis=0)
    position_offsets = np.cumsum([0] + [len(s.positions) for s in states])
    rest_lengths = np.concatenate([s.rest_lengths for s in states], axis=0)
    rest_offsets = np.cumsum([0] + [len(s.rest_lengths) for s in states])
    meta = {
        "schema_version": SCHEMA_VERSION,
        "parameters": parameters.__dict__,
        "metadata": dict(metadata or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        positions=positions,
        position_offsets=position_offsets,
        rest_lengths=rest_lengths,
        rest_offsets=rest_offsets,
        times=np.asarray([s.time for s in states], dtype=float),
        steps=np.asarray([s.step for s in states], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    return destination
