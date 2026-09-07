"""Simple, non-pickle trajectory serialization for the prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .model import FilamentState, ModelParameters
from .reproducibility import build_manifest, parameters_dict

SCHEMA_VERSION = "continuum-filament-0.1"


def save_trajectory(
    path: str | Path,
    trajectory: Iterable[FilamentState],
    parameters: ModelParameters,
    metadata: Optional[Mapping[str, object]] = None,
    events: Optional[Sequence[Mapping[str, Any]]] = None,
    manifest: Optional[Mapping[str, Any]] = None,
    input_data: Any = None,
) -> Path:
    """Save a trajectory, structured events, and a replay manifest.

    Variable-node trajectories remain primitive arrays plus JSON metadata; no
    pickle/object arrays are introduced.  ``events`` and ``manifest`` are
    optional for compatibility with the original Gate 0/1 serializer.
    """

    states = list(trajectory)
    if not states:
        raise ValueError("trajectory must not be empty")
    positions = np.concatenate([s.positions for s in states], axis=0)
    position_offsets = np.cumsum([0] + [len(s.positions) for s in states])
    rest_lengths = np.concatenate([s.rest_lengths for s in states], axis=0)
    rest_offsets = np.cumsum([0] + [len(s.rest_lengths) for s in states])
    event_values = [dict(event) for event in (events or ())]
    manifest_value = dict(manifest) if manifest is not None else build_manifest(
        parameters,
        states[0],
        final_state=states[-1],
        events=event_values,
        metadata=metadata,
        input_data=input_data,
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "parameters": parameters_dict(parameters),
        "metadata": dict(metadata or {}),
        "manifest": manifest_value,
        "events": event_values,
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
        metadata_json=np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False)),
    )
    return destination


def load_trajectory_metadata(path: str | Path) -> dict[str, Any]:
    """Load only the JSON metadata, events, and manifest from a trajectory."""

    with np.load(Path(path), allow_pickle=False) as archive:
        raw = archive["metadata_json"]
        value = raw.item() if raw.ndim == 0 else raw.tolist()
    return json.loads(str(value))
