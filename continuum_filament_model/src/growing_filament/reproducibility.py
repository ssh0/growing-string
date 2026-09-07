"""Canonical run metadata, hashing, and replay-comparison helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .model import FilamentState, ModelParameters

MANIFEST_SCHEMA_VERSION = "continuum-filament-manifest-1"


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/dataclass values to strict JSON primitives."""

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"non-finite value cannot be serialized: {value!r}")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with stable ordering and no insignificant whitespace."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_state_hash(state: FilamentState) -> str:
    """Hash state values independently of NumPy dtype/endianness defaults."""

    positions = np.asarray(state.positions, dtype="<f8", order="C")
    rest_lengths = np.asarray(state.rest_lengths, dtype="<f8", order="C")
    header = canonical_json_bytes(
        {
            "schema": "filament-state-canonical-1",
            "positions_shape": list(positions.shape),
            "rest_lengths_shape": list(rest_lengths.shape),
            "time": float(state.time),
            "step": int(state.step),
        }
    )
    return sha256_hex(
        b"continuum-filament-state\0"
        + header
        + b"\0"
        + positions.tobytes(order="C")
        + b"\0"
        + rest_lengths.tobytes(order="C")
    )


def canonical_input_hash(input_data: Any) -> str:
    """Hash a JSON-like input object or a NumPy array deterministically."""

    if isinstance(input_data, np.ndarray):
        values = np.asarray(input_data, dtype="<f8", order="C")
        payload = (
            b"continuum-filament-input-array\0"
            + canonical_json_bytes({"shape": list(values.shape), "dtype": "<f8"})
            + b"\0"
            + values.tobytes(order="C")
        )
        return sha256_hex(payload)
    return sha256_hex(b"continuum-filament-input\0" + canonical_json_bytes(input_data))


def detect_git_revision(cwd: Optional[str | Path] = None) -> Optional[str]:
    """Return the current revision when this run is inside a Git checkout."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def parameters_dict(parameters: ModelParameters) -> dict[str, Any]:
    return _jsonable(asdict(parameters))


def build_manifest(
    parameters: ModelParameters,
    initial_state: FilamentState,
    final_state: Optional[FilamentState] = None,
    events: Optional[Sequence[Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    input_data: Any = None,
    git_revision: Optional[str] = None,
) -> dict[str, Any]:
    """Build a self-contained manifest for a deterministic simulation run."""

    initial_hash = canonical_state_hash(initial_state)
    final_hash = canonical_state_hash(final_state or initial_state)
    if input_data is None:
        # A no-file fixture still has a meaningful input fingerprint: the
        # initial state and parameters that fully define its deterministic run.
        input_hash = canonical_input_hash(
            {"initial_state_hash": initial_hash, "parameters": parameters_dict(parameters)}
        )
        input_source = "initial_state_and_parameters"
    else:
        input_hash = canonical_input_hash(input_data)
        input_source = "provided_input"
    event_values = [_jsonable(event) for event in (events or ())]
    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "git_revision": git_revision if git_revision is not None else detect_git_revision(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "input_hash": input_hash,
        "input_hash_source": input_source,
        "parameters": parameters_dict(parameters),
        "initial_state_hash": initial_hash,
        "canonical_state_hash": final_hash,
        "accepted_steps": int(sum(1 for event in event_values if event.get("event_type") == "step_attempt" and event.get("accepted") is True)),
        "rejected_steps": int(sum(1 for event in event_values if event.get("event_type") == "step_attempt" and event.get("accepted") is False)),
        "event_count": len(event_values),
        "events": event_values,
        "metadata": _jsonable(dict(metadata or {})),
    }
    return manifest


def event_sequence_hash(events: Sequence[Mapping[str, Any]]) -> str:
    return sha256_hex(b"continuum-filament-events\0" + canonical_json_bytes(list(events)))


def reproducibility_signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the fields needed to compare two replay results."""

    events = manifest.get("events", [])
    return {
        "event_sequence_hash": event_sequence_hash(events),
        "canonical_state_hash": manifest.get("canonical_state_hash"),
        "initial_state_hash": manifest.get("initial_state_hash"),
        "input_hash": manifest.get("input_hash"),
        "parameters": _jsonable(manifest.get("parameters", {})),
        "python_version": manifest.get("python_version"),
        "numpy_version": manifest.get("numpy_version"),
        "git_revision": manifest.get("git_revision"),
        "metadata": _jsonable(manifest.get("metadata", {})),
        "accepted_steps": manifest.get("accepted_steps"),
        "rejected_steps": manifest.get("rejected_steps"),
    }


def compare_reproducibility(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare event sequence, canonical state hash, and run metadata."""

    left = reproducibility_signature(first)
    right = reproducibility_signature(second)
    differences = {
        key: {"first": left[key], "second": right[key]}
        for key in left
        if left[key] != right[key]
    }
    return {"match": not differences, "differences": differences}


def save_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return destination


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
