"""Observables shared by simulations and experiment comparison."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .model import FilamentState


def segment_lengths(state: FilamentState) -> np.ndarray:
    return np.linalg.norm(np.diff(state.positions, axis=0), axis=1)


def contour_length(state: FilamentState) -> float:
    return float(np.sum(segment_lengths(state)))


def center_of_mass(state: FilamentState) -> np.ndarray:
    return np.mean(state.positions, axis=0)


def radius_of_gyration(state: FilamentState) -> float:
    centered = state.positions - center_of_mass(state)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def discrete_curvature(state: FilamentState) -> np.ndarray:
    """Return a length-normalized second difference at interior nodes."""

    if state.n_nodes < 3:
        return np.empty(0, dtype=float)
    q = state.positions[:-2] - 2.0 * state.positions[1:-1] + state.positions[2:]
    local = 0.5 * (state.rest_lengths[:-1] + state.rest_lengths[1:])
    return np.linalg.norm(q, axis=1) / (local * local)


def summary(state: FilamentState) -> Dict[str, float]:
    curvature = discrete_curvature(state)
    return {
        "time": float(state.time),
        "step": float(state.step),
        "n_nodes": float(state.n_nodes),
        "reference_length": float(np.sum(state.rest_lengths)),
        "contour_length": contour_length(state),
        "radius_of_gyration": radius_of_gyration(state),
        "max_curvature": float(np.max(curvature)) if len(curvature) else 0.0,
    }
