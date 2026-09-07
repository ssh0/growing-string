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


def _arc_length_moments(state: FilamentState) -> tuple[float, np.ndarray, float]:
    """Return contour length, first moment, and second raw moment.

    Each segment is treated as a straight, linearly interpolated piece.  The
    formulas are exact for that geometry and therefore do not change when a
    segment is split at its midpoint.
    """

    starts = state.positions[:-1]
    ends = state.positions[1:]
    lengths = np.linalg.norm(ends - starts, axis=1)
    total_length = float(np.sum(lengths))
    if total_length <= 1.0e-12:
        raise ValueError("filament contour length must be positive")
    first_moment = np.sum(
        lengths[:, None] * 0.5 * (starts + ends),
        axis=0,
    )
    second_moment = float(
        np.sum(
            lengths
            * (
                np.sum(starts * starts, axis=1)
                + np.sum(starts * ends, axis=1)
                + np.sum(ends * ends, axis=1)
            )
            / 3.0
        )
    )
    return total_length, first_moment, second_moment


def arc_length_weighted_center_of_mass(state: FilamentState) -> np.ndarray:
    """Return the center of mass of a uniform-density polyline by arc length."""

    total_length, first_moment, _ = _arc_length_moments(state)
    return first_moment / total_length


def arc_length_weighted_radius_of_gyration(state: FilamentState) -> float:
    """Return the node-count-independent radius of gyration of the polyline.

    The existing :func:`radius_of_gyration` intentionally keeps its historical
    node-average meaning.  This function instead integrates uniformly along
    the geometric contour, so midpoint remeshing leaves its value unchanged
    up to floating-point round-off.
    """

    total_length, first_moment, second_moment = _arc_length_moments(state)
    center = first_moment / total_length
    squared_radius = second_moment / total_length - float(np.dot(center, center))
    return float(np.sqrt(max(squared_radius, 0.0)))


def discrete_curvature(state: FilamentState) -> np.ndarray:
    """Return the historical length-normalized second difference."""

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
