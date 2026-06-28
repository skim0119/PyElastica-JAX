"""Shared helpers for ActiveMatter examples."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class _CylinderPackingParameters(Protocol):
    n_snakes: int
    length: float
    packing_initial_radial_span_ratio: float
    packing_initial_vertical_span_ratio: float

    @property
    def radius(self) -> float: ...


def instantiate_rods_in_cylinder(
    parameters: _CylinderPackingParameters,
    seed: int,
    *,
    max_attempts: int = 10_000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Place rods with random pose inside a vertical packing cylinder.

    Each rod is accepted only when both endpoints lie inside the cylinder.
    """
    rng = np.random.default_rng(seed)
    n_rods = parameters.n_snakes
    cylinder_radius = parameters.packing_initial_radial_span_ratio * parameters.length
    height = parameters.packing_initial_vertical_span_ratio * parameters.length
    min_z = parameters.radius
    rod_length = parameters.length

    rho = cylinder_radius * np.sqrt(rng.random((n_rods, max_attempts)))
    phi = 2.0 * np.pi * rng.random((n_rods, max_attempts))
    start_z = min_z + rng.random((n_rods, max_attempts)) * (height - min_z)
    start_x = rho * np.cos(phi)
    start_y = rho * np.sin(phi)

    alpha = 2.0 * np.pi * rng.random((n_rods, max_attempts))
    beta = 0.2 * np.pi * rng.random((n_rods, max_attempts))
    cos_beta = np.cos(beta)
    dir_x = np.cos(alpha) * cos_beta
    dir_y = np.sin(alpha) * cos_beta
    dir_z = np.sin(beta)

    end_x = start_x + rod_length * dir_x
    end_y = start_y + rod_length * dir_y
    end_z = start_z + rod_length * dir_z

    valid = (
        (np.hypot(end_x, end_y) <= cylinder_radius)
        & (min_z <= end_z)
        & (end_z <= height)
    )
    if not np.all(valid.any(axis=1)):
        raise RuntimeError("Could not place a snake inside the random cylinder.")

    attempt_idx = valid.argmax(axis=1)
    rod_idx = np.arange(n_rods)
    starts = np.stack(
        (
            start_x[rod_idx, attempt_idx],
            start_y[rod_idx, attempt_idx],
            start_z[rod_idx, attempt_idx],
        ),
        axis=1,
    )
    directions = np.stack(
        (
            dir_x[rod_idx, attempt_idx],
            dir_y[rod_idx, attempt_idx],
            dir_z[rod_idx, attempt_idx],
        ),
        axis=1,
    )
    return [(starts[i], directions[i]) for i in range(n_rods)]
