"""Helpers for two-block indexing and host extraction."""

from __future__ import annotations

import numpy as np

import elastica as ea
import elastica_jax as eaj


def distinct_body_and_muscle_rod_types() -> tuple[
    type[ea.CosseratRod], type[ea.CosseratRod]
]:
    """Return distinct Cosserat rod subclasses for separate memory blocks."""
    skip_attrs = {
        "__dict__",
        "__weakref__",
        "__module__",
        "__annotations__",
        "__doc__",
        "__qualname__",
    }
    rod_dict = {
        key: value
        for key, value in ea.CosseratRod.__dict__.items()
        if key not in skip_attrs
    }
    body_type = type("MuscularSnakeBodyRod", ea.CosseratRod.__bases__, rod_dict)
    muscle_type = type("MuscularSnakeMuscleRod", ea.CosseratRod.__bases__, rod_dict)
    return body_type, muscle_type


def extract_rod_positions(
    block: eaj._CosseratRodMemoryBlock,
    *,
    n_rods: int,
) -> list[np.ndarray]:
    """Return per-rod node positions as ``(3, n_nodes)`` host arrays."""
    positions = np.asarray(block.device_state["position_collection"])
    assert positions.shape[0] == 3, "Rod positions must be stored as (3, n_nodes)."
    rod_positions: list[np.ndarray] = []
    for rod_idx in range(n_rods):
        start = int(block.start_idx_in_rod_nodes[rod_idx])
        end = int(block.end_idx_in_rod_nodes[rod_idx])
        rod_positions.append(positions[:, start:end].copy())
    return rod_positions


def extract_rod_velocities(
    block: eaj._CosseratRodMemoryBlock,
    *,
    n_rods: int,
) -> list[np.ndarray]:
    """Return per-rod nodal velocities as ``(3, n_nodes)`` host arrays."""
    velocities = np.asarray(block.device_state["velocity_collection"])
    rod_velocities: list[np.ndarray] = []
    for rod_idx in range(n_rods):
        start = int(block.start_idx_in_rod_nodes[rod_idx])
        end = int(block.end_idx_in_rod_nodes[rod_idx])
        rod_velocities.append(velocities[:, start:end].copy())
    return rod_velocities


def extract_simulation_rod_positions(
    body_block: eaj._CosseratRodMemoryBlock,
    muscle_block: eaj._CosseratRodMemoryBlock,
) -> list[np.ndarray]:
    """Return body then muscle rod positions from the two packed blocks."""
    return extract_rod_positions(body_block, n_rods=1) + extract_rod_positions(
        muscle_block,
        n_rods=muscle_block.n_rods,
    )


def extract_simulation_rod_velocities(
    body_block: eaj._CosseratRodMemoryBlock,
    muscle_block: eaj._CosseratRodMemoryBlock,
) -> list[np.ndarray]:
    """Return body then muscle rod velocities from the two packed blocks."""
    return extract_rod_velocities(body_block, n_rods=1) + extract_rod_velocities(
        muscle_block,
        n_rods=muscle_block.n_rods,
    )


def body_kinematics(
    body_position: np.ndarray,
    body_velocity: np.ndarray,
    body_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mass-weighted center-of-mass position and velocity for the body."""
    total_mass = np.sum(body_mass)
    center_of_mass = np.sum(body_position * body_mass[None, :], axis=1) / total_mass
    avg_velocity = np.sum(body_velocity * body_mass[None, :], axis=1) / total_mass
    return center_of_mass, avg_velocity
