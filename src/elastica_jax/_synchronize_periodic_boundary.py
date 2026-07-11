"""Synchronize periodic boundaries for circular rods in block memory."""

from __future__ import annotations

from typing import Any


def _synchronize_periodic_boundary_of_vector_collection(
    input: Any, periodic_idx: Any
) -> None:
    input[..., periodic_idx[0, :]] = input[..., periodic_idx[1, :]]


def _synchronize_periodic_boundary_of_matrix_collection(
    input: Any, periodic_idx: Any
) -> None:
    input[..., periodic_idx[0, :]] = input[..., periodic_idx[1, :]]


def _synchronize_periodic_boundary_of_scalar_collection(
    input: Any, periodic_idx: Any
) -> None:
    input[..., periodic_idx[0, :]] = input[..., periodic_idx[1, :]]
