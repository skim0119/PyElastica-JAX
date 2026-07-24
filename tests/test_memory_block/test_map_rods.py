"""Tests for Block.map_rods layout projection (ADR-0005)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
CPU_DEVICE = jax.devices("cpu")[0]

from elastica.rod.cosserat_rod import CosseratRod  # noqa: E402
from elastica_jax.memory_block.memory_block_rod_jax import (  # noqa: E402
    _CosseratRodMemoryBlock,
)
from elastica_jax.memory_block.memory_block_rod_vertical_jax import (  # noqa: E402
    _CosseratRodVerticalMemoryBlock,
)
from elastica_jax.memory_block.protocol import RodLocalOp  # noqa: E402
from elastica_jax.memory_block.rod_local_map import RodLocalState  # noqa: E402


def _build_rod(n_elems: int = 4, *, seed: int = 0) -> CosseratRod:
    rng = np.random.default_rng(seed)
    return CosseratRod.straight_rod(
        n_elements=n_elems,
        start=rng.standard_normal(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.0,
        base_radius=0.05,
        density=1_000.0,
        youngs_modulus=1.0e6,
    )


def _scale_external_forces(
    rod_view: RodLocalState,
    time: np.float64,
) -> RodLocalState:
    del time
    rod_view.external_forces = rod_view.external_forces * np.float64(2.0)
    return rod_view


def _scale_external_forces_by(scale: float) -> RodLocalOp:
    def _op(rod_view: RodLocalState, time: np.float64) -> RodLocalState:
        del time
        rod_view.external_forces = rod_view.external_forces * np.float64(scale)
        return rod_view

    return _op


def test_horizontal_map_rods_scales_each_rod_external_forces() -> None:
    rods = [_build_rod(4, seed=0), _build_rod(4, seed=1)]
    with jax.default_device(CPU_DEVICE):
        block = _CosseratRodMemoryBlock(device=CPU_DEVICE, device_dtype=np.float64)
        block(rods, [0, 1])
        state = block.jax_get_state()
        updated = block.map_rods(state, _scale_external_forces, np.float64(0.0))

    for rod_idx in range(2):
        start = int(block.start_idx_in_rod_nodes[rod_idx])
        end = int(block.end_idx_in_rod_nodes[rod_idx])
        np.testing.assert_allclose(
            np.asarray(updated["external_forces"])[:, start:end],
            2.0 * np.asarray(state["external_forces"])[:, start:end],
        )


def test_block_rods_returns_packed_host_rods_in_order() -> None:
    rods = [_build_rod(4, seed=0), _build_rod(4, seed=1)]
    with jax.default_device(CPU_DEVICE):
        horizontal = _CosseratRodMemoryBlock(device=CPU_DEVICE, device_dtype=np.float64)
        horizontal(rods, [0, 1])
        vertical = _CosseratRodVerticalMemoryBlock(
            device=CPU_DEVICE, device_dtype=np.float64
        )
        vertical(rods, [0, 1])

    assert horizontal.rods() == tuple(rods)
    assert vertical.rods() == tuple(rods)


def test_horizontal_map_rods_applies_heterogeneous_ops_per_rod() -> None:
    rods = [_build_rod(4, seed=0), _build_rod(4, seed=1)]
    with jax.default_device(CPU_DEVICE):
        block = _CosseratRodMemoryBlock(device=CPU_DEVICE, device_dtype=np.float64)
        block(rods, [0, 1])
        state = block.jax_get_state()
        updated = block.map_rods(
            state,
            (_scale_external_forces_by(2.0), _scale_external_forces_by(3.0)),
            np.float64(0.0),
        )

    start0 = int(block.start_idx_in_rod_nodes[0])
    end0 = int(block.end_idx_in_rod_nodes[0])
    start1 = int(block.start_idx_in_rod_nodes[1])
    end1 = int(block.end_idx_in_rod_nodes[1])
    np.testing.assert_allclose(
        np.asarray(updated["external_forces"])[:, start0:end0],
        2.0 * np.asarray(state["external_forces"])[:, start0:end0],
    )
    np.testing.assert_allclose(
        np.asarray(updated["external_forces"])[:, start1:end1],
        3.0 * np.asarray(state["external_forces"])[:, start1:end1],
    )


def test_vertical_map_rods_scales_each_rod_external_forces() -> None:
    rods = [_build_rod(4, seed=0), _build_rod(4, seed=1)]
    with jax.default_device(CPU_DEVICE):
        block = _CosseratRodVerticalMemoryBlock(
            device=CPU_DEVICE, device_dtype=np.float64
        )
        block(rods, [0, 1])
        state = block.jax_get_state()
        updated = block.map_rods(state, _scale_external_forces, np.float64(0.0))

    np.testing.assert_allclose(
        np.asarray(updated["external_forces"]),
        2.0 * np.asarray(state["external_forces"]),
    )
