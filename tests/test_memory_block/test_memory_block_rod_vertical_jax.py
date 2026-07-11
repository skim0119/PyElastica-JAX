"""Tests for stacked-axis Cosserat rod memory blocks."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
CPU_DEVICE = jax.devices("cpu")[0]

from elastica.modules import BaseSystemCollection  # noqa: E402
from elastica.rod.cosserat_rod import CosseratRod  # noqa: E402
from elastica_jax.memory_block.block_factory import configure_rod_block  # noqa: E402
from elastica_jax.memory_block.memory_block_rod_jax import (  # noqa: E402
    _CosseratRodMemoryBlock,
)
from elastica_jax.memory_block.memory_block_rod_vertical_jax import (  # noqa: E402
    _CosseratRodVerticalMemoryBlock,
)


def _build_rod(n_elems: int = 8, *, seed: int = 0) -> CosseratRod:
    rng = np.random.default_rng(seed)
    start = rng.standard_normal(3)
    return CosseratRod.straight_rod(
        n_elements=n_elems,
        start=start,
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.0,
        base_radius=0.05,
        density=1_000.0,
        youngs_modulus=1.0e6,
    )


def _make_vertical_block(rods: list[CosseratRod]) -> _CosseratRodVerticalMemoryBlock:
    with jax.default_device(CPU_DEVICE):
        block = _CosseratRodVerticalMemoryBlock(
            device=CPU_DEVICE, device_dtype=np.float64
        )
        return block(rods, list(range(len(rods))))


class _DummySimulator(BaseSystemCollection):
    pass


def test_vertical_block_stacks_on_leading_axis() -> None:
    rods = [_build_rod(8, seed=0), _build_rod(8, seed=1), _build_rod(8, seed=2)]
    block = _make_vertical_block(rods)

    assert block.n_rods == 3
    assert block.position_collection.shape == (3, 3, 9)
    assert block.director_collection.shape == (3, 3, 3, 8)
    assert block.mass.shape == (3, 9)
    assert block.omega_collection.shape == (3, 3, 8)
    assert_array_equal(block.position_collection[1], rods[1].position_collection)


def test_vertical_block_rejects_unequal_lengths() -> None:
    rods = [_build_rod(8, seed=0), _build_rod(6, seed=1)]
    block = _CosseratRodVerticalMemoryBlock(device=CPU_DEVICE, device_dtype=np.float64)
    with pytest.raises(AssertionError, match="equal-length"):
        block(rods, [0, 1])


def test_vertical_block_to_from_device_roundtrip() -> None:
    rods = [_build_rod(6, seed=0), _build_rod(6, seed=1)]
    block = _make_vertical_block(rods)

    updated = np.asarray(block._device_state["position_collection"]) + 1.5
    block._device_state["position_collection"] = jax.device_put(
        updated, device=CPU_DEVICE
    )
    block.from_device(variables=("position_collection",))

    assert_array_equal(block.position_collection, updated)
    assert_array_equal(rods[0].position_collection, updated[0])
    assert_array_equal(rods[1].position_collection, updated[1])


def test_vertical_block_jax_force_matches_horizontal_per_rod() -> None:
    rods = [_build_rod(8, seed=0), _build_rod(8, seed=1)]
    vertical = _make_vertical_block(rods)

    horizontal_forces = []
    for rod in rods:
        horizontal = _CosseratRodMemoryBlock(device=CPU_DEVICE, device_dtype=np.float64)
        horizontal([rod], [0])
        state = horizontal.jax_compute_internal_forces_and_torques(
            horizontal.jax_get_state(), np.float64(0.0)
        )
        horizontal_forces.append(np.asarray(state["internal_forces"]))

    vertical_state = vertical.jax_compute_internal_forces_and_torques(
        vertical.jax_get_state(), np.float64(0.0)
    )
    vertical_forces = np.asarray(vertical_state["internal_forces"])

    for rod_idx, expected in enumerate(horizontal_forces):
        assert_allclose(vertical_forces[rod_idx], expected, rtol=1.0e-10, atol=1.0e-10)


def test_configure_rod_block_accepts_vertical_inner_cls() -> None:
    simulator = _DummySimulator()
    rod_block = configure_rod_block(
        device=CPU_DEVICE,
        device_dtype=np.float64,
        inner_block_cls=_CosseratRodVerticalMemoryBlock,
    )
    simulator.enable_block_supports(CosseratRod, rod_block)

    rods = [_build_rod(8, seed=0), _build_rod(8, seed=1)]
    for rod in rods:
        simulator.append(rod)
    simulator.finalize()

    assert isinstance(rod_block, _CosseratRodVerticalMemoryBlock)
    assert rod_block.n_rods == 2
    assert rod_block.position_collection.shape == (2, 3, 9)
