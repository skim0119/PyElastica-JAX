"""Tests for rod block host/device communication."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica as ea
import elastica_jax as eaj


class _RodBlockSimulator(eaj.Simulator):
    pass


def _build_rod(*, n_elements: int = 8) -> ea.CosseratRod:
    return ea.CosseratRod.straight_rod(
        n_elements=n_elements,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.0,
        base_radius=0.05,
        density=1_000.0,
        youngs_modulus=1.0e6,
    )


def _build_simulator_with_block(
    rods: list[ea.CosseratRod],
) -> tuple[_RodBlockSimulator, eaj._CosseratRodMemoryBlock]:
    simulator = _RodBlockSimulator()
    rod_block = eaj.configure_rod_block()
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    for rod in rods:
        simulator.append(rod)
    simulator.finalize()
    assert isinstance(rod_block, eaj._CosseratRodMemoryBlock)
    return simulator, rod_block


def test_from_device_single_rod_after_integration():
    rod = _build_rod()
    rod.external_forces[1, :] = 25.0
    initial_position = rod.position_collection.copy()
    simulator, block = _build_simulator_with_block([rod])

    stepper = eaj.PositionVerletJAX()
    stepper.integrate(simulator, time=0.0, final_time=0.01, dt=0.001)

    np.testing.assert_allclose(rod.position_collection, initial_position)

    block.from_device(rod, variables=("position_collection",))
    assert not np.allclose(rod.position_collection, initial_position)


def test_from_device_all_rods():
    rods = [_build_rod(), _build_rod(n_elements=10)]
    for rod in rods:
        rod.external_forces[1, :] = 25.0
    simulator, block = _build_simulator_with_block(rods)

    stepper = eaj.PositionVerletJAX()
    stepper.integrate(simulator, time=0.0, final_time=0.005, dt=0.001)

    block.from_device(variables=("position_collection",))
    integrated = [rod.position_collection.copy() for rod in rods]

    for rod in rods:
        rod.position_collection.fill(0.0)

    block.from_device(variables=("position_collection",))

    for rod, expected in zip(rods, integrated, strict=True):
        np.testing.assert_allclose(rod.position_collection, expected)


def test_from_device_selected_rod_subset():
    rods = [_build_rod(), _build_rod(n_elements=10)]
    rods[0].external_forces[1, :] = 25.0
    initial_positions = [rod.position_collection.copy() for rod in rods]
    simulator, block = _build_simulator_with_block(rods)

    stepper = eaj.PositionVerletJAX()
    stepper.integrate(simulator, time=0.0, final_time=0.005, dt=0.001)

    block.from_device(rods[0], variables=("position_collection",))

    assert not np.allclose(rods[0].position_collection, initial_positions[0])
    np.testing.assert_allclose(rods[1].position_collection, initial_positions[1])


def test_to_device_resets_block_from_rod():
    rod = _build_rod()
    rod.external_forces[1, :] = 25.0
    simulator, block = _build_simulator_with_block([rod])
    reset_position = rod.position_collection.copy()

    stepper = eaj.PositionVerletJAX()
    stepper.integrate(simulator, time=0.0, final_time=0.01, dt=0.001)

    block.from_device(rod, variables=("position_collection",))
    assert not np.allclose(
        np.asarray(block._device_state["position_collection"]),
        reset_position,
    )

    rod.position_collection[:] = reset_position
    block.to_device(rod, variables=("position_collection",))

    np.testing.assert_allclose(
        np.asarray(block._device_state["position_collection"]),
        reset_position,
    )


def test_from_device_unknown_variable_raises_key_error():
    rod = _build_rod()
    _, block = _build_simulator_with_block([rod])

    with pytest.raises(KeyError, match="not_a_real_field"):
        block.from_device(rod, variables=("not_a_real_field",))


def test_from_device_unknown_rod_raises_value_error():
    rod = _build_rod()
    other_rod = _build_rod(n_elements=12)
    _, block = _build_simulator_with_block([rod])

    with pytest.raises(ValueError, match="was not packed into this block"):
        block.from_device(other_rod, variables=("position_collection",))


def test_iterate_rods_yields_each_rod_slice():
    rods = [_build_rod(n_elements=8), _build_rod(n_elements=10)]
    _, block = _build_simulator_with_block(rods)

    views = list(block.iterate_rods())
    assert len(views) == len(rods)

    for view, rod in zip(views, rods, strict=True):
        positions = np.asarray(view.position_collection)
        assert positions.shape == rod.position_collection.shape
        np.testing.assert_allclose(positions, rod.position_collection)

        masses = np.asarray(view.mass)
        assert masses.shape == rod.mass.shape
        np.testing.assert_allclose(masses, rod.mass)


def test_iterate_rods_reflects_integrated_device_state():
    rods = [_build_rod(n_elements=8), _build_rod(n_elements=10)]
    for rod in rods:
        rod.external_forces[1, :] = 25.0
    initial_positions = [rod.position_collection.copy() for rod in rods]
    simulator, block = _build_simulator_with_block(rods)

    stepper = eaj.PositionVerletJAX()
    stepper.integrate(simulator, time=0.0, final_time=0.005, dt=0.001)

    view_positions = [
        np.asarray(view.position_collection) for view in block.iterate_rods()
    ]
    for positions, start in zip(view_positions, initial_positions, strict=True):
        assert not np.allclose(positions, start)

    block.from_device(variables=("position_collection",))
    for positions, rod in zip(view_positions, rods, strict=True):
        np.testing.assert_allclose(positions, rod.position_collection)


def test_rod_block_is_pytree_compatible():
    rod = _build_rod()
    simulator, block = _build_simulator_with_block([rod])

    leaves, treedef = jax.tree_util.tree_flatten(block)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert jax.tree_util.tree_unflatten(treedef, leaves) is block

    stepper = eaj.PositionVerletJAX()
    stepper.integrate(simulator, time=0.0, final_time=0.001, dt=0.001)
    jax.block_until_ready(block)
