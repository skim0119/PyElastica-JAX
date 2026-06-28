from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402


def _two_straight_rods() -> tuple[list[ea.CosseratRod], list[int]]:
    rods = [
        ea.CosseratRod.straight_rod(
            4,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            0.5,
            0.01,
            1000.0,
            youngs_modulus=1.0e6,
            shear_modulus=1.0e6 / 1.5,
        ),
        ea.CosseratRod.straight_rod(
            4,
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            0.5,
            0.01,
            1000.0,
            youngs_modulus=1.0e6,
            shear_modulus=1.0e6 / 1.5,
        ),
    ]
    return rods, [0, 1]


def _make_two_rod_block(
    devices: Sequence[jax.Device],
) -> eaj._ShardedCosseratRodBlock:
    rod_block_cls = eaj.configure_rod_block_sharded(
        devices=devices,
        device_dtype=np.float64,
    )
    rods, system_idx = _two_straight_rods()
    return rod_block_cls(rods, system_idx)


def _make_two_rod_memory_block(
    device: jax.Device,
) -> eaj._CosseratRodMemoryBlock:
    rod_block_cls = eaj.configure_rod_block(device=device, device_dtype=np.float64)
    rods, system_idx = _two_straight_rods()
    return rod_block_cls(rods, system_idx)


def test_sharded_block_splits_across_devices_when_mesh_has_two_shards() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    block = _make_two_rod_block(devices[:2])
    state = block.jax_get_state()
    assert "shards" in state
    assert len(state["shards"]) == 2
    assert (
        state["shards"][0]["position_collection"].devices()
        != state["shards"][1]["position_collection"].devices()
    )


def test_sharded_position_collection_device_lives_on_primary_shard() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    block = _make_two_rod_block(devices[:2])
    positions = block.position_collection_device
    assert positions.shape == block.position_collection.shape
    assert positions.device == devices[0]
    assert np.allclose(
        np.asarray(positions),
        block.position_collection,
        rtol=0.0,
        atol=0.0,
    )


def test_sharded_merge_shard_states_concatenates_on_primary_device() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    block = _make_two_rod_block(devices[:2])
    state = block.jax_get_state()
    merged = block.merge_shard_states(state)
    assert merged["position_collection"].device == devices[0]
    assert merged["position_collection"].shape == block.position_collection.shape


def test_sharded_scatter_merged_state_returns_shard_local_devices() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    block = _make_two_rod_block(devices[:2])
    state = block.jax_get_state()
    merged = block.merge_shard_states(state)
    scattered = block.scatter_merged_state(merged, state)
    for shard_index, shard_state in enumerate(scattered["shards"]):
        assert (
            shard_state["position_collection"].devices()
            == state["shards"][shard_index]["position_collection"].devices()
        )


def test_sharded_merge_scatter_roundtrip_preserves_array_shapes() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    block = _make_two_rod_block(devices[:2])
    state = block.jax_get_state()
    merged = block.merge_shard_states(state)
    scattered = block.scatter_merged_state(merged, state)
    shape_keys = (
        "mass",
        "position_collection",
        "external_forces",
        "external_torques",
        "director_collection",
        "dilatation",
        "kappa",
    )
    for shard_index, shard_state in enumerate(scattered["shards"]):
        original = state["shards"][shard_index]
        for key in shape_keys:
            assert shard_state[key].shape == original[key].shape, key


def test_wrap_jax_block_operator_handles_sharded_state() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    from elastica_jax.modules.jax_ops_block import JAXOpsBlock

    block = _make_two_rod_block(devices[:2])
    state = block.jax_get_state()

    def identity_operator(merged_state, time):
        del time
        assert "director_collection" in merged_state
        return merged_state

    wrapped = JAXOpsBlock._wrap_jax_block_operator(
        block_state_idx=0,
        block_system=block,
        operator=identity_operator,
    )
    updated = wrapped(states=(state,), time=np.float64(0.0))
    assert "shards" in updated[0]


def test_sharded_block_exposes_global_rest_lengths_view() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    block = _make_two_rod_block(devices[:2])
    expected = np.concatenate(
        [shard_block.rest_lengths for shard_block in block._shard_blocks]
    )
    assert np.array_equal(block.rest_lengths, expected)

    unified_block = _make_two_rod_memory_block(devices[0])
    for rod_index in range(block.n_rods):
        sharded_start = int(block.start_idx_in_rod_elems[rod_index])
        sharded_end = int(block.end_idx_in_rod_elems[rod_index])
        unified_start = int(unified_block.start_idx_in_rod_elems[rod_index])
        unified_end = int(unified_block.end_idx_in_rod_elems[rod_index])
        assert np.array_equal(
            block.rest_lengths[sharded_start:sharded_end],
            unified_block.rest_lengths[unified_start:unified_end],
        )


def test_sharded_from_device_all_rods() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return

    class Simulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
        pass

    simulator = Simulator()
    rod_block = eaj.configure_rod_block_sharded(
        devices=devices[:2], device_dtype=np.float64
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    rods = [
        ea.CosseratRod.straight_rod(
            4,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            0.5,
            0.01,
            1000.0,
            youngs_modulus=1.0e6,
            shear_modulus=1.0e6 / 1.5,
        ),
        ea.CosseratRod.straight_rod(
            4,
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            0.5,
            0.01,
            1000.0,
            youngs_modulus=1.0e6,
            shear_modulus=1.0e6 / 1.5,
        ),
    ]
    for rod in rods:
        rod.external_forces[1, :] = 25.0
        simulator.append(rod)
    simulator.finalize()

    initial_positions = [rod.position_collection.copy() for rod in rods]
    eaj.PositionVerletJAX().integrate(
        simulator,
        time=0.0,
        final_time=0.005,
        dt=0.001,
    )

    rod_block.from_device()
    integrated = [rod.position_collection.copy() for rod in rods]
    assert all(
        not np.allclose(current, initial)
        for current, initial in zip(integrated, initial_positions, strict=True)
    )

    for rod in rods:
        rod.position_collection.fill(0.0)

    rod_block.from_device()
    for rod, expected in zip(rods, integrated, strict=True):
        np.testing.assert_allclose(rod.position_collection, expected)
