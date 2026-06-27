from __future__ import annotations

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import elastica as ea
import elastica_jax as eaj


def _make_two_rod_block(
    mesh: eaj.ExecutionMesh,
) -> eaj._ShardedCosseratRodBlock:
    rod_block_cls = eaj.configure_rod_block_sharded(mesh=mesh, device_dtype=np.float64)
    return rod_block_cls(
        [
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
        ],
        [0, 1],
    )


def test_sharded_block_unified_state_matches_primary() -> None:
    mesh = eaj.ExecutionMesh.from_devices(
        [jax.devices("cpu")[0]],
        n_rods=2,
    )
    block = _make_two_rod_block(mesh)
    state = block.jax_get_state()
    assert not eaj.is_sharded_block_state(state)
    primary = block._primary_block.jax_get_state()
    assert set(state) == set(primary)


def test_sharded_block_splits_across_devices_when_mesh_has_two_shards() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    mesh = eaj.ExecutionMesh.for_n_rods(2, devices=devices[:2])
    block = _make_two_rod_block(mesh)
    state = block.jax_get_state()
    assert eaj.is_sharded_block_state(state)
    assert len(state["shards"]) == 2
    assert (
        state["shards"][0]["position_collection"].devices()
        != state["shards"][1]["position_collection"].devices()
    )


def test_sharded_position_collection_device_lives_on_primary_shard() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    mesh = eaj.ExecutionMesh.for_n_rods(2, devices=devices[:2])
    block = _make_two_rod_block(mesh)
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
    mesh = eaj.ExecutionMesh.for_n_rods(2, devices=devices[:2])
    block = _make_two_rod_block(mesh)
    state = block.jax_get_state()
    merged = block.merge_shard_states(state)
    assert merged["position_collection"].device == devices[0]
    assert merged["position_collection"].shape == block.position_collection.shape


def test_sharded_block_exposes_global_rest_lengths_view() -> None:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        return
    mesh = eaj.ExecutionMesh.for_n_rods(2, devices=devices[:2])
    block = _make_two_rod_block(mesh)
    expected = np.concatenate(
        [shard_block.rest_lengths for shard_block in block._shard_blocks]
    )
    assert np.array_equal(block.rest_lengths, expected)

    unified_block = _make_two_rod_block(
        eaj.ExecutionMesh.from_devices([devices[0]], n_rods=2)
    )
    for rod_index in range(block.n_rods):
        sharded_start = int(block.start_idx_in_rod_elems[rod_index])
        sharded_end = int(block.end_idx_in_rod_elems[rod_index])
        unified_start = int(unified_block.start_idx_in_rod_elems[rod_index])
        unified_end = int(unified_block.end_idx_in_rod_elems[rod_index])
        assert np.array_equal(
            block.rest_lengths[sharded_start:sharded_end],
            unified_block.rest_lengths[unified_start:unified_end],
        )
