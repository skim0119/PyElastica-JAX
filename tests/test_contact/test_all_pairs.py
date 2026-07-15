from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from elastica_jax.contact.spatial_hash import (
    estimate_all_cross_rod_pairs,
    rebuild_all_pairs,
)
from elastica_jax.contact.spatial_hash_jax import rebuild_all_pairs_jax

jax.config.update("jax_enable_x64", True)


def _pair_set(
    first: np.ndarray, second: np.ndarray, count: int
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for slot in range(int(count)):
        left = int(first[slot])
        right = int(second[slot])
        assert left < right
        pairs.add((left, right))
    return pairs


def test_estimate_all_cross_rod_pairs_counts_distinct_rods_only() -> None:
    rod_ids = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    # 5 elements: C(5,2)=10 pairs. Same-rod: one pair in rod 0, one in rod 1.
    assert estimate_all_cross_rod_pairs(rod_ids) == 8


def test_all_pairs_host_and_device_agree_and_skip_same_rod() -> None:
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.01, 0.02, 0.0],
        ],
        dtype=np.float64,
    )
    rod_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    axes = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
    lengths = np.full(4, 0.1, dtype=np.float64)
    radii = np.full(4, 0.05, dtype=np.float64)
    max_pairs = 16

    host = rebuild_all_pairs(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        max_pairs=max_pairs,
    )
    device_first, device_second, device_count = rebuild_all_pairs_jax(
        centers=jnp.asarray(centers),
        rod_ids=jnp.asarray(rod_ids),
        axes=jnp.asarray(axes),
        lengths=jnp.asarray(lengths),
        radii=jnp.asarray(radii),
        max_pairs=max_pairs,
    )
    device_count_np = int(np.asarray(device_count))
    assert host.pair_count == device_count_np
    assert host.pair_count > 0
    host_pairs = _pair_set(host.pair_first, host.pair_second, host.pair_count)
    device_pairs = _pair_set(
        np.asarray(device_first), np.asarray(device_second), device_count_np
    )
    assert host_pairs == device_pairs
    for left, right in host_pairs:
        assert rod_ids[left] != rod_ids[right]


def test_all_pairs_ignores_same_rod_only_block() -> None:
    centers = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float64)
    rod_ids = np.array([0, 0], dtype=np.int32)
    axes = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    lengths = np.full(2, 0.1, dtype=np.float64)
    radii = np.full(2, 0.05, dtype=np.float64)
    host = rebuild_all_pairs(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        max_pairs=8,
    )
    assert host.pair_count == 0
    _, _, device_count = rebuild_all_pairs_jax(
        centers=jnp.asarray(centers),
        rod_ids=jnp.asarray(rod_ids),
        axes=jnp.asarray(axes),
        lengths=jnp.asarray(lengths),
        radii=jnp.asarray(radii),
        max_pairs=8,
    )
    assert int(np.asarray(device_count)) == 0
