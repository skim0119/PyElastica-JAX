from __future__ import annotations

import numpy as np

from elastica_jax.contact.spatial_hash import rebuild_spatial_hash_pairs


def test_spatial_hash_finds_cross_rod_neighbors_only() -> None:
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.03, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rod_ids = np.array([0, 0, 1, 1], dtype=np.int32)
    axes = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
    lengths = np.full(4, 0.1, dtype=np.float64)
    radii = np.full(4, 0.01, dtype=np.float64)
    buffer = rebuild_spatial_hash_pairs(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        cell_size=0.5,
        max_pairs=16,
    )
    assert buffer.pair_count >= 1
    assert buffer.pair_first[0] != buffer.pair_second[0]
    assert rod_ids[buffer.pair_first[0]] != rod_ids[buffer.pair_second[0]]


def test_spatial_hash_ignores_same_rod_pairs() -> None:
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rod_ids = np.array([0, 0], dtype=np.int32)
    axes = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    lengths = np.full(2, 0.1, dtype=np.float64)
    radii = np.full(2, 0.01, dtype=np.float64)
    buffer = rebuild_spatial_hash_pairs(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        cell_size=0.5,
        max_pairs=16,
    )
    assert buffer.pair_count == 0
