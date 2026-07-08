from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from elastica_jax.contact.spatial_hash import rebuild_spatial_hash_pairs
from elastica_jax.contact.spatial_hash_jax import rebuild_spatial_hash_pairs_jax

jax.config.update("jax_enable_x64", True)


def _reference_and_jax_buffers(
  *,
  centers: np.ndarray,
  rod_ids: np.ndarray,
  axes: np.ndarray,
  lengths: np.ndarray,
  radii: np.ndarray,
  cell_size: float,
  max_pairs: int,
  max_cell_occ: int = 64,
):
    host = rebuild_spatial_hash_pairs(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        cell_size=cell_size,
        max_pairs=max_pairs,
    )
    device_first, device_second, device_count = rebuild_spatial_hash_pairs_jax(
        centers=jnp.asarray(centers),
        rod_ids=jnp.asarray(rod_ids),
        axes=jnp.asarray(axes),
        lengths=jnp.asarray(lengths),
        radii=jnp.asarray(radii),
        cell_size=cell_size,
        max_pairs=max_pairs,
        max_cell_occ=max_cell_occ,
    )
    return host, np.asarray(device_count), np.asarray(device_first), np.asarray(device_second)


def _pair_set(first: np.ndarray, second: np.ndarray, count: int) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for slot in range(int(count)):
        left = int(first[slot])
        right = int(second[slot])
        assert left < right
        pairs.add((left, right))
    return pairs


def test_device_spatial_hash_matches_host_reference() -> None:
    rng = np.random.default_rng(2026)
    n = 24
    centers = rng.normal(size=(n, 3)) * 0.2
    rod_ids = np.repeat(np.arange(6, dtype=np.int32), 4)
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    lengths = np.full(n, 0.1, dtype=np.float64)
    radii = np.full(n, 0.02, dtype=np.float64)
    host, device_count, device_first, device_second = _reference_and_jax_buffers(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        cell_size=0.25,
        max_pairs=256,
    )
    assert int(device_count) == host.pair_count
    assert _pair_set(device_first, device_second, int(device_count)) == _pair_set(
        host.pair_first, host.pair_second, host.pair_count
    )


def test_device_spatial_hash_ignores_same_rod_pairs() -> None:
    centers = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float64)
    rod_ids = np.array([0, 0], dtype=np.int32)
    axes = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    lengths = np.full(2, 0.1, dtype=np.float64)
    radii = np.full(2, 0.01, dtype=np.float64)
    _, device_count, _, _ = _reference_and_jax_buffers(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        cell_size=0.5,
        max_pairs=16,
    )
    assert int(device_count) == 0


@pytest.mark.parametrize("n", [0, 1])
def test_device_spatial_hash_handles_small_inputs(n: int) -> None:
    centers = np.zeros((n, 3), dtype=np.float64)
    rod_ids = np.zeros(n, dtype=np.int32)
    axes = np.tile(np.array([1.0, 0.0, 0.0]), (max(n, 1), 1))[:n]
    lengths = np.full(max(n, 1), 0.1, dtype=np.float64)[:n]
    radii = np.full(max(n, 1), 0.01, dtype=np.float64)[:n]
    if n == 0:
        lengths = np.array([], dtype=np.float64)
        radii = np.array([], dtype=np.float64)
        axes = np.zeros((0, 3), dtype=np.float64)
    _, device_count, device_first, device_second = _reference_and_jax_buffers(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        cell_size=0.5,
        max_pairs=8,
    )
    assert int(device_count) == 0
    assert np.all(device_first == -1)
    assert np.all(device_second == -1)
