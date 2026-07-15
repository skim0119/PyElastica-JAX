"""Device-side spatial-hash broad phase for capsule contact."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from elastica_jax.contact.spatial_hash import _NEIGHBOR_OFFSETS

_HASH_PRIMES = (73856093, 19349663, 83492791)
_MAX_CELL_WINDOWS = 8


def hash_cell_coords(cells: jax.Array) -> jax.Array:
    """Map ``(N, 3)`` integer cell coordinates to sortable scalar keys."""
    coords = cells.astype(jnp.int64)
    primes = jnp.asarray(_HASH_PRIMES, dtype=coords.dtype)
    return jnp.sum(coords * primes[None, :], axis=1)


def capsule_aabb_overlap(
    center_a: jax.Array,
    axis_a: jax.Array,
    half_length_a: jax.Array,
    radius_a: jax.Array,
    center_b: jax.Array,
    axis_b: jax.Array,
    half_length_b: jax.Array,
    radius_b: jax.Array,
) -> jax.Array:
    """Vectorized axis-aligned bounding-box overlap test for capsules."""
    extent_a = half_length_a[..., None] * jnp.abs(axis_a) + radius_a[..., None]
    extent_b = half_length_b[..., None] * jnp.abs(axis_b) + radius_b[..., None]
    return jnp.all(jnp.abs(center_a - center_b) <= extent_a + extent_b, axis=-1)


def _neighbor_offsets_without_origin() -> jax.Array:
    offsets = [offset for offset in _NEIGHBOR_OFFSETS if offset != (0, 0, 0)]
    return jnp.asarray(offsets, dtype=jnp.int32)


def _segment_bounds(sorted_keys: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return per-sorted-position half-open segment bounds ``[lo, hi)``."""
    n = sorted_keys.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)
    is_start = jnp.concatenate([jnp.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
    seg_lo = jnp.maximum.accumulate(jnp.where(is_start, indices, jnp.int32(-1)))
    seg_lo = jnp.where(seg_lo < 0, jnp.int32(0), seg_lo)
    is_end = jnp.concatenate([sorted_keys[1:] != sorted_keys[:-1], jnp.array([True])])
    seg_hi_fill = jnp.where(is_end, indices, jnp.int32(n))
    seg_hi = jnp.minimum.accumulate(seg_hi_fill[::-1])[::-1] + jnp.int32(1)
    return seg_lo, seg_hi


def _windowed_segment_pairs(
    *,
    anchor_idx: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    require_after: jax.Array,
    sorted_indices: jax.Array,
    max_cell_occ: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    windows = jnp.arange(_MAX_CELL_WINDOWS, dtype=jnp.int32)
    rel = jnp.arange(max_cell_occ, dtype=jnp.int32)
    partner_pos = (lo + windows[:, None] * max_cell_occ + rel[None, :]).reshape(-1)
    partner_idx = sorted_indices[partner_pos]
    first = jnp.minimum(anchor_idx, partner_idx)
    second = jnp.maximum(anchor_idx, partner_idx)
    valid = (
        (partner_pos >= lo)
        & (partner_pos < hi)
        & (partner_pos > require_after)
        & (first < second)
    )
    return first, second, valid


def _same_cell_candidates(
    *,
    pos: jax.Array,
    seg_lo: jax.Array,
    seg_hi: jax.Array,
    sorted_indices: jax.Array,
    max_cell_occ: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _windowed_segment_pairs(
        anchor_idx=sorted_indices[pos],
        lo=seg_lo,
        hi=seg_hi,
        require_after=pos,
        sorted_indices=sorted_indices,
        max_cell_occ=max_cell_occ,
    )


def _neighbor_cell_candidates(
    *,
    pos: jax.Array,
    sorted_keys: jax.Array,
    sorted_indices: jax.Array,
    cells: jax.Array,
    max_cell_occ: int,
    neighbor_offsets: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    idx_i = sorted_indices[pos]
    cell_i = cells[idx_i]
    neighbor_cells = cell_i[None, :] + neighbor_offsets
    neighbor_keys = hash_cell_coords(neighbor_cells)

    def one_neighbor(neighbor_key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        lo = jnp.searchsorted(sorted_keys, neighbor_key, side="left")
        hi = jnp.searchsorted(sorted_keys, neighbor_key, side="right")
        return _windowed_segment_pairs(
            anchor_idx=idx_i,
            lo=lo,
            hi=hi,
            require_after=lo - 1,
            sorted_indices=sorted_indices,
            max_cell_occ=max_cell_occ,
        )

    first, second, valid = jax.vmap(one_neighbor)(neighbor_keys)
    return first, second, valid


def _position_candidates(
    pos: jax.Array,
    seg_lo: jax.Array,
    seg_hi: jax.Array,
    sorted_keys: jax.Array,
    sorted_indices: jax.Array,
    cells: jax.Array,
    max_cell_occ: int,
    neighbor_offsets: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    same_first, same_second, same_valid = _same_cell_candidates(
        pos=pos,
        seg_lo=seg_lo[pos],
        seg_hi=seg_hi[pos],
        sorted_indices=sorted_indices,
        max_cell_occ=max_cell_occ,
    )
    neigh_first, neigh_second, neigh_valid = _neighbor_cell_candidates(
        pos=pos,
        sorted_keys=sorted_keys,
        sorted_indices=sorted_indices,
        cells=cells,
        max_cell_occ=max_cell_occ,
        neighbor_offsets=neighbor_offsets,
    )
    first = jnp.concatenate([same_first, neigh_first.reshape(-1)], axis=0)
    second = jnp.concatenate([same_second, neigh_second.reshape(-1)], axis=0)
    valid = jnp.concatenate([same_valid, neigh_valid.reshape(-1)], axis=0)
    return first, second, valid


def _compact_pair_buffer(
    first: jax.Array,
    second: jax.Array,
    valid: jax.Array,
    *,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_candidates = first.shape[0]
    order_key = jnp.where(
        valid,
        jnp.arange(n_candidates, dtype=jnp.int32),
        n_candidates + 1,
    )
    perm = jnp.argsort(order_key)
    first_sorted = first[perm]
    second_sorted = second[perm]
    pair_count = jnp.minimum(jnp.sum(valid, dtype=jnp.int32), jnp.int32(max_pairs))
    pair_first = jnp.concatenate(
        [first_sorted, jnp.full(max_pairs, -1, dtype=jnp.int32)]
    )[:max_pairs]
    pair_second = jnp.concatenate(
        [second_sorted, jnp.full(max_pairs, -1, dtype=jnp.int32)]
    )[:max_pairs]
    slots = jnp.arange(max_pairs, dtype=jnp.int32)
    active = slots < pair_count
    pair_first = jnp.where(active, pair_first, jnp.int32(-1))
    pair_second = jnp.where(active, pair_second, jnp.int32(-1))
    return pair_first, pair_second, pair_count


def rebuild_spatial_hash_pairs_jax(
    *,
    centers: jax.Array,
    rod_ids: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    cell_size: float,
    max_pairs: int,
    max_cell_occ: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rebuild a bounded cross-rod pair list entirely on device.

    Parameters
    ----------
    centers, rod_ids, axes, lengths, radii
        Per-capsule geometry aligned on the leading axis ``N``.
    cell_size : float
        Uniform spatial-hash cell edge length.
    max_pairs : int
        Fixed-size output buffer capacity.
    max_cell_occ : int
        Window size when scanning same-cell or neighbor-cell buckets. Up to
        ``_MAX_CELL_WINDOWS * max_cell_occ`` capsules are examined per bucket.

    Returns
    -------
    pair_first, pair_second, pair_count
        Fixed buffers and the number of active entries.
    """
    n = centers.shape[0]
    empty_first = jnp.full(max_pairs, -1, dtype=jnp.int32)
    empty_second = jnp.full(max_pairs, -1, dtype=jnp.int32)
    empty_count = jnp.int32(0)
    if n == 0:
        return empty_first, empty_second, empty_count

    cells = jnp.floor(centers / cell_size).astype(jnp.int32)
    keys = hash_cell_coords(cells)
    order = jnp.argsort(keys)
    sorted_keys = keys[order]
    sorted_indices = order.astype(jnp.int32)
    seg_lo, seg_hi = _segment_bounds(sorted_keys)
    neighbor_offsets = _neighbor_offsets_without_origin()

    def collect_for_position(pos: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _position_candidates(
            pos=pos,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            sorted_keys=sorted_keys,
            sorted_indices=sorted_indices,
            cells=cells,
            max_cell_occ=max_cell_occ,
            neighbor_offsets=neighbor_offsets,
        )

    positions = jnp.arange(n, dtype=jnp.int32)
    first, second, valid = jax.vmap(collect_for_position)(positions)
    first = first.reshape(-1)
    second = second.reshape(-1)
    valid = valid.reshape(-1)

    cross_rod = rod_ids[first] != rod_ids[second]
    half_lengths = 0.5 * lengths
    overlap = capsule_aabb_overlap(
        centers[first],
        axes[first],
        half_lengths[first],
        radii[first],
        centers[second],
        axes[second],
        half_lengths[second],
        radii[second],
    )
    valid = valid & cross_rod & overlap
    return _compact_pair_buffer(first, second, valid, max_pairs=max_pairs)


def rebuild_all_pairs_jax(
    *,
    centers: jax.Array,
    rod_ids: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rebuild a bounded cross-rod pair list by all-pairs AABB testing on device.

    This mirrors PyElastica's non-hashed rod-rod rung: every unordered element
    pair from distinct rods is considered and kept only if AABBs overlap.

    Parameters
    ----------
    centers, rod_ids, axes, lengths, radii
        Per-capsule geometry aligned on the leading axis ``N``.
    max_pairs : int
        Fixed-size output buffer capacity.

    Returns
    -------
    pair_first, pair_second, pair_count
        Fixed buffers and the number of active entries.
    """
    n = centers.shape[0]
    empty_first = jnp.full(max_pairs, -1, dtype=jnp.int32)
    empty_second = jnp.full(max_pairs, -1, dtype=jnp.int32)
    empty_count = jnp.int32(0)
    if n < 2:
        return empty_first, empty_second, empty_count

    first_grid, second_grid = jnp.triu_indices(n, k=1)
    cross_rod = rod_ids[first_grid] != rod_ids[second_grid]
    half_lengths = 0.5 * lengths
    overlap = capsule_aabb_overlap(
        centers[first_grid],
        axes[first_grid],
        half_lengths[first_grid],
        radii[first_grid],
        centers[second_grid],
        axes[second_grid],
        half_lengths[second_grid],
        radii[second_grid],
    )
    valid = cross_rod & overlap
    return _compact_pair_buffer(
        first_grid.astype(jnp.int32),
        second_grid.astype(jnp.int32),
        valid,
        max_pairs=max_pairs,
    )
