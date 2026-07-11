"""Host-side spatial-hash broad phase (reference / test implementation).

Runtime capsule contact uses the device path in ``spatial_hash_jax``.
This module remains for unit tests and host-side pair-count estimates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
)


@dataclass(frozen=True)
class SpatialHashPairBuffer:
    """Fixed-capacity pair list produced by the spatial hash broad phase."""

    pair_first: np.ndarray
    pair_second: np.ndarray
    pair_count: int
    max_pairs: int

    @property
    def pair_active(self) -> np.ndarray:
        count = np.arange(self.max_pairs, dtype=np.int32)
        return count < self.pair_count


def default_cell_size(*, radii: np.ndarray, lengths: np.ndarray) -> float:
    return float(2.0 * np.max(radii) + np.max(lengths))


def estimate_max_pairs(
    n_capsules: int,
    *,
    max_neighbors_per_capsule: int = 64,
) -> int:
    assert n_capsules >= 0, "n_capsules must be nonnegative."
    assert max_neighbors_per_capsule > 0, "max_neighbors_per_capsule must be positive."
    return int(
        min(n_capsules * max_neighbors_per_capsule, n_capsules * (n_capsules - 1) // 2)
    )


def estimate_all_cross_rod_pairs(rod_ids: np.ndarray) -> int:
    """Return the number of unordered element pairs from distinct rods.

    This matches the PyElastica rod-rod registration cost for a packed block:
    every cross-rod element pair may enter the broad phase.
    """
    rod_ids = np.asarray(rod_ids)
    n_capsules = int(rod_ids.size)
    assert n_capsules >= 0, "rod_ids must be nonempty or empty."
    if n_capsules < 2:
        return 0
    _, counts = np.unique(rod_ids, return_counts=True)
    same_rod = int(np.sum(counts * (counts - 1) // 2))
    return int(n_capsules * (n_capsules - 1) // 2 - same_rod)


def rebuild_all_pairs(
    *,
    centers: np.ndarray,
    rod_ids: np.ndarray,
    axes: np.ndarray,
    lengths: np.ndarray,
    radii: np.ndarray,
    max_pairs: int,
) -> SpatialHashPairBuffer:
    """Build a bounded cross-rod pair list by all-pairs AABB testing.

    This mirrors PyElastica's non-hashed rung: every cross-rod candidate is
    considered, then pruned by bounding-box overlap (no spatial hash).
    """
    assert centers.ndim == 2 and centers.shape[1] == 3, "centers must have shape (N, 3)."
    n_capsules = centers.shape[0]
    assert (
        rod_ids.shape == (n_capsules,)
        and axes.shape == (n_capsules, 3)
        and lengths.shape == (n_capsules,)
        and radii.shape == (n_capsules,)
    ), "Capsule metadata must be aligned with centers."
    assert max_pairs > 0, "max_pairs must be positive."

    pair_first: list[int] = []
    pair_second: list[int] = []
    half_lengths = 0.5 * lengths
    for first_index in range(n_capsules):
        for second_index in range(first_index + 1, n_capsules):
            if rod_ids[first_index] == rod_ids[second_index]:
                continue
            if not _capsule_aabb_overlap(
                centers[first_index],
                axes[first_index],
                half_lengths[first_index],
                radii[first_index],
                centers[second_index],
                axes[second_index],
                half_lengths[second_index],
                radii[second_index],
            ):
                continue
            pair_first.append(first_index)
            pair_second.append(second_index)
            if len(pair_first) >= max_pairs:
                break
        if len(pair_first) >= max_pairs:
            break

    assert len(pair_first) <= max_pairs, "All-pairs buffer overflowed."
    buffer_first = np.full(max_pairs, -1, dtype=np.int32)
    buffer_second = np.full(max_pairs, -1, dtype=np.int32)
    pair_count = len(pair_first)
    if pair_count > 0:
        buffer_first[:pair_count] = np.asarray(pair_first, dtype=np.int32)
        buffer_second[:pair_count] = np.asarray(pair_second, dtype=np.int32)
    return SpatialHashPairBuffer(
        pair_first=buffer_first,
        pair_second=buffer_second,
        pair_count=pair_count,
        max_pairs=max_pairs,
    )


def _capsule_aabb_overlap(
    center_a: np.ndarray,
    axis_a: np.ndarray,
    half_length_a: float,
    radius_a: float,
    center_b: np.ndarray,
    axis_b: np.ndarray,
    half_length_b: float,
    radius_b: float,
) -> bool:
    extent_a = half_length_a * np.abs(axis_a) + radius_a
    extent_b = half_length_b * np.abs(axis_b) + radius_b
    return bool(np.all(np.abs(center_a - center_b) <= extent_a + extent_b))


def rebuild_spatial_hash_pairs(
    *,
    centers: np.ndarray,
    rod_ids: np.ndarray,
    axes: np.ndarray,
    lengths: np.ndarray,
    radii: np.ndarray,
    cell_size: float,
    max_pairs: int,
) -> SpatialHashPairBuffer:
    """Rebuild a bounded cross-rod capsule pair list from capsule geometry."""
    assert (
        centers.ndim == 2 and centers.shape[1] == 3
    ), "centers must have shape (N, 3)."
    n_capsules = centers.shape[0]
    assert (
        rod_ids.shape == (n_capsules,)
        and axes.shape == (n_capsules, 3)
        and lengths.shape == (n_capsules,)
        and radii.shape == (n_capsules,)
    ), "Capsule metadata must be aligned with centers."
    assert cell_size > 0.0, "cell_size must be positive."
    assert max_pairs > 0, "max_pairs must be positive."

    pair_first: list[int] = []
    pair_second: list[int] = []
    cells = np.floor(centers / cell_size).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index in range(n_capsules):
        buckets[tuple(cells[index])].append(index)

    half_lengths = 0.5 * lengths

    def try_add_pair(first_index: int, second_index: int) -> bool:
        if first_index >= second_index:
            return False
        if rod_ids[first_index] == rod_ids[second_index]:
            return False
        if not _capsule_aabb_overlap(
            centers[first_index],
            axes[first_index],
            half_lengths[first_index],
            radii[first_index],
            centers[second_index],
            axes[second_index],
            half_lengths[second_index],
            radii[second_index],
        ):
            return False
        pair_first.append(first_index)
        pair_second.append(second_index)
        return len(pair_first) >= max_pairs

    for cell_key, indices in buckets.items():
        for local_a in range(len(indices)):
            for local_b in range(local_a + 1, len(indices)):
                if try_add_pair(indices[local_a], indices[local_b]):
                    break
            if len(pair_first) >= max_pairs:
                break
        if len(pair_first) >= max_pairs:
            break

        for offset in _NEIGHBOR_OFFSETS:
            if offset == (0, 0, 0):
                continue
            neighbor_key = (
                cell_key[0] + offset[0],
                cell_key[1] + offset[1],
                cell_key[2] + offset[2],
            )
            neighbor_indices = buckets.get(neighbor_key)
            if neighbor_indices is None:
                continue
            for left in indices:
                for right in neighbor_indices:
                    if try_add_pair(min(left, right), max(left, right)):
                        break
                if len(pair_first) >= max_pairs:
                    break
            if len(pair_first) >= max_pairs:
                break
        if len(pair_first) >= max_pairs:
            break

    assert len(pair_first) <= max_pairs, "Spatial hash pair buffer overflowed."
    buffer_first = np.full(max_pairs, -1, dtype=np.int32)
    buffer_second = np.full(max_pairs, -1, dtype=np.int32)
    pair_count = len(pair_first)
    if pair_count > 0:
        buffer_first[:pair_count] = np.asarray(pair_first, dtype=np.int32)
        buffer_second[:pair_count] = np.asarray(pair_second, dtype=np.int32)
    return SpatialHashPairBuffer(
        pair_first=buffer_first,
        pair_second=buffer_second,
        pair_count=pair_count,
        max_pairs=max_pairs,
    )
