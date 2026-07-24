"""Capsule layout and contact-state helpers for packed and stacked blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from elastica_jax.contact.spatial_hash import (
    default_cell_size,
    estimate_all_cross_rod_pairs,
    estimate_max_pairs,
)

CONTACT_STATE_PAIR_FIRST = "capsule_contact_pair_first"
CONTACT_STATE_PAIR_SECOND = "capsule_contact_pair_second"
CONTACT_STATE_PAIR_COUNT = "capsule_contact_pair_count"
CONTACT_STATE_CANDIDATE_MASK = "capsule_contact_candidate_mask"
CONTACT_STATE_LAST_DETECTION_TIME = "capsule_contact_last_detection_time"

CONTACT_STATE_KEYS = (
    CONTACT_STATE_PAIR_FIRST,
    CONTACT_STATE_PAIR_SECOND,
    CONTACT_STATE_PAIR_COUNT,
    CONTACT_STATE_CANDIDATE_MASK,
    CONTACT_STATE_LAST_DETECTION_TIME,
)

type CapsuleLayout = Literal["packed", "stacked"]


@runtime_checkable
class CapsuleRodBlock(Protocol):
    """Minimal block face needed to build capsule contact metadata."""

    n_rods: int
    radius: np.ndarray
    lengths: np.ndarray
    start_idx_in_rod_elems: np.ndarray
    end_idx_in_rod_elems: np.ndarray
    start_idx_in_rod_nodes: np.ndarray

    def jax_get_state(self) -> dict[str, jax.Array]: ...

    def jax_set_state(self, state: dict[str, jax.Array]) -> None: ...


@dataclass(frozen=True)
class BlockCapsuleMetadata:
    """Per-element capsule layout for packed or stacked Cosserat rod blocks.

    Parameters
    ----------
    layout :
        ``"packed"`` indexes into flat horizontal block arrays; ``"stacked"``
        indexes with ``(rod_id, local_element)`` into vertical block arrays.
    """

    n_rods: int
    n_elements_per_rod: int
    element_indices: np.ndarray
    node_indices: np.ndarray
    rod_ids: np.ndarray
    block_element_indices: np.ndarray
    max_pairs: int
    cell_size: float
    max_neighbors_per_capsule: int = 64
    broad_phase: str = "spatial_hash"
    layout: CapsuleLayout = "packed"

    @property
    def n_capsules(self) -> int:
        return int(self.rod_ids.size)


def _detect_capsule_layout(block: CapsuleRodBlock) -> CapsuleLayout:
    radius = np.asarray(block.radius)
    if radius.ndim == 2:
        return "stacked"
    assert radius.ndim == 1, (
        f"Unsupported radius ndim={radius.ndim}; expected 1 (packed) or 2 (stacked)."
    )
    return "packed"


def build_block_capsule_metadata(
    block: CapsuleRodBlock,
    *,
    n_elements_per_rod: int,
    cell_size: float | None = None,
    max_neighbors_per_capsule: int = 64,
    broad_phase: str = "spatial_hash",
) -> BlockCapsuleMetadata:
    """Build capsule indexing metadata for a packed or stacked rod block.

    Parameters
    ----------
    block :
        Finalized Cosserat rod memory block (horizontal packed or vertical
        stacked). Element counts must be uniform across rods.
    n_elements_per_rod :
        Elements per rod; must match the block's uniform rod width.
    cell_size :
        Spatial-hash cell size. When omitted, derived from element radii and
        lengths.
    max_neighbors_per_capsule :
        Bound used to size the spatial-hash pair buffer.
    broad_phase :
        ``"spatial_hash"`` or ``"all_pairs"``.

    Returns
    -------
    BlockCapsuleMetadata
        Layout-aware capsule indices and broad-phase sizing.
    """
    assert broad_phase in {"spatial_hash", "all_pairs"}, (
        f"Unsupported broad_phase {broad_phase!r}; "
        "expected 'spatial_hash' or 'all_pairs'."
    )
    widths = block.end_idx_in_rod_elems - block.start_idx_in_rod_elems
    assert np.all(widths == widths[0]), (
        "Capsule contact requires uniform element counts across rods."
    )
    assert int(widths[0]) == n_elements_per_rod, (
        "n_elements_per_rod must match each rod's element count."
    )
    n_rods = int(block.n_rods)
    layout = _detect_capsule_layout(block)
    offsets = np.arange(n_elements_per_rod, dtype=np.int32)
    node_offsets = np.arange(n_elements_per_rod + 1, dtype=np.int32)
    if layout == "stacked":
        element_indices = np.broadcast_to(offsets, (n_rods, n_elements_per_rod)).copy()
        node_indices = np.broadcast_to(
            node_offsets, (n_rods, n_elements_per_rod + 1)
        ).copy()
    else:
        element_indices = (
            block.start_idx_in_rod_elems[:, None].astype(np.int32) + offsets[None, :]
        )
        node_indices = (
            block.start_idx_in_rod_nodes[:, None].astype(np.int32)
            + node_offsets[None, :]
        )
    rod_ids = np.repeat(np.arange(n_rods, dtype=np.int32), n_elements_per_rod)
    block_element_indices = element_indices.reshape(-1)
    n_capsules = rod_ids.size
    if broad_phase == "all_pairs":
        max_pairs = max(1, estimate_all_cross_rod_pairs(rod_ids))
    else:
        max_pairs = estimate_max_pairs(
            n_capsules, max_neighbors_per_capsule=max_neighbors_per_capsule
        )
    if cell_size is None:
        if layout == "stacked":
            cell_size = default_cell_size(
                radii=np.asarray(block.radius).reshape(-1),
                lengths=np.asarray(block.lengths).reshape(-1),
            )
        else:
            cell_size = default_cell_size(
                radii=np.asarray(block.radius)[block_element_indices],
                lengths=np.asarray(block.lengths)[block_element_indices],
            )
    return BlockCapsuleMetadata(
        n_rods=n_rods,
        n_elements_per_rod=n_elements_per_rod,
        element_indices=element_indices,
        node_indices=node_indices,
        rod_ids=rod_ids,
        block_element_indices=block_element_indices,
        max_pairs=max_pairs,
        cell_size=float(cell_size),
        max_neighbors_per_capsule=max_neighbors_per_capsule,
        broad_phase=broad_phase,
        layout=layout,
    )


def initialize_capsule_contact_state(
    metadata: BlockCapsuleMetadata,
    *,
    device: jax.Device | None,
    dtype: np.dtype,
) -> dict[str, jax.Array]:
    max_pairs = metadata.max_pairs
    return {
        CONTACT_STATE_PAIR_FIRST: jax.device_put(
            np.full(max_pairs, -1, dtype=np.int32), device=device
        ),
        CONTACT_STATE_PAIR_SECOND: jax.device_put(
            np.full(max_pairs, -1, dtype=np.int32), device=device
        ),
        CONTACT_STATE_PAIR_COUNT: jax.device_put(np.int32(0), device=device),
        CONTACT_STATE_CANDIDATE_MASK: jax.device_put(
            np.zeros(max_pairs, dtype=bool), device=device
        ),
        CONTACT_STATE_LAST_DETECTION_TIME: jax.device_put(
            np.array(-np.inf, dtype=dtype), device=device
        ),
    }


def install_capsule_contact_state(
    block: CapsuleRodBlock,
    metadata: BlockCapsuleMetadata,
    *,
    device: jax.Device | None,
    dtype: np.dtype,
) -> None:
    contact_state = initialize_capsule_contact_state(
        metadata, device=device, dtype=dtype
    )
    state = block.jax_get_state()
    block.jax_set_state({**state, **contact_state})


def _kinematics_packed(
    state: dict[str, Any],
    metadata: BlockCapsuleMetadata,
) -> dict[str, jax.Array]:
    elem = jnp.asarray(metadata.block_element_indices)
    nodes = jnp.asarray(metadata.node_indices)
    positions = state["position_collection"][:, nodes]
    velocities = state["velocity_collection"][:, nodes]
    masses = state["mass"][nodes]
    centers = 0.5 * (positions[:, :, :-1] + positions[:, :, 1:])
    numerator = masses[:, :-1][None, :, :] * velocities[:, :, :-1]
    numerator += masses[:, 1:][None, :, :] * velocities[:, :, 1:]
    element_velocity = numerator / (masses[:, :-1] + masses[:, 1:])[None, :, :]
    centers = jnp.moveaxis(centers, 0, -1).reshape(-1, 3)
    element_velocity = jnp.moveaxis(element_velocity, 0, -1).reshape(-1, 3)
    axes = state["tangents"][:, elem].T
    lengths = state["lengths"][elem]
    radii = state["radius"][elem]
    directors = jnp.moveaxis(state["director_collection"][:, :, elem], 2, 0)
    omega_material = state["omega_collection"][:, elem].T
    omega_world = jnp.einsum("nji,nj->ni", directors, omega_material)
    return {
        "centers": centers,
        "velocities": element_velocity,
        "axes": axes,
        "lengths": lengths,
        "radii": radii,
        "directors": directors,
        "omega": omega_world,
        "block_element_indices": elem,
        "rod_ids": jnp.asarray(metadata.rod_ids),
    }


def _kinematics_stacked(
    state: dict[str, Any],
    metadata: BlockCapsuleMetadata,
) -> dict[str, jax.Array]:
    """Gather flat capsule kinematics from stacked ``(n_rods, ...)`` state."""
    positions = state["position_collection"]
    velocities = state["velocity_collection"]
    masses = state["mass"]
    centers = 0.5 * (positions[:, :, :-1] + positions[:, :, 1:])
    mass_left = masses[:, :-1]
    mass_right = masses[:, 1:]
    numerator = mass_left[:, None, :] * velocities[:, :, :-1]
    numerator = numerator + mass_right[:, None, :] * velocities[:, :, 1:]
    element_velocity = numerator / (mass_left + mass_right)[:, None, :]
    centers = jnp.moveaxis(centers, 1, -1).reshape(-1, 3)
    element_velocity = jnp.moveaxis(element_velocity, 1, -1).reshape(-1, 3)
    axes = jnp.moveaxis(state["tangents"], 1, -1).reshape(-1, 3)
    lengths = state["lengths"].reshape(-1)
    radii = state["radius"].reshape(-1)
    directors = jnp.moveaxis(state["director_collection"], 3, 1).reshape(-1, 3, 3)
    omega_material = jnp.moveaxis(state["omega_collection"], 1, -1).reshape(-1, 3)
    omega_world = jnp.einsum("nji,nj->ni", directors, omega_material)
    return {
        "centers": centers,
        "velocities": element_velocity,
        "axes": axes,
        "lengths": lengths,
        "radii": radii,
        "directors": directors,
        "omega": omega_world,
        "block_element_indices": jnp.asarray(metadata.block_element_indices),
        "rod_ids": jnp.asarray(metadata.rod_ids),
    }


def capsule_kinematics_from_block_state(
    state: dict[str, Any],
    metadata: BlockCapsuleMetadata,
) -> dict[str, jax.Array]:
    """Extract flat per-capsule kinematics from packed or stacked block state."""
    if metadata.layout == "stacked":
        return _kinematics_stacked(state, metadata)
    return _kinematics_packed(state, metadata)
