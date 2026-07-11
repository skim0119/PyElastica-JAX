"""Packed-block capsule layout and contact-state helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from elastica_jax.contact.spatial_hash import (
    default_cell_size,
    estimate_all_cross_rod_pairs,
    estimate_max_pairs,
)
from elastica_jax.memory_block.memory_block_rod_jax import _CosseratRodMemoryBlock
from elastica_jax.memory_block.sharded_cosserat_rod_jax import (
    SHARDED_STATE_KEY,
)

if TYPE_CHECKING:
    from elastica_jax.memory_block.sharded_cosserat_rod_jax import (
        _ShardedCosseratRodBlock,
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


@dataclass(frozen=True)
class BlockCapsuleMetadata:
    """Packed block layout for per-element capsule contact."""

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

    @property
    def n_capsules(self) -> int:
        return int(self.rod_ids.size)


def build_block_capsule_metadata(
    block: _CosseratRodMemoryBlock | _ShardedCosseratRodBlock,
    *,
    n_elements_per_rod: int,
    cell_size: float | None = None,
    max_neighbors_per_capsule: int = 64,
    broad_phase: str = "spatial_hash",
) -> BlockCapsuleMetadata:
    assert broad_phase in {"spatial_hash", "all_pairs"}, (
        f"Unsupported broad_phase {broad_phase!r}; "
        "expected 'spatial_hash' or 'all_pairs'."
    )
    widths = block.end_idx_in_rod_elems - block.start_idx_in_rod_elems
    assert np.all(widths == widths[0]), (
        "Capsule contact requires uniform element counts across rods."
    )
    assert int(widths[0]) == n_elements_per_rod, (
        "n_elements_per_rod must match the packed block width."
    )
    n_rods = int(block.n_rods)
    offsets = np.arange(n_elements_per_rod, dtype=np.int32)
    element_indices = (
        block.start_idx_in_rod_elems[:, None].astype(np.int32) + offsets[None, :]
    )
    node_indices = (
        block.start_idx_in_rod_nodes[:, None].astype(np.int32)
        + np.arange(n_elements_per_rod + 1, dtype=np.int32)[None, :]
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
        cell_size = default_cell_size(
            radii=np.asarray(block.radius[block_element_indices]),
            lengths=np.asarray(block.lengths[block_element_indices]),
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
        CONTACT_STATE_PAIR_COUNT: jax.device_put(
            np.asarray(0, dtype=np.int32), device=device
        ),
        CONTACT_STATE_CANDIDATE_MASK: jax.device_put(
            np.zeros(max_pairs, dtype=bool), device=device
        ),
        CONTACT_STATE_LAST_DETECTION_TIME: jax.device_put(
            np.asarray(-np.inf, dtype=dtype), device=device
        ),
    }


def install_capsule_contact_state(
    block: _CosseratRodMemoryBlock | _ShardedCosseratRodBlock,
    metadata: BlockCapsuleMetadata,
    *,
    device: jax.Device | None,
    dtype: np.dtype,
) -> None:
    contact_state = initialize_capsule_contact_state(
        metadata, device=device, dtype=dtype
    )
    state = block.jax_get_state()
    if state.get(SHARDED_STATE_KEY, False):
        shard_states = list(state["shards"])
        shard_states[0] = {**shard_states[0], **contact_state}
        block.jax_set_state({SHARDED_STATE_KEY: True, "shards": tuple(shard_states)})
        return
    block.jax_set_state({**state, **contact_state})


def capsule_kinematics_from_block_state(
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
    }
