"""MPI halo helpers for cross-rank CapsuleContact (HALO_READ)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import mpi4jax
import numpy as np

from elastica_jax.contact.capsule_metadata import BlockCapsuleMetadata
from elastica_jax.contact.spatial_hash import (
    estimate_all_cross_rod_pairs,
    estimate_max_pairs,
)

type CapsuleKinematics = dict[str, jax.Array]


def inflate_capsule_metadata_for_mpi(
    metadata: BlockCapsuleMetadata,
    *,
    world_size: int,
) -> BlockCapsuleMetadata:
    """Resize pair buffers for the world-wide capsule count after allgather.

    Parameters
    ----------
    metadata :
        Rank-local capsule metadata.
    world_size :
        MPI communicator size. Capsule counts are assumed equal on every rank
        (required by ``mpi4jax.allgather``).

    Returns
    -------
    BlockCapsuleMetadata
        Copy with ``max_pairs`` sized for ``world_size * n_capsules``.
    """
    assert world_size >= 1, "world_size must be positive."
    n_global_capsules = metadata.n_capsules * world_size
    n_global_rods = metadata.n_rods * world_size
    global_rod_ids = np.repeat(
        np.arange(n_global_rods, dtype=np.int32),
        metadata.n_elements_per_rod,
    )
    if metadata.broad_phase == "all_pairs":
        max_pairs = max(1, estimate_all_cross_rod_pairs(global_rod_ids))
    else:
        max_pairs = max(
            1,
            estimate_max_pairs(
                n_global_capsules,
                max_neighbors_per_capsule=metadata.max_neighbors_per_capsule,
            ),
        )
    return replace(metadata, max_pairs=max_pairs)


def _allgather_capsules(local: jax.Array, *, comm: Any) -> jax.Array:
    """Allgather equal-shaped per-rank capsules and flatten the rank axis."""
    gathered = mpi4jax.allgather(local, comm=comm)
    return gathered.reshape((-1, *local.shape[1:]))


def assemble_mpi_halo_kinematics(
    *,
    kinematics: CapsuleKinematics,
    global_rod_indices: jax.Array,
    comm: Any,
    rank: int,
    world_size: int,
) -> tuple[CapsuleKinematics, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Allgather capsule kinematics and build owned scatter bookkeeping.

    Parameters
    ----------
    kinematics :
        Rank-local capsule kinematics from ``capsule_kinematics_from_block_state``.
    global_rod_indices :
        Global rod index for each local rod (length ``n_local_rods``).
    comm, rank, world_size :
        MPI communicator identity.

    Returns
    -------
    tuple
        ``(global_kinematics, owned_mask, scatter_rod_ids, scatter_elem,
        scatter_directors)``. Global kinematics use global rod ids for broad
        phase; scatter arrays are valid only on owned capsules (ghosts are
        zero-filled and combined with ``owned_mask``).
    """
    assert world_size >= 1, "world_size must be positive."
    local_centers = kinematics["centers"]
    n_local = local_centers.shape[0]
    local_rod_ids = kinematics["rod_ids"]
    local_global_rod_ids = global_rod_indices[local_rod_ids]

    global_kin: CapsuleKinematics = {
        "centers": _allgather_capsules(local_centers, comm=comm),
        "velocities": _allgather_capsules(kinematics["velocities"], comm=comm),
        "axes": _allgather_capsules(kinematics["axes"], comm=comm),
        "lengths": _allgather_capsules(kinematics["lengths"], comm=comm),
        "radii": _allgather_capsules(kinematics["radii"], comm=comm),
        "omega": _allgather_capsules(kinematics["omega"], comm=comm),
        "directors": _allgather_capsules(kinematics["directors"], comm=comm),
        "rod_ids": _allgather_capsules(local_global_rod_ids, comm=comm),
        "block_element_indices": kinematics["block_element_indices"],
    }
    n_global = n_local * world_size
    capsule_index = jnp.arange(n_global, dtype=jnp.int32)
    owned_mask = (capsule_index // n_local) == rank
    start = rank * n_local
    end = start + n_local
    scatter_elem = (
        jnp.zeros((n_global,), dtype=jnp.int32)
        .at[start:end]
        .set(kinematics["block_element_indices"])
    )
    scatter_rod_ids = (
        jnp.zeros((n_global,), dtype=jnp.int32).at[start:end].set(local_rod_ids)
    )
    scatter_directors = (
        jnp.zeros((n_global, 3, 3), dtype=local_centers.dtype)
        .at[start:end]
        .set(kinematics["directors"])
    )
    return global_kin, owned_mask, scatter_rod_ids, scatter_elem, scatter_directors
