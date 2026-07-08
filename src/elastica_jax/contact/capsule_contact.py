"""Capsule contact operators and block metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from elastica_jax.block_operation import CommunicationScope, NoBlockOpJax
from elastica_jax.contact.kernels import apply_capsule_pair_forces, apply_wall_contacts
from elastica_jax.contact.spatial_hash import (
    default_cell_size,
    estimate_all_cross_rod_pairs,
    estimate_max_pairs,
    rebuild_all_pairs,
    rebuild_spatial_hash_pairs,
)
from elastica_jax.contact.spatial_hash_jax import (
    rebuild_all_pairs_jax,
    rebuild_spatial_hash_pairs_jax,
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
    assert np.all(
        widths == widths[0]
    ), "Capsule contact requires uniform element counts across rods."
    assert (
        int(widths[0]) == n_elements_per_rod
    ), "n_elements_per_rod must match the packed block width."
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


def rebuild_contact_pairs_jax(
    *,
    centers: jax.Array,
    rod_ids: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    cell_size: float,
    max_pairs: int,
    max_cell_occ: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rebuild bounded pair buffers via on-device spatial hash."""
    return rebuild_spatial_hash_pairs_jax(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        cell_size=cell_size,
        max_pairs=max_pairs,
        max_cell_occ=max_cell_occ,
    )


def rebuild_contact_pairs_all_pairs_jax(
    *,
    centers: jax.Array,
    rod_ids: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rebuild bounded pair buffers via on-device all-pairs AABB testing."""
    return rebuild_all_pairs_jax(
        centers=centers,
        rod_ids=rod_ids,
        axes=axes,
        lengths=lengths,
        radii=radii,
        max_pairs=max_pairs,
    )


def _rebuild_pairs_numpy(
    centers,
    rod_ids,
    axes,
    lengths,
    radii,
    cell_size,
    max_pairs,
) -> tuple[np.ndarray, np.ndarray, np.int32]:
    buffer = rebuild_spatial_hash_pairs(
        centers=np.asarray(centers),
        rod_ids=np.asarray(rod_ids),
        axes=np.asarray(axes),
        lengths=np.asarray(lengths),
        radii=np.asarray(radii),
        cell_size=float(cell_size),
        max_pairs=int(max_pairs),
    )
    return buffer.pair_first, buffer.pair_second, np.int32(buffer.pair_count)


def _rebuild_all_pairs_numpy(
    centers,
    rod_ids,
    axes,
    lengths,
    radii,
    max_pairs,
) -> tuple[np.ndarray, np.ndarray, np.int32]:
    buffer = rebuild_all_pairs(
        centers=np.asarray(centers),
        rod_ids=np.asarray(rod_ids),
        axes=np.asarray(axes),
        lengths=np.asarray(lengths),
        radii=np.asarray(radii),
        max_pairs=int(max_pairs),
    )
    return buffer.pair_first, buffer.pair_second, np.int32(buffer.pair_count)


def rebuild_contact_pairs_host_jax(
    *,
    centers: jax.Array,
    rod_ids: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    cell_size: float,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rebuild bounded pair buffers via host spatial-hash callback."""
    result_shape = (
        jax.ShapeDtypeStruct((max_pairs,), jnp.int32),
        jax.ShapeDtypeStruct((max_pairs,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    pair_first, pair_second, pair_count = jax.pure_callback(
        _rebuild_pairs_numpy,
        result_shape,
        centers,
        rod_ids,
        axes,
        lengths,
        radii,
        cell_size,
        max_pairs,
        vmap_method="sequential",
    )
    return pair_first, pair_second, pair_count


def rebuild_contact_pairs_all_pairs_host_jax(
    *,
    centers: jax.Array,
    rod_ids: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Rebuild bounded pair buffers via host all-pairs AABB callback."""
    result_shape = (
        jax.ShapeDtypeStruct((max_pairs,), jnp.int32),
        jax.ShapeDtypeStruct((max_pairs,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    pair_first, pair_second, pair_count = jax.pure_callback(
        _rebuild_all_pairs_numpy,
        result_shape,
        centers,
        rod_ids,
        axes,
        lengths,
        radii,
        max_pairs,
        vmap_method="sequential",
    )
    return pair_first, pair_second, pair_count


def _system_uses_cuda_backend(system: object) -> bool:
    """Return whether the configured block executes on CUDA devices."""

    candidate_devices: list[object] = []
    if hasattr(system, "_devices"):
        candidate_devices.extend(list(getattr(system, "_devices")))
    if hasattr(system, "_device"):
        candidate_devices.append(getattr(system, "_device"))
    if hasattr(system, "_initial_device"):
        candidate_devices.append(getattr(system, "_initial_device"))
    if hasattr(system, "_primary_block"):
        primary = getattr(system, "_primary_block")
        if hasattr(primary, "_initial_device"):
            candidate_devices.append(getattr(primary, "_initial_device"))

    for device in candidate_devices:
        platform = str(getattr(device, "platform", "")).lower()
        if platform in {"cuda", "gpu"}:
            return True
    return False


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


def _apply_capsule_contact_unified(
    op: CapsuleContactOp,
    state: dict[str, Any],
    time: jax.Array,
) -> dict[str, Any]:
    kinematics = capsule_kinematics_from_block_state(state, op.metadata)
    detection_interval = op.steps_between_detection * op.time_step
    detection_due = (detection_interval == 0.0) | (
        time - state[CONTACT_STATE_LAST_DETECTION_TIME] >= detection_interval
    )

    pair_first = state[CONTACT_STATE_PAIR_FIRST]
    pair_second = state[CONTACT_STATE_PAIR_SECOND]
    pair_count = state[CONTACT_STATE_PAIR_COUNT]

    def _rebuild(_):
        if op.metadata.broad_phase == "all_pairs":
            if op._use_device_spatial_hash:
                return rebuild_contact_pairs_all_pairs_jax(
                    centers=kinematics["centers"],
                    rod_ids=op._rod_ids,
                    axes=kinematics["axes"],
                    lengths=kinematics["lengths"],
                    radii=kinematics["radii"],
                    max_pairs=op.metadata.max_pairs,
                )
            return rebuild_contact_pairs_all_pairs_host_jax(
                centers=kinematics["centers"],
                rod_ids=op._rod_ids,
                axes=kinematics["axes"],
                lengths=kinematics["lengths"],
                radii=kinematics["radii"],
                max_pairs=op.metadata.max_pairs,
            )
        if op._use_device_spatial_hash:
            return rebuild_contact_pairs_jax(
                centers=kinematics["centers"],
                rod_ids=op._rod_ids,
                axes=kinematics["axes"],
                lengths=kinematics["lengths"],
                radii=kinematics["radii"],
                cell_size=op.metadata.cell_size,
                max_pairs=op.metadata.max_pairs,
                max_cell_occ=op.metadata.max_neighbors_per_capsule,
            )
        return rebuild_contact_pairs_host_jax(
            centers=kinematics["centers"],
            rod_ids=op._rod_ids,
            axes=kinematics["axes"],
            lengths=kinematics["lengths"],
            radii=kinematics["radii"],
            cell_size=op.metadata.cell_size,
            max_pairs=op.metadata.max_pairs,
        )

    def _keep(_):
        return pair_first, pair_second, pair_count

    pair_first, pair_second, pair_count = jax.lax.cond(
        detection_due, _rebuild, _keep, operand=None
    )

    pair_slots = jnp.arange(op.metadata.max_pairs, dtype=jnp.int32)
    pair_active = pair_slots < pair_count

    if op.contact_stiffness_initial is not None:
        contact_stiffness = jnp.where(
            time < op.stiffness_ramp_time,
            op.contact_stiffness_initial,
            op.contact_stiffness,
        )
    else:
        contact_stiffness = op.contact_stiffness
    if op.contact_damping_initial is not None:
        contact_damping = jnp.where(
            time < op.stiffness_ramp_time,
            op.contact_damping_initial,
            op.contact_damping,
        )
    else:
        contact_damping = op.contact_damping
    friction_gate = jnp.where(time >= op.friction_start_time, 1.0, 0.0)

    external_forces = state["external_forces"]
    external_torques = state["external_torques"]
    (
        external_forces,
        external_torques,
        candidate_mask,
        last_detection_time,
    ) = apply_capsule_pair_forces(
        pair_first=pair_first,
        pair_second=pair_second,
        pair_active=pair_active,
        centers=kinematics["centers"],
        velocities=kinematics["velocities"],
        axes=kinematics["axes"],
        lengths=kinematics["lengths"],
        radii=kinematics["radii"],
        omega=kinematics["omega"],
        directors=kinematics["directors"],
        block_element_indices=kinematics["block_element_indices"],
        external_forces=external_forces,
        external_torques=external_torques,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
        cached_candidates=state[CONTACT_STATE_CANDIDATE_MASK],
        last_detection_time=state[CONTACT_STATE_LAST_DETECTION_TIME],
        time=time,
        steps_between_detection=op.steps_between_detection,
        time_step=op.time_step,
        hertzian=op.hertzian,
        friction_coefficient=op.friction_coefficient,
        static_velocity_threshold=op.static_velocity_threshold,
        friction_gate=friction_gate,
    )
    updated = dict(state)
    updated["external_forces"] = external_forces
    updated["external_torques"] = external_torques
    updated[CONTACT_STATE_PAIR_FIRST] = pair_first
    updated[CONTACT_STATE_PAIR_SECOND] = pair_second
    updated[CONTACT_STATE_PAIR_COUNT] = pair_count
    updated[CONTACT_STATE_CANDIDATE_MASK] = candidate_mask
    updated[CONTACT_STATE_LAST_DETECTION_TIME] = last_detection_time
    return updated


class CapsuleContactOp(NoBlockOpJax):
    """Capsule-capsule contact for packed Cosserat rod blocks.

    Broad-phase candidates are collected by either a spatial hash
    (``broad_phase="spatial_hash"``, default) or by all-pairs AABB testing
    (``broad_phase="all_pairs"``), which mirrors PyElastica's non-hashed
    rod-rod contact registration style.

    By default the contact is the linear normal-spring/damper law. Optional
    arguments enable a Hertzian ``gamma^1.5`` normal law (``hertzian=True``), a
    time-ramped soft-to-hard stiffness (``contact_stiffness_initial`` /
    ``contact_damping_initial`` / ``stiffness_ramp_time``), and isotropic kinetic
    Coulomb friction activated after ``friction_start_time``. These reproduce the
    C++ nest-simulator contact model while leaving the default behaviour intact
    for other cases.
    """

    communication_scope = CommunicationScope.HALO_READ

    def __init__(
        self,
        *,
        metadata: BlockCapsuleMetadata | None = None,
        n_elements_per_rod: int | None = None,
        contact_stiffness: float,
        contact_damping: float,
        steps_between_detection: int = 0,
        time_step: float,
        max_neighbors_per_capsule: int = 64,
        broad_phase: str = "spatial_hash",
        hertzian: bool = False,
        contact_stiffness_initial: float | None = None,
        contact_damping_initial: float | None = None,
        stiffness_ramp_time: float = 0.0,
        friction_coefficient: float = 0.0,
        static_velocity_threshold: float = 1.0,
        friction_start_time: float = float("inf"),
        _system=None,
    ) -> None:
        assert (
            _system is not None
        ), "CapsuleContactOp requires a finalized block system."
        assert broad_phase in {"spatial_hash", "all_pairs"}, (
            f"Unsupported broad_phase {broad_phase!r}; "
            "expected 'spatial_hash' or 'all_pairs'."
        )
        if metadata is None:
            assert (
                n_elements_per_rod is not None
            ), "Provide metadata or n_elements_per_rod for CapsuleContactOp."
            metadata = build_block_capsule_metadata(
                _system,
                n_elements_per_rod=n_elements_per_rod,
                max_neighbors_per_capsule=max_neighbors_per_capsule,
                broad_phase=broad_phase,
            )
        else:
            assert metadata.broad_phase == broad_phase, (
                "Provided metadata.broad_phase does not match CapsuleContactOp "
                f"broad_phase={broad_phase!r}."
            )
        self._block = _system
        self.metadata = metadata
        self.contact_stiffness = contact_stiffness
        self.contact_damping = contact_damping
        self.steps_between_detection = steps_between_detection
        self.time_step = time_step
        self.hertzian = hertzian
        self.contact_stiffness_initial = contact_stiffness_initial
        self.contact_damping_initial = contact_damping_initial
        self.stiffness_ramp_time = stiffness_ramp_time
        self.friction_coefficient = friction_coefficient
        self.static_velocity_threshold = static_velocity_threshold
        self.friction_start_time = friction_start_time
        self._rod_ids = jnp.asarray(metadata.rod_ids)
        self._use_device_spatial_hash = _system_uses_cuda_backend(_system)

    def jax_block_operate_synchronize(self, state, time):
        if state.get(SHARDED_STATE_KEY, False):
            from elastica_jax.memory_block.sharded_cosserat_rod_jax import (
                _ShardedCosseratRodBlock,
            )

            assert isinstance(
                self._block, _ShardedCosseratRodBlock
            ), "Sharded capsule contact requires a _ShardedCosseratRodBlock system."
            merged = self._block.merge_shard_states(state)
            for key in CONTACT_STATE_KEYS:
                merged[key] = state["shards"][0][key]
            merged = _apply_capsule_contact_unified(self, merged, time)
            return self._block.scatter_merged_state(merged, state)
        return _apply_capsule_contact_unified(self, state, time)


def _apply_wall_contact_unified(
    op: WallContactOp,
    state: dict[str, Any],
    time: jax.Array,
) -> dict[str, Any]:
    del time
    kinematics = capsule_kinematics_from_block_state(state, op.metadata)
    external_forces, external_torques = apply_wall_contacts(
        wall_origins=op.wall_origins,
        wall_normals=op.wall_normals,
        centers=kinematics["centers"],
        velocities=kinematics["velocities"],
        axes=kinematics["axes"],
        lengths=kinematics["lengths"],
        radii=kinematics["radii"],
        omega=kinematics["omega"],
        directors=kinematics["directors"],
        block_element_indices=kinematics["block_element_indices"],
        external_forces=state["external_forces"],
        external_torques=state["external_torques"],
        contact_stiffness=op.contact_stiffness,
        contact_damping=op.contact_damping,
    )
    updated = dict(state)
    updated["external_forces"] = external_forces
    updated["external_torques"] = external_torques
    return updated


class WallContactOp(NoBlockOpJax):
    """Half-space wall contact for packed capsule elements."""

    communication_scope = CommunicationScope.LOCAL

    def __init__(
        self,
        *,
        metadata: BlockCapsuleMetadata | None = None,
        n_elements_per_rod: int | None = None,
        wall_origins: np.ndarray,
        wall_normals: np.ndarray,
        contact_stiffness: float,
        contact_damping: float,
        max_neighbors_per_capsule: int = 64,
        _system=None,
    ) -> None:
        assert _system is not None, "WallContactOp requires a finalized block system."
        if metadata is None:
            assert (
                n_elements_per_rod is not None
            ), "Provide metadata or n_elements_per_rod for WallContactOp."
            metadata = build_block_capsule_metadata(
                _system,
                n_elements_per_rod=n_elements_per_rod,
                max_neighbors_per_capsule=max_neighbors_per_capsule,
            )
        self._block = _system
        self.metadata = metadata
        self.wall_origins = np.asarray(wall_origins, dtype=np.float64)
        self.wall_normals = np.asarray(wall_normals, dtype=np.float64)
        self.contact_stiffness = contact_stiffness
        self.contact_damping = contact_damping

    def jax_block_operate_synchronize(self, state, time):
        if state.get(SHARDED_STATE_KEY, False):
            from elastica_jax.memory_block.sharded_cosserat_rod_jax import (
                _ShardedCosseratRodBlock,
            )

            assert isinstance(
                self._block, _ShardedCosseratRodBlock
            ), "Sharded wall contact requires a _ShardedCosseratRodBlock system."
            merged = self._block.merge_shard_states(state)
            merged = _apply_wall_contact_unified(self, merged, time)
            return self._block.scatter_merged_state(merged, state)
        return _apply_wall_contact_unified(self, state, time)
