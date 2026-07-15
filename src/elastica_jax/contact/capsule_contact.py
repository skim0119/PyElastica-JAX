"""Capsule and wall contact operators for packed Cosserat rod blocks."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from elastica_jax.block_operation import CommunicationScope, NoBlockOpJax
from elastica_jax.contact.capsule_metadata import (
    CONTACT_STATE_CANDIDATE_MASK,
    CONTACT_STATE_LAST_DETECTION_TIME,
    CONTACT_STATE_PAIR_COUNT,
    CONTACT_STATE_PAIR_FIRST,
    CONTACT_STATE_PAIR_SECOND,
    BlockCapsuleMetadata,
    build_block_capsule_metadata,
    capsule_kinematics_from_block_state,
)
from elastica_jax.contact.kernels import apply_capsule_pair_forces, apply_wall_contacts
from elastica_jax.contact.spatial_hash_jax import (
    rebuild_all_pairs_jax,
    rebuild_spatial_hash_pairs_jax,
)


def _rebuild_broad_phase_pairs(
    *,
    broad_phase: str,
    centers: jax.Array,
    rod_ids: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    cell_size: float,
    max_pairs: int,
    max_cell_occ: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if broad_phase == "all_pairs":
        return rebuild_all_pairs_jax(
            centers=centers,
            rod_ids=rod_ids,
            axes=axes,
            lengths=lengths,
            radii=radii,
            max_pairs=max_pairs,
        )
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
        return _rebuild_broad_phase_pairs(
            broad_phase=op.metadata.broad_phase,
            centers=kinematics["centers"],
            rod_ids=op._rod_ids,
            axes=kinematics["axes"],
            lengths=kinematics["lengths"],
            radii=kinematics["radii"],
            cell_size=op.metadata.cell_size,
            max_pairs=op.metadata.max_pairs,
            max_cell_occ=op.metadata.max_neighbors_per_capsule,
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
    rod-rod contact registration style. Both paths run as pure JAX on the
    block device.

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
        assert _system is not None, (
            "CapsuleContactOp requires a finalized block system."
        )
        assert broad_phase in {"spatial_hash", "all_pairs"}, (
            f"Unsupported broad_phase {broad_phase!r}; "
            "expected 'spatial_hash' or 'all_pairs'."
        )
        if metadata is None:
            assert n_elements_per_rod is not None, (
                "Provide metadata or n_elements_per_rod for CapsuleContactOp."
            )
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

    def jax_block_operate_synchronize(self, state, time):
        return _apply_capsule_contact_unified(self, state, time)


def _apply_wall_contact_unified(
    op: WallContactOp,
    state: dict[str, Any],
    time: jax.Array,
) -> dict[str, Any]:
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
            assert n_elements_per_rod is not None, (
                "Provide metadata or n_elements_per_rod for WallContactOp."
            )
            metadata = build_block_capsule_metadata(
                _system,
                n_elements_per_rod=n_elements_per_rod,
                max_neighbors_per_capsule=max_neighbors_per_capsule,
            )
        self._block = _system
        self.metadata = metadata
        self.wall_origins = wall_origins
        self.wall_normals = wall_normals
        self.contact_stiffness = contact_stiffness
        self.contact_damping = contact_damping

    def jax_block_operate_synchronize(self, state, time):
        return _apply_wall_contact_unified(self, state, time)
