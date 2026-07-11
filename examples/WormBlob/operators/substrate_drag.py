"""Floor-parallel substrate drag with thermal-role modulation."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

import elastica_jax as eaj

from block_utils import gather_vector_batch, uniform_index_matrix
from role_map import role_weights_jax


class SubstrateDragJax(eaj.NoBlockOpJax):
    r"""Linear floor-parallel drag on near-floor nodes, reduced for wigglers.

    Only nodes within ``contact_tolerance`` of the floor experience drag. The
    drag opposes the floor-parallel (``x``-``y``) velocity component with
    coefficient ``drag_base * (1 - wiggle_drag_reduction * wiggle_weight)``,
    emulating the rear-lifting / friction-reduction stroke of hot-side wigglers.
    The per-rod wiggle weight comes from the centroid thermal fraction (see
    :func:`role_map.role_weights_jax`).

    Parameters
    ----------
    cold_x : float
        World ``x`` of the coldest edge (puller side).
    hot_x : float
        World ``x`` of the hottest edge (wiggler side).
    drag_base : float
        Baseline linear drag coefficient at zero wiggler weight.
    wiggle_drag_reduction : float
        Fractional drag reduction at full wiggler weight, in ``[0, 1]``.
    floor_height : float
        World ``z`` of the floor plane.
    contact_tolerance : float
        Vertical band above the floor within which drag is applied.
    """

    def __init__(
        self,
        *,
        cold_x: float,
        hot_x: float,
        drag_base: float,
        wiggle_drag_reduction: float,
        floor_height: float,
        contact_tolerance: float,
        _system,
    ) -> None:
        self.node_indices = jnp.asarray(
            uniform_index_matrix(
                _system.start_idx_in_rod_nodes,
                _system.end_idx_in_rod_nodes,
            )
        )
        self.cold_x = cold_x
        self.hot_x = hot_x
        self.drag_base = drag_base
        self.wiggle_drag_reduction = wiggle_drag_reduction
        self.floor_height = floor_height
        self.contact_tolerance = contact_tolerance

    def jax_block_operate_synchronize(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        del time
        positions = gather_vector_batch(state["position_collection"], self.node_indices)
        velocities = gather_vector_batch(
            state["velocity_collection"], self.node_indices
        )
        rod_center_x = jnp.mean(positions, axis=2)[:, 0]
        _pull_weight, wiggle_weight, _binder_weight = role_weights_jax(
            rod_center_x,
            cold_x=float(self.cold_x),
            hot_x=float(self.hot_x),
        )
        drag_scale = jnp.clip(
            1.0 - self.wiggle_drag_reduction * wiggle_weight, 0.0, 1.0
        )
        drag_coeff = self.drag_base * drag_scale[:, None]
        near_floor = positions[:, 2, :] <= (self.floor_height + self.contact_tolerance)
        tangential_velocity = velocities.at[:, 2, :].set(0.0)
        drag_force = -drag_coeff[:, None, :] * tangential_velocity
        drag_force = jnp.where(near_floor[:, None, :], drag_force, 0.0)
        external_forces = (
            state["external_forces"]
            .at[:, self.node_indices]
            .add(jnp.moveaxis(drag_force, 0, 1))
        )
        return {**state, "external_forces": external_forces}
