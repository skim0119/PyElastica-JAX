"""Side-by-side surface joint connecting muscle rods to the snake body."""

from __future__ import annotations

import numpy as np
from elastica.experimental.connection_contact_joint.parallel_connection import (
    get_connection_vector_straight_straight_rod,
)

import elastica_jax as eaj
from elastica_jax._linalg import (
    _jax_batch_cross as _batch_cross,
    _jax_batch_matvec as _batch_matvec,
)

import jax.numpy as jnp

__all__ = [
    "SurfaceJointSideBySideJax",
    "get_connection_vector_straight_straight_rod",
]


class SurfaceJointSideBySideJax(eaj.NoRodRodBlockOpJax):
    """Surface joint binding two parallel rods along their contacting faces.

    JAX port of the experimental elastica side-by-side surface joint. Each paired
    element connection applies an elastic restoring force ``k``, viscous damping
    ``nu``, and a one-sided repulsive contact force ``k_repulsive`` between the
    paired surface points, keeping each muscle rod glued to and offset from the
    snake body.

    Parameters
    ----------
    k : numpy.ndarray
        Per-connection elastic stiffness [N/m].
    nu : float
        Viscous damping coefficient [N s/m].
    k_repulsive : numpy.ndarray
        Per-connection repulsive contact stiffness [N/m].
    rod_one_direction_vec_in_material_frame : numpy.ndarray
        Body-frame connection direction on the body rod, shape ``(3, n_conn)``.
    rod_two_direction_vec_in_material_frame : numpy.ndarray
        Body-frame connection direction on the muscle rod, shape ``(3, n_conn)``.
    offset_btw_rods : numpy.ndarray
        Rest surface separation between paired connection points [m].
    body_element_index : numpy.ndarray
        Body element indices for each connection.
    muscle_element_index : numpy.ndarray
        Muscle element indices for each connection.
    """

    def __init__(
        self,
        *,
        k: np.ndarray,
        nu: float,
        k_repulsive: np.ndarray,
        rod_one_direction_vec_in_material_frame: np.ndarray,
        rod_two_direction_vec_in_material_frame: np.ndarray,
        offset_btw_rods: np.ndarray,
        body_element_index: np.ndarray,
        muscle_element_index: np.ndarray,
        _first_system=None,
        _second_system=None,
    ) -> None:
        self.k = k
        self.nu = nu
        self.k_repulsive = k_repulsive
        self.offset_btw_rods = offset_btw_rods
        self.rod_one_direction_vec_in_material_frame = (
            rod_one_direction_vec_in_material_frame
        )
        self.rod_two_direction_vec_in_material_frame = (
            rod_two_direction_vec_in_material_frame
        )
        self.body_element_index = body_element_index
        self.muscle_element_index = muscle_element_index

    def jax_operation(self, rod_one_view, rod_two_view, time):
        del time
        return _apply_surface_joint(
            rod_one_view,
            rod_two_view,
            k=self.k,
            nu=self.nu,
            k_repulsive=self.k_repulsive,
            rod_one_direction_vec_in_material_frame=(
                self.rod_one_direction_vec_in_material_frame
            ),
            rod_two_direction_vec_in_material_frame=(
                self.rod_two_direction_vec_in_material_frame
            ),
            offset_btw_rods=self.offset_btw_rods,
            body_element_index=self.body_element_index,
            muscle_element_index=self.muscle_element_index,
        )


def _apply_surface_joint(
    rod_one_view,
    rod_two_view,
    *,
    k: np.ndarray,
    nu: np.float64,
    k_repulsive: np.ndarray,
    rod_one_direction_vec_in_material_frame: np.ndarray,
    rod_two_direction_vec_in_material_frame: np.ndarray,
    offset_btw_rods: np.ndarray,
    body_element_index: np.ndarray,
    muscle_element_index: np.ndarray,
):
    body_idx = jnp.asarray(body_element_index)
    muscle_idx = jnp.asarray(muscle_element_index)

    rod_one_to_rod_two_connection_vec = _batch_matvec(
        jnp.transpose(rod_one_view.director_collection[:, :, body_idx], (1, 0, 2)),
        rod_one_direction_vec_in_material_frame,
    )
    rod_two_to_rod_one_connection_vec = _batch_matvec(
        jnp.transpose(rod_two_view.director_collection[:, :, muscle_idx], (1, 0, 2)),
        rod_two_direction_vec_in_material_frame,
    )

    rod_one_element_position = 0.5 * (
        rod_one_view.position_collection[:, body_idx]
        + rod_one_view.position_collection[:, body_idx + 1]
    )
    rod_two_element_position = 0.5 * (
        rod_two_view.position_collection[:, muscle_idx]
        + rod_two_view.position_collection[:, muscle_idx + 1]
    )

    offset_rod_one = 0.5 * offset_btw_rods / jnp.sqrt(rod_one_view.dilatation[body_idx])
    offset_rod_two = (
        0.5 * offset_btw_rods / jnp.sqrt(rod_two_view.dilatation[muscle_idx])
    )

    rod_one_rd2 = (
        rod_one_to_rod_two_connection_vec
        * (rod_one_view.radius[body_idx] + offset_rod_one)[None, :]
    )
    rod_two_rd2 = (
        rod_two_to_rod_one_connection_vec
        * (rod_two_view.radius[muscle_idx] + offset_rod_two)[None, :]
    )

    surface_position_rod_one = rod_one_element_position + rod_one_rd2
    surface_position_rod_two = rod_two_element_position + rod_two_rd2

    distance_vector = surface_position_rod_two - surface_position_rod_one
    spring_force = k[None, :] * distance_vector

    rod_one_element_velocity = 0.5 * (
        rod_one_view.velocity_collection[:, body_idx]
        + rod_one_view.velocity_collection[:, body_idx + 1]
    )
    rod_two_element_velocity = 0.5 * (
        rod_two_view.velocity_collection[:, muscle_idx]
        + rod_two_view.velocity_collection[:, muscle_idx + 1]
    )
    relative_velocity = rod_two_element_velocity - rod_one_element_velocity
    damping_force = nu * relative_velocity
    total_force = spring_force + damping_force

    center_distance = rod_two_element_position - rod_one_element_position
    center_distance_unit_vec = center_distance / jnp.linalg.norm(
        center_distance, axis=0, keepdims=True
    )
    penetration_strain = jnp.linalg.norm(center_distance, axis=0) - (
        rod_one_view.radius[body_idx]
        + offset_rod_one
        + rod_two_view.radius[muscle_idx]
        + offset_rod_two
    )
    k_contact = jnp.where(
        penetration_strain < 0.0,
        -k_repulsive * jnp.abs(penetration_strain) ** 1.5,
        0.0,
    )
    contact_force = k_contact[None, :] * center_distance_unit_vec
    total_force = total_force + contact_force

    half_force = 0.5 * total_force
    rod_one_view.external_forces = rod_one_view.external_forces.at[:, body_idx].add(
        half_force
    )
    rod_one_view.external_forces = rod_one_view.external_forces.at[:, body_idx + 1].add(
        half_force
    )
    rod_two_view.external_forces = rod_two_view.external_forces.at[:, muscle_idx].add(
        -half_force
    )
    rod_two_view.external_forces = rod_two_view.external_forces.at[
        :, muscle_idx + 1
    ].add(-half_force)

    torque_on_rod_one = _batch_cross(rod_one_rd2, spring_force)
    torque_on_rod_two = _batch_cross(rod_two_rd2, -spring_force)
    torque_on_rod_one_material_frame = _batch_matvec(
        rod_one_view.director_collection[:, :, body_idx], torque_on_rod_one
    )
    torque_on_rod_two_material_frame = _batch_matvec(
        rod_two_view.director_collection[:, :, muscle_idx], torque_on_rod_two
    )
    rod_one_view.external_torques = rod_one_view.external_torques.at[:, body_idx].add(
        torque_on_rod_one_material_frame
    )
    rod_two_view.external_torques = rod_two_view.external_torques.at[:, muscle_idx].add(
        torque_on_rod_two_material_frame
    )

    return rod_one_view, rod_two_view
