"""Anisotropic-friction rod-plane contact for the muscular snake body."""

from __future__ import annotations

import numpy as np

import elastica_jax as eaj
from elastica_jax._linalg import (
    _jax_batch_cross as _batch_cross,
    _jax_batch_matvec as _batch_matvec,
)

import jax
import jax.numpy as jnp


class MuscularSnakePlaneContactJax(eaj.NoOpsJax):
    """Rod-plane contact with anisotropic friction on the snake body block."""

    def __init__(
        self,
        *,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
        slip_velocity_tol: float,
        contact_k: float,
        contact_nu: float,
        kinetic_mu_array: np.ndarray,
        static_mu_array: np.ndarray,
        _system=None,
    ) -> None:
        del _system, static_mu_array
        self.plane_origin = plane_origin
        self.plane_normal = plane_normal
        self.surface_tol = 1.0e-4
        self.slip_velocity_tol = slip_velocity_tol
        self.k = contact_k
        self.nu = contact_nu
        self.kinetic_mu_forward = kinetic_mu_array[0]
        self.kinetic_mu_backward = kinetic_mu_array[1]
        self.kinetic_mu_sideways = kinetic_mu_array[2]

    def jax_operate_synchronize(self, rod_view, time):
        del time
        plane_origin = self.plane_origin
        plane_normal = self.plane_normal
        surface_tol = self.surface_tol
        slip_velocity_tol = self.slip_velocity_tol
        k = self.k
        nu = self.nu
        kinetic_mu_sideways = self.kinetic_mu_sideways

        nodal_total_forces = rod_view.internal_forces + rod_view.external_forces
        element_total_forces = _node_to_element_mass_or_force_jax(nodal_total_forces)
        force_component_along_normal_direction = jnp.sum(
            plane_normal[:, None] * element_total_forces, axis=0
        )
        forces_along_normal_direction = (
            plane_normal[:, None] * force_component_along_normal_direction[None, :]
        )
        forces_along_normal_direction = jnp.where(
            force_component_along_normal_direction[None, :] > 0.0,
            0.0,
            forces_along_normal_direction,
        )
        plane_response_force = -forces_along_normal_direction

        element_position = _node_to_element_position_jax(rod_view.position_collection)
        distance_from_plane = jnp.sum(
            plane_normal[:, None] * (element_position - plane_origin[:, None]), axis=0
        )
        plane_penetration = jnp.minimum(distance_from_plane - rod_view.radius, 0.0)
        elastic_force = -k * plane_normal[:, None] * plane_penetration[None, :]

        element_velocity = _node_to_element_velocity_jax(
            rod_view.mass, rod_view.velocity_collection
        )
        normal_component_of_element_velocity = jnp.sum(
            plane_normal[:, None] * element_velocity, axis=0
        )
        damping_force = (
            -nu * plane_normal[:, None] * normal_component_of_element_velocity[None, :]
        )

        plane_response_force_total = (
            plane_response_force + elastic_force + damping_force
        )
        no_contact = (distance_from_plane - rod_view.radius) > surface_tol
        plane_response_force = jnp.where(no_contact[None, :], 0.0, plane_response_force)
        plane_response_force_total = jnp.where(
            no_contact[None, :], 0.0, plane_response_force_total
        )

        external_forces = rod_view.external_forces + _elements_to_nodes_jax(
            plane_response_force_total
        )
        plane_response_force_mag = jnp.linalg.norm(plane_response_force, axis=0)

        tangent_along_normal_direction = jnp.sum(
            plane_normal[:, None] * rod_view.tangents, axis=0
        )
        tangent_perpendicular_to_normal_direction = (
            rod_view.tangents
            - plane_normal[:, None] * tangent_along_normal_direction[None, :]
        )
        tangent_perpendicular_mag = jnp.linalg.norm(
            tangent_perpendicular_to_normal_direction, axis=0
        )
        axial_direction = tangent_perpendicular_to_normal_direction / (
            tangent_perpendicular_mag[None, :] + 1.0e-14
        )

        velocity_mag_along_axial_direction = jnp.sum(
            element_velocity * axial_direction, axis=0
        )
        velocity_along_axial_direction = (
            axial_direction * velocity_mag_along_axial_direction[None, :]
        )
        velocity_sign_along_axial_direction = jnp.sign(
            velocity_mag_along_axial_direction
        )
        kinetic_mu = 0.5 * (
            self.kinetic_mu_forward * (1.0 + velocity_sign_along_axial_direction)
            + self.kinetic_mu_backward * (1.0 - velocity_sign_along_axial_direction)
        )
        slip_function_along_axial_direction = _find_slipping_elements_jax(
            velocity_along_axial_direction, slip_velocity_tol
        )

        rolling_direction = _batch_cross(
            axial_direction,
            jnp.repeat(plane_normal[:, None], axial_direction.shape[1], axis=1),
        )
        torque_arm = -plane_normal[:, None] * rod_view.radius[None, :]
        velocity_along_rolling_direction = jnp.sum(
            element_velocity * rolling_direction, axis=0
        )
        directors_transpose = jnp.transpose(rod_view.director_collection, (1, 0, 2))
        rotation_velocity = _batch_matvec(
            directors_transpose,
            _batch_cross(
                rod_view.omega_collection,
                _batch_matvec(rod_view.director_collection, torque_arm),
            ),
        )
        rotation_velocity_along_rolling_direction = jnp.sum(
            rotation_velocity * rolling_direction, axis=0
        )
        slip_velocity_mag_along_rolling_direction = (
            velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
        )
        slip_velocity_along_rolling_direction = (
            rolling_direction * slip_velocity_mag_along_rolling_direction[None, :]
        )
        slip_function_along_rolling_direction = _find_slipping_elements_jax(
            slip_velocity_along_rolling_direction, slip_velocity_tol
        )

        unitized_total_velocity = (
            slip_velocity_along_rolling_direction + velocity_along_axial_direction
        )
        unitized_total_velocity = (
            unitized_total_velocity
            / (jnp.linalg.norm(unitized_total_velocity + 1.0e-14, axis=0)[None, :])
        )
        kinetic_friction_force_along_axial_direction = (
            -(
                (1.0 - slip_function_along_axial_direction)
                * kinetic_mu
                * plane_response_force_mag
                * jnp.sum(unitized_total_velocity * axial_direction, axis=0)
            )[None, :]
            * axial_direction
        )
        kinetic_friction_force_along_axial_direction = jnp.where(
            no_contact[None, :], 0.0, kinetic_friction_force_along_axial_direction
        )
        external_forces = external_forces + _elements_to_nodes_jax(
            kinetic_friction_force_along_axial_direction
        )

        kinetic_friction_force_along_rolling_direction = (
            -(
                (1.0 - slip_function_along_rolling_direction)
                * kinetic_mu_sideways
                * plane_response_force_mag
                * jnp.sum(unitized_total_velocity * rolling_direction, axis=0)
            )[None, :]
            * rolling_direction
        )
        kinetic_friction_force_along_rolling_direction = jnp.where(
            no_contact[None, :], 0.0, kinetic_friction_force_along_rolling_direction
        )
        external_forces = external_forces + _elements_to_nodes_jax(
            kinetic_friction_force_along_rolling_direction
        )

        external_torques = rod_view.external_torques + _batch_matvec(
            rod_view.director_collection,
            _batch_cross(torque_arm, kinetic_friction_force_along_rolling_direction),
        )

        rod_view.external_forces = external_forces
        rod_view.external_torques = external_torques
        return rod_view


def _node_to_element_position_jax(position_collection: jax.Array) -> jax.Array:
    return 0.5 * (position_collection[:, 1:] + position_collection[:, :-1])


def _node_to_element_velocity_jax(
    mass: jax.Array,
    velocity_collection: jax.Array,
) -> jax.Array:
    numerator = (
        mass[jnp.newaxis, 1:] * velocity_collection[:, 1:]
        + mass[jnp.newaxis, :-1] * velocity_collection[:, :-1]
    )
    denominator = mass[jnp.newaxis, 1:] + mass[jnp.newaxis, :-1]
    return numerator / denominator


def _node_to_element_mass_or_force_jax(nodal_collection: jax.Array) -> jax.Array:
    elemental_collection = 0.5 * (nodal_collection[:, :-1] + nodal_collection[:, 1:])
    elemental_collection = elemental_collection.at[:, 0].add(
        0.5 * nodal_collection[:, 0]
    )
    elemental_collection = elemental_collection.at[:, -1].add(
        0.5 * nodal_collection[:, -1]
    )
    return elemental_collection


def _elements_to_nodes_jax(element_collection: jax.Array) -> jax.Array:
    node_collection = jnp.zeros(
        (element_collection.shape[0], element_collection.shape[1] + 1),
        dtype=element_collection.dtype,
    )
    node_collection = node_collection.at[:, :-1].add(0.5 * element_collection)
    node_collection = node_collection.at[:, 1:].add(0.5 * element_collection)
    return node_collection


def _find_slipping_elements_jax(
    velocity_slip: jax.Array, velocity_threshold: jax.Array
) -> jax.Array:
    abs_velocity_slip = jnp.linalg.norm(velocity_slip, axis=0)
    normalized = abs_velocity_slip / velocity_threshold - 1.0
    slipped = jnp.minimum(1.0, normalized)
    slip_function = jnp.ones_like(abs_velocity_slip)
    slip_values = jnp.abs(1.0 - slipped)
    return jnp.where(abs_velocity_slip > velocity_threshold, slip_values, slip_function)
