"""JAX block operators for the snake self-activation benchmark."""

from __future__ import annotations

import numpy as np

import elastica_jax as eaj
from elastica.utils import _bspline

import jax
import jax.numpy as jnp

from elastica_jax.memory_block.rod_local_map import RodLocalState
from elastica_jax._linalg import _jax_batch_cross, _jax_batch_matvec


class SnakeMuscleTorquesBlockJax(eaj.NoBlockOpJax):
    """Muscle torques on one rod; Block.map_rods batches across the Block."""

    def __init__(
        self,
        *,
        rest_lengths: np.ndarray,
        b_coeff: np.ndarray,
        period: float,
        **kwargs: object,
    ) -> None:
        s = np.cumsum(rest_lengths)
        s /= s[-1]
        spline_fn, _, _ = _bspline(b_coeff[:-1])
        self.spline = spline_fn(s)
        self.s = s
        self.direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = 2.0 * np.pi / float(b_coeff[-1])
        self.phase_shift = 0.0
        self.ramp_up_time = period

    def jax_per_rod_operate_synchronize(
        self,
        rod_view: RodLocalState,
        time: np.float64,
    ) -> RodLocalState:
        directors = rod_view.director_collection
        factor = jnp.minimum(1.0, time / self.ramp_up_time)
        torque_mag = (
            factor
            * self.spline
            * jnp.sin(
                self.angular_frequency * time
                - self.wave_number * self.s
                + self.phase_shift
            )
        )
        torque_local = self.direction[:, None] * torque_mag[::-1][None, :]
        torque_world = _jax_batch_matvec(directors, torque_local)
        external_torques = rod_view.external_torques
        external_torques = external_torques.at[:, 1:].add(torque_world[:, 1:])
        previous_world = _jax_batch_matvec(directors[:, :, :-1], torque_local[:, 1:])
        external_torques = external_torques.at[:, :-1].add(-previous_world)
        rod_view.external_torques = external_torques
        return rod_view


class GravityPlaneContactBlockJax(eaj.NoBlockOpJax):
    """Gravity and plane contact on one rod; Block.map_rods batches across rods."""

    def __init__(
        self,
        *,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
        slip_velocity_tol: float,
        k: float,
        nu: float,
        kinetic_mu_array: np.ndarray,
        **kwargs: object,
    ) -> None:
        self.plane_origin = plane_origin
        self.plane_normal = plane_normal
        self.gravity = np.array([0.0, -9.80665, 0.0], dtype=np.float64)
        self.surface_tol = 1.0e-4
        self.slip_velocity_tol = slip_velocity_tol
        self.k = k
        self.nu = nu
        self.kinetic_mu_forward = kinetic_mu_array[0]
        self.kinetic_mu_backward = kinetic_mu_array[1]
        self.kinetic_mu_sideways = kinetic_mu_array[2]

    def jax_per_rod_operate_synchronize(
        self,
        rod_view: RodLocalState,
        _time: np.float64,
    ) -> RodLocalState:
        position = rod_view.position_collection
        velocity = rod_view.velocity_collection
        mass = rod_view.mass
        radius = rod_view.radius
        tangents = rod_view.tangents
        directors = rod_view.director_collection
        omegas = rod_view.omega_collection
        internal_forces = rod_view.internal_forces
        external_forces = rod_view.external_forces
        external_torques = rod_view.external_torques

        external_forces = external_forces + self.gravity[:, None] * mass

        nodal_total_forces = internal_forces + external_forces
        element_total_forces = _node_to_element_mass_or_force(nodal_total_forces)
        plane_normal = self.plane_normal[:, None]
        force_component_along_normal_direction = jnp.sum(
            plane_normal * element_total_forces, axis=0
        )
        forces_along_normal_direction = (
            plane_normal * force_component_along_normal_direction
        )
        forces_along_normal_direction = jnp.where(
            force_component_along_normal_direction > 0.0,
            0.0,
            forces_along_normal_direction,
        )
        plane_response_force = -forces_along_normal_direction

        element_position = _node_to_element_position(position)
        distance_from_plane = jnp.sum(
            plane_normal * (element_position - self.plane_origin[:, None]),
            axis=0,
        )
        plane_penetration = jnp.minimum(distance_from_plane - radius, 0.0)
        elastic_force = -self.k * plane_normal * plane_penetration
        element_velocity = _node_to_element_velocity(mass, velocity)
        normal_component_of_element_velocity = jnp.sum(
            plane_normal * element_velocity, axis=0
        )
        damping_force = -self.nu * plane_normal * normal_component_of_element_velocity
        plane_response_force_total = (
            plane_response_force + elastic_force + damping_force
        )
        no_contact = (distance_from_plane - radius) > self.surface_tol
        plane_response_force = jnp.where(no_contact, 0.0, plane_response_force)
        plane_response_force_total = jnp.where(
            no_contact, 0.0, plane_response_force_total
        )

        plane_response_force_mag = jnp.linalg.norm(plane_response_force, axis=0)
        tangent_along_normal_direction = jnp.sum(plane_normal * tangents, axis=0)
        tangent_perpendicular_to_normal_direction = (
            tangents - plane_normal * tangent_along_normal_direction
        )
        tangent_perpendicular_mag = jnp.linalg.norm(
            tangent_perpendicular_to_normal_direction, axis=0
        )
        axial_direction = tangent_perpendicular_to_normal_direction / (
            tangent_perpendicular_mag + 1.0e-14
        )
        element_velocity = _node_to_element_velocity(mass, velocity)
        velocity_mag_along_axial_direction = jnp.sum(
            element_velocity * axial_direction, axis=0
        )
        velocity_along_axial_direction = (
            axial_direction * velocity_mag_along_axial_direction
        )
        velocity_sign_along_axial_direction = jnp.sign(
            velocity_mag_along_axial_direction
        )
        kinetic_mu = 0.5 * (
            self.kinetic_mu_forward * (1.0 + velocity_sign_along_axial_direction)
            + self.kinetic_mu_backward * (1.0 - velocity_sign_along_axial_direction)
        )
        rolling_direction = _jax_batch_cross(
            axial_direction,
            jnp.broadcast_to(plane_normal, axial_direction.shape),
        )
        torque_arm = -plane_normal * radius
        velocity_mag_along_rolling_direction = jnp.sum(
            element_velocity * rolling_direction, axis=0
        )
        directors_transpose = jnp.transpose(directors, (1, 0, 2))
        rotation_velocity = _jax_batch_matvec(
            directors_transpose,
            _jax_batch_cross(
                omegas,
                _jax_batch_matvec(directors, torque_arm),
            ),
        )
        rotation_velocity_along_rolling_direction = jnp.sum(
            rotation_velocity * rolling_direction, axis=0
        )
        slip_velocity_mag_along_rolling_direction = (
            velocity_mag_along_rolling_direction
            + rotation_velocity_along_rolling_direction
        )
        slip_velocity_along_rolling_direction = (
            rolling_direction * slip_velocity_mag_along_rolling_direction
        )
        slip_function_along_axial_direction = _find_slipping_elements(
            velocity_along_axial_direction,
            self.slip_velocity_tol,
        )
        slip_function_along_rolling_direction = _find_slipping_elements(
            slip_velocity_along_rolling_direction,
            self.slip_velocity_tol,
        )
        unitized_total_velocity = (
            slip_velocity_along_rolling_direction + velocity_along_axial_direction
        )
        unitized_total_velocity = unitized_total_velocity / (
            jnp.linalg.norm(unitized_total_velocity + 1.0e-14, axis=0)
        )
        kinetic_friction_force_along_axial_direction = (
            -(1.0 - slip_function_along_axial_direction)
            * kinetic_mu
            * plane_response_force_mag
            * jnp.sum(unitized_total_velocity * axial_direction, axis=0)
            * axial_direction
        )
        kinetic_friction_force_along_axial_direction = jnp.where(
            no_contact,
            0.0,
            kinetic_friction_force_along_axial_direction,
        )
        kinetic_friction_force_along_rolling_direction = (
            -(1.0 - slip_function_along_rolling_direction)
            * self.kinetic_mu_sideways
            * plane_response_force_mag
            * jnp.sum(unitized_total_velocity * rolling_direction, axis=0)
            * rolling_direction
        )
        kinetic_friction_force_along_rolling_direction = jnp.where(
            no_contact,
            0.0,
            kinetic_friction_force_along_rolling_direction,
        )
        external_forces = external_forces + _elements_to_nodes(
            plane_response_force_total
        )
        external_forces = external_forces + _elements_to_nodes(
            kinetic_friction_force_along_axial_direction
        )
        external_forces = external_forces + _elements_to_nodes(
            kinetic_friction_force_along_rolling_direction
        )
        external_torques = external_torques + _jax_batch_matvec(
            directors,
            _jax_batch_cross(
                torque_arm,
                kinetic_friction_force_along_rolling_direction,
            ),
        )

        rod_view.external_forces = external_forces
        rod_view.external_torques = external_torques
        return rod_view


def _node_to_element_position(position_collection: jax.Array) -> jax.Array:
    return 0.5 * (position_collection[:, 1:] + position_collection[:, :-1])


def _node_to_element_velocity(
    mass: jax.Array, velocity_collection: jax.Array
) -> jax.Array:
    numerator = (
        mass[None, 1:] * velocity_collection[:, 1:]
        + mass[None, :-1] * velocity_collection[:, :-1]
    )
    denominator = mass[None, 1:] + mass[None, :-1]
    return numerator / denominator


def _node_to_element_mass_or_force(nodal_collection: jax.Array) -> jax.Array:
    elemental_collection = 0.5 * (nodal_collection[:, :-1] + nodal_collection[:, 1:])
    elemental_collection = elemental_collection.at[:, 0].add(
        0.5 * nodal_collection[:, 0]
    )
    elemental_collection = elemental_collection.at[:, -1].add(
        0.5 * nodal_collection[:, -1]
    )
    return elemental_collection


def _elements_to_nodes(element_collection: jax.Array) -> jax.Array:
    node_collection = jnp.zeros(
        (element_collection.shape[0], element_collection.shape[1] + 1),
        dtype=element_collection.dtype,
    )
    node_collection = node_collection.at[:, :-1].add(0.5 * element_collection)
    node_collection = node_collection.at[:, 1:].add(0.5 * element_collection)
    return node_collection


def _find_slipping_elements(
    velocity_slip: jax.Array, velocity_threshold: jax.Array
) -> jax.Array:
    abs_velocity_slip = jnp.linalg.norm(velocity_slip, axis=0)
    normalized = abs_velocity_slip / velocity_threshold - 1.0
    slipped = jnp.minimum(1.0, normalized)
    slip_function = jnp.ones_like(abs_velocity_slip)
    slip_values = jnp.abs(1.0 - slipped)
    return jnp.where(abs_velocity_slip > velocity_threshold, slip_values, slip_function)
