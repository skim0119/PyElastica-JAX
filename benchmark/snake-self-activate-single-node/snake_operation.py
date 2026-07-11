"""JAX block operators for the snake self-activation benchmark."""

from __future__ import annotations

import numpy as np

import elastica as ea
import elastica_jax as eaj

import jax
import jax.numpy as jnp

from elastica_jax._linalg import _jax_batch_cross, _jax_batch_matvec


class SnakeMuscleTorquesBlockJax(eaj.NoBlockOpJax):
    """Muscle torques on one rod-shaped block state.

    Author against a single rod's arrays (``(3, N)``, etc.). The simulator
    backend batches across rods with ``vmap`` (and gather/scatter when the
    block is packed horizontally).
    """

    def __init__(
        self,
        *,
        b_coeff: np.ndarray,
        period: float,
        base_length: float,
        _system,
    ) -> None:
        template_rod = _system._systems[0]
        template = ea.MuscleTorques(
            base_length=base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / float(b_coeff[-1]),
            phase_shift=0.0,
            direction=np.array([0.0, 1.0, 0.0]),
            rest_lengths=np.asarray(template_rod.rest_lengths),
            ramp_up_time=period,
            with_spline=True,
        )
        self.direction = jnp.asarray(np.array([0.0, 1.0, 0.0], dtype=np.float64))
        self.s = jnp.asarray(np.asarray(template.s, dtype=np.float64))
        self.spline = jnp.asarray(np.asarray(template.my_spline, dtype=np.float64))
        self.angular_frequency = np.float64(2.0 * np.pi / period)
        self.wave_number = np.float64(2.0 * np.pi / float(b_coeff[-1]))
        self.phase_shift = np.float64(0.0)
        self.ramp_up_time = np.float64(period)

    def jax_block_operate_synchronize(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        dtype = state["director_collection"].dtype
        directors = state["director_collection"]
        factor = jnp.minimum(
            jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(time, dtype=dtype)
            / jnp.asarray(self.ramp_up_time, dtype=dtype),
        )
        torque_mag = (
            factor
            * jnp.asarray(self.spline, dtype=dtype)
            * jnp.sin(
                jnp.asarray(self.angular_frequency, dtype=dtype)
                * jnp.asarray(time, dtype=dtype)
                - jnp.asarray(self.wave_number, dtype=dtype)
                * jnp.asarray(self.s, dtype=dtype)
                + jnp.asarray(self.phase_shift, dtype=dtype)
            )
        )
        torque_local = (
            jnp.asarray(self.direction, dtype=dtype)[:, None]
            * torque_mag[::-1][None, :]
        )
        torque_world = _jax_batch_matvec(directors, torque_local)
        external_torques = state["external_torques"]
        external_torques = external_torques.at[:, 1:].add(torque_world[:, 1:])
        previous_world = _jax_batch_matvec(directors[:, :, :-1], torque_local[:, 1:])
        external_torques = external_torques.at[:, :-1].add(-previous_world)
        updated = dict(state)
        updated["external_torques"] = external_torques
        return updated


class GravityPlaneContactBlockJax(eaj.NoBlockOpJax):
    """Gravity and plane contact on one rod-shaped block state.

    Batched across rods by the simulator backend (see
    :class:`SnakeMuscleTorquesBlockJax`).
    """

    def __init__(
        self,
        *,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
        slip_velocity_tol: float,
        k: float,
        nu: float,
        kinetic_mu_array: np.ndarray,
        static_mu_array: np.ndarray,
        _system,
    ) -> None:
        del static_mu_array, _system
        self.plane_origin = jnp.asarray(np.asarray(plane_origin, dtype=np.float64))
        self.plane_normal = jnp.asarray(np.asarray(plane_normal, dtype=np.float64))
        self.gravity = jnp.asarray(np.array([0.0, -9.80665, 0.0], dtype=np.float64))
        self.surface_tol = np.float64(1.0e-4)
        self.slip_velocity_tol = np.float64(slip_velocity_tol)
        self.k = np.float64(k)
        self.nu = np.float64(nu)
        self.kinetic_mu_forward = np.float64(kinetic_mu_array[0])
        self.kinetic_mu_backward = np.float64(kinetic_mu_array[1])
        self.kinetic_mu_sideways = np.float64(kinetic_mu_array[2])

    def jax_block_operate_synchronize(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        del time
        dtype = state["position_collection"].dtype
        position = state["position_collection"]
        velocity = state["velocity_collection"]
        mass = state["mass"]
        radius = state["radius"]
        tangents = state["tangents"]
        directors = state["director_collection"]
        omegas = state["omega_collection"]
        internal_forces = state["internal_forces"]
        external_forces = state["external_forces"]
        external_torques = state["external_torques"]

        external_forces = (
            external_forces + jnp.asarray(self.gravity, dtype=dtype)[:, None] * mass
        )

        nodal_total_forces = internal_forces + external_forces
        element_total_forces = _node_to_element_mass_or_force(nodal_total_forces)
        plane_normal = jnp.asarray(self.plane_normal, dtype=dtype)[:, None]
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
            plane_normal
            * (element_position - jnp.asarray(self.plane_origin, dtype=dtype)[:, None]),
            axis=0,
        )
        plane_penetration = jnp.minimum(distance_from_plane - radius, 0.0)
        elastic_force = (
            -jnp.asarray(self.k, dtype=dtype) * plane_normal * plane_penetration
        )
        element_velocity = _node_to_element_velocity(mass, velocity)
        normal_component_of_element_velocity = jnp.sum(
            plane_normal * element_velocity, axis=0
        )
        damping_force = (
            -jnp.asarray(self.nu, dtype=dtype)
            * plane_normal
            * normal_component_of_element_velocity
        )
        plane_response_force_total = (
            plane_response_force + elastic_force + damping_force
        )
        no_contact = (distance_from_plane - radius) > jnp.asarray(
            self.surface_tol, dtype=dtype
        )
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
            tangent_perpendicular_mag + jnp.asarray(1.0e-14, dtype=dtype)
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
            jnp.asarray(self.kinetic_mu_forward, dtype=dtype)
            * (1.0 + velocity_sign_along_axial_direction)
            + jnp.asarray(self.kinetic_mu_backward, dtype=dtype)
            * (1.0 - velocity_sign_along_axial_direction)
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
            jnp.asarray(self.slip_velocity_tol, dtype=dtype),
        )
        slip_function_along_rolling_direction = _find_slipping_elements(
            slip_velocity_along_rolling_direction,
            jnp.asarray(self.slip_velocity_tol, dtype=dtype),
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
            * jnp.asarray(self.kinetic_mu_sideways, dtype=dtype)
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

        updated = dict(state)
        updated["external_forces"] = external_forces
        updated["external_torques"] = external_torques
        return updated


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
