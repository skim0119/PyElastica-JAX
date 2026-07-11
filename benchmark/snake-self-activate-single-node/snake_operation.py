"""JAX block operators for the snake self-activation benchmark."""

from __future__ import annotations

import numpy as np

import elastica as ea
import elastica_jax as eaj

import jax
import jax.numpy as jnp

from elastica_jax._linalg import _jax_batch_cross, _jax_batch_matvec


def _is_stacked_block(system: object) -> bool:
    return isinstance(system, eaj._CosseratRodVerticalMemoryBlock)


class SnakeMuscleTorquesBlockJax(eaj.NoBlockOpJax):
    """Muscle torques for batched snake rods on a JAX memory block."""

    def __init__(
        self,
        *,
        b_coeff: np.ndarray,
        period: float,
        base_length: float,
        _system,
    ) -> None:
        widths = _system.end_idx_in_rod_elems - _system.start_idx_in_rod_elems
        assert np.all(widths == widths[0]), (
            "SnakeMuscleTorquesBlockJax requires uniform element counts across rods."
        )
        if _is_stacked_block(_system):
            rest_lengths = np.asarray(_system.rest_lengths[0])
        else:
            rest_lengths = np.asarray(
                _system.rest_lengths[
                    _system.start_idx_in_rod_elems[0] : _system.end_idx_in_rod_elems[0]
                ]
            )
        template = ea.MuscleTorques(
            base_length=base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / float(b_coeff[-1]),
            phase_shift=0.0,
            direction=np.array([0.0, 1.0, 0.0]),
            rest_lengths=rest_lengths,
            ramp_up_time=period,
            with_spline=True,
        )
        self.stacked = _is_stacked_block(_system)
        self.elem_indices = jnp.asarray(
            _uniform_index_matrix(
                _system.start_idx_in_rod_elems,
                _system.end_idx_in_rod_elems,
            )
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
        directors = _gather_tensor_batch(
            state["director_collection"], self.elem_indices
        )
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
            jnp.asarray(self.direction, dtype=dtype)[None, :, None]
            * torque_mag[::-1][None, None, :]
        )
        torque_local = jnp.broadcast_to(
            torque_local,
            (directors.shape[0], torque_local.shape[1], torque_local.shape[2]),
        )
        torque_world = _batch_matvec_over_rods(directors, torque_local)
        external_torques = state["external_torques"]
        if self.stacked:
            external_torques = external_torques.at[:, :, 1:].add(torque_world[:, :, 1:])
            previous_directors = directors[:, :, :, :-1]
            next_local = torque_local[:, :, 1:]
            previous_world = _batch_matvec_over_rods(previous_directors, next_local)
            external_torques = external_torques.at[:, :, :-1].add(-previous_world)
        else:
            external_torques = external_torques.at[:, self.elem_indices[:, 1:]].add(
                jnp.moveaxis(torque_world[:, :, 1:], 0, 1)
            )
            previous_directors = directors[:, :, :, :-1]
            next_local = torque_local[:, :, 1:]
            previous_world = _batch_matvec_over_rods(previous_directors, next_local)
            external_torques = external_torques.at[:, self.elem_indices[:, :-1]].add(
                -jnp.moveaxis(previous_world, 0, 1)
            )
        updated = dict(state)
        updated["external_torques"] = external_torques
        return updated


class GravityPlaneContactBlockJax(eaj.NoBlockOpJax):
    """Gravity and plane contact for batched snake rods on a JAX memory block."""

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
        del static_mu_array
        self.stacked = _is_stacked_block(_system)
        self.node_indices = jnp.asarray(
            _uniform_index_matrix(
                _system.start_idx_in_rod_nodes,
                _system.end_idx_in_rod_nodes,
            )
        )
        self.elem_indices = jnp.asarray(
            _uniform_index_matrix(
                _system.start_idx_in_rod_elems,
                _system.end_idx_in_rod_elems,
            )
        )
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
        position = _gather_vector_batch(state["position_collection"], self.node_indices)
        velocity = _gather_vector_batch(state["velocity_collection"], self.node_indices)
        mass = _gather_scalar_batch(state["mass"], self.node_indices)
        radius = _gather_scalar_batch(state["radius"], self.elem_indices)
        tangents = _gather_vector_batch(state["tangents"], self.elem_indices)
        directors = _gather_tensor_batch(
            state["director_collection"], self.elem_indices
        )
        omegas = _gather_vector_batch(state["omega_collection"], self.elem_indices)
        internal_forces = _gather_vector_batch(
            state["internal_forces"], self.node_indices
        )
        external_forces = _gather_vector_batch(
            state["external_forces"], self.node_indices
        )
        external_torques = _gather_vector_batch(
            state["external_torques"], self.elem_indices
        )

        external_forces = (
            external_forces
            + jnp.asarray(self.gravity, dtype=dtype)[None, :, None] * mass[:, None, :]
        )

        nodal_total_forces = internal_forces + external_forces
        element_total_forces = _node_to_element_mass_or_force_batch(nodal_total_forces)
        plane_normal = jnp.asarray(self.plane_normal, dtype=dtype)[None, :, None]
        force_component_along_normal_direction = jnp.sum(
            plane_normal * element_total_forces, axis=1
        )
        forces_along_normal_direction = (
            plane_normal * force_component_along_normal_direction[:, None, :]
        )
        forces_along_normal_direction = jnp.where(
            force_component_along_normal_direction[:, None, :] > 0.0,
            0.0,
            forces_along_normal_direction,
        )
        plane_response_force = -forces_along_normal_direction

        element_position = _node_to_element_position_batch(position)
        distance_from_plane = jnp.sum(
            plane_normal
            * (
                element_position
                - jnp.asarray(self.plane_origin, dtype=dtype)[None, :, None]
            ),
            axis=1,
        )
        plane_penetration = jnp.minimum(distance_from_plane - radius, 0.0)
        elastic_force = (
            -jnp.asarray(self.k, dtype=dtype)
            * plane_normal
            * plane_penetration[:, None, :]
        )
        element_velocity = _node_to_element_velocity_batch(mass, velocity)
        normal_component_of_element_velocity = jnp.sum(
            plane_normal * element_velocity, axis=1
        )
        damping_force = (
            -jnp.asarray(self.nu, dtype=dtype)
            * plane_normal
            * normal_component_of_element_velocity[:, None, :]
        )
        plane_response_force_total = (
            plane_response_force + elastic_force + damping_force
        )
        no_contact = (distance_from_plane - radius) > jnp.asarray(
            self.surface_tol, dtype=dtype
        )
        plane_response_force = jnp.where(
            no_contact[:, None, :], 0.0, plane_response_force
        )
        plane_response_force_total = jnp.where(
            no_contact[:, None, :], 0.0, plane_response_force_total
        )

        plane_response_force_mag = jnp.linalg.norm(plane_response_force, axis=1)
        tangent_along_normal_direction = jnp.sum(plane_normal * tangents, axis=1)
        tangent_perpendicular_to_normal_direction = (
            tangents - plane_normal * tangent_along_normal_direction[:, None, :]
        )
        tangent_perpendicular_mag = jnp.linalg.norm(
            tangent_perpendicular_to_normal_direction, axis=1
        )
        axial_direction = tangent_perpendicular_to_normal_direction / (
            tangent_perpendicular_mag[:, None, :] + jnp.asarray(1.0e-14, dtype=dtype)
        )
        element_velocity = _node_to_element_velocity_batch(mass, velocity)
        velocity_mag_along_axial_direction = jnp.sum(
            element_velocity * axial_direction, axis=1
        )
        velocity_along_axial_direction = (
            axial_direction * velocity_mag_along_axial_direction[:, None, :]
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
        rolling_direction = _batch_cross_over_rods(
            axial_direction,
            jnp.broadcast_to(plane_normal, axial_direction.shape),
        )
        torque_arm = -plane_normal * radius[:, None, :]
        velocity_mag_along_rolling_direction = jnp.sum(
            element_velocity * rolling_direction, axis=1
        )
        directors_transpose = jnp.transpose(directors, (0, 2, 1, 3))
        rotation_velocity = _batch_matvec_over_rods(
            directors_transpose,
            _batch_cross_over_rods(
                omegas,
                _batch_matvec_over_rods(directors, torque_arm),
            ),
        )
        rotation_velocity_along_rolling_direction = jnp.sum(
            rotation_velocity * rolling_direction, axis=1
        )
        slip_velocity_mag_along_rolling_direction = (
            velocity_mag_along_rolling_direction
            + rotation_velocity_along_rolling_direction
        )
        slip_velocity_along_rolling_direction = (
            rolling_direction * slip_velocity_mag_along_rolling_direction[:, None, :]
        )
        slip_function_along_axial_direction = _find_slipping_elements_batch(
            velocity_along_axial_direction,
            jnp.asarray(self.slip_velocity_tol, dtype=dtype),
        )
        slip_function_along_rolling_direction = _find_slipping_elements_batch(
            slip_velocity_along_rolling_direction,
            jnp.asarray(self.slip_velocity_tol, dtype=dtype),
        )
        unitized_total_velocity = (
            slip_velocity_along_rolling_direction + velocity_along_axial_direction
        )
        unitized_total_velocity = unitized_total_velocity / (
            jnp.linalg.norm(unitized_total_velocity + 1.0e-14, axis=1)[:, None, :]
        )
        kinetic_friction_force_along_axial_direction = (
            -(1.0 - slip_function_along_axial_direction)[:, None, :]
            * kinetic_mu[:, None, :]
            * plane_response_force_mag[:, None, :]
            * jnp.sum(unitized_total_velocity * axial_direction, axis=1)[:, None, :]
            * axial_direction
        )
        kinetic_friction_force_along_axial_direction = jnp.where(
            no_contact[:, None, :],
            0.0,
            kinetic_friction_force_along_axial_direction,
        )
        kinetic_friction_force_along_rolling_direction = (
            -(1.0 - slip_function_along_rolling_direction)[:, None, :]
            * jnp.asarray(self.kinetic_mu_sideways, dtype=dtype)
            * plane_response_force_mag[:, None, :]
            * jnp.sum(unitized_total_velocity * rolling_direction, axis=1)[:, None, :]
            * rolling_direction
        )
        kinetic_friction_force_along_rolling_direction = jnp.where(
            no_contact[:, None, :],
            0.0,
            kinetic_friction_force_along_rolling_direction,
        )
        external_forces = external_forces + _elements_to_nodes_batch(
            plane_response_force_total
        )
        external_forces = external_forces + _elements_to_nodes_batch(
            kinetic_friction_force_along_axial_direction
        )
        external_forces = external_forces + _elements_to_nodes_batch(
            kinetic_friction_force_along_rolling_direction
        )
        external_torques = external_torques + _batch_matvec_over_rods(
            directors,
            _batch_cross_over_rods(
                torque_arm,
                kinetic_friction_force_along_rolling_direction,
            ),
        )

        updated = dict(state)
        updated["external_forces"] = _scatter_set_vector_batch(
            state["external_forces"],
            self.node_indices,
            external_forces,
        )
        updated["external_torques"] = _scatter_set_vector_batch(
            state["external_torques"],
            self.elem_indices,
            external_torques,
        )
        return updated


def _uniform_index_matrix(
    start_idx: np.ndarray,
    end_idx: np.ndarray,
) -> np.ndarray:
    widths = end_idx - start_idx
    assert np.all(widths == widths[0]), "All rods must share the same discretization."
    offsets = np.arange(int(widths[0]), dtype=np.int32)
    return start_idx[:, None].astype(np.int32) + offsets[None, :]


def _gather_vector_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    if array.ndim == 3 and array.shape[1] == 3:
        return jnp.take_along_axis(array, indices[:, None, :], axis=-1)
    return jnp.moveaxis(jnp.take(array, indices, axis=-1), 1, 0)


def _gather_tensor_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    if array.ndim == 4 and array.shape[1] == 3 and array.shape[2] == 3:
        return jnp.take_along_axis(array, indices[:, None, None, :], axis=-1)
    return jnp.moveaxis(jnp.take(array, indices, axis=-1), 2, 0)


def _gather_scalar_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    if array.ndim == 2:
        return jnp.take_along_axis(array, indices, axis=-1)
    return jnp.take(array, indices, axis=-1)


def _scatter_set_vector_batch(
    array: jax.Array, indices: jax.Array, values: jax.Array
) -> jax.Array:
    if array.ndim == 3 and array.shape[1] == 3:
        del indices
        return values
    return array.at[:, indices].set(jnp.moveaxis(values, 0, 1))


def _batch_matvec_over_rods(
    matrix_collection: jax.Array,
    vector_collection: jax.Array,
) -> jax.Array:
    return jax.vmap(_jax_batch_matvec, in_axes=(0, 0))(
        matrix_collection, vector_collection
    )


def _batch_cross_over_rods(
    first_vector_collection: jax.Array,
    second_vector_collection: jax.Array,
) -> jax.Array:
    return jax.vmap(_jax_batch_cross, in_axes=(0, 0))(
        first_vector_collection,
        second_vector_collection,
    )


def _node_to_element_position_batch(position_collection: jax.Array) -> jax.Array:
    return 0.5 * (position_collection[:, :, 1:] + position_collection[:, :, :-1])


def _node_to_element_velocity_batch(
    mass: jax.Array, velocity_collection: jax.Array
) -> jax.Array:
    numerator = (
        mass[:, None, 1:] * velocity_collection[:, :, 1:]
        + mass[:, None, :-1] * velocity_collection[:, :, :-1]
    )
    denominator = mass[:, None, 1:] + mass[:, None, :-1]
    return numerator / denominator


def _node_to_element_mass_or_force_batch(nodal_collection: jax.Array) -> jax.Array:
    elemental_collection = 0.5 * (
        nodal_collection[:, :, :-1] + nodal_collection[:, :, 1:]
    )
    elemental_collection = elemental_collection.at[:, :, 0].add(
        0.5 * nodal_collection[:, :, 0]
    )
    elemental_collection = elemental_collection.at[:, :, -1].add(
        0.5 * nodal_collection[:, :, -1]
    )
    return elemental_collection


def _elements_to_nodes_batch(element_collection: jax.Array) -> jax.Array:
    node_collection = jnp.zeros(
        (
            element_collection.shape[0],
            element_collection.shape[1],
            element_collection.shape[2] + 1,
        ),
        dtype=element_collection.dtype,
    )
    node_collection = node_collection.at[:, :, :-1].add(0.5 * element_collection)
    node_collection = node_collection.at[:, :, 1:].add(0.5 * element_collection)
    return node_collection


def _find_slipping_elements_batch(
    velocity_slip: jax.Array, velocity_threshold: jax.Array
) -> jax.Array:
    abs_velocity_slip = jnp.linalg.norm(velocity_slip, axis=1)
    normalized = abs_velocity_slip / velocity_threshold - 1.0
    slipped = jnp.minimum(1.0, normalized)
    slip_function = jnp.ones_like(abs_velocity_slip)
    slip_values = jnp.abs(1.0 - slipped)
    return jnp.where(abs_velocity_slip > velocity_threshold, slip_values, slip_function)
