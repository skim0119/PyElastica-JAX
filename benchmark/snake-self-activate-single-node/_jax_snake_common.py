"""Shared helpers for multi-snake Numba vs JAX benchmarks."""

from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import elastica as ea
import elastica_jax as eaj

import jax
import jax.numpy as jnp

from elastica_jax._linalg import _jax_batch_cross, _jax_batch_matvec


jax.config.update("jax_enable_x64", True)

DEFAULT_PERIOD = 2.0
DEFAULT_BASE_LENGTH = 0.35
DEFAULT_DENSITY = 1000.0
DEFAULT_YOUNGS_MODULUS = 1.0e6
DEFAULT_POISSON_RATIO = 0.5
DEFAULT_GRAVITY = -9.80665
DEFAULT_DAMPING = 2.0e-3
DEFAULT_FROUDE = 0.1
DEFAULT_N_ELEM = 50
DEFAULT_DT = 1.0e-4


def default_b_coeff() -> np.ndarray:
    return np.array(
        [3.4e-3, 3.3e-3, 4.2e-3, 2.6e-3, 3.6e-3, 3.5e-3, 1.0],
        dtype=np.float64,
    )


def build_rod(
    *,
    n_elem: int,
    base_length: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
) -> ea.CosseratRod:
    base_radius = base_length * 0.011
    shear_modulus = youngs_modulus / (poisson_ratio + 1.0)
    return ea.CosseratRod.straight_rod(
        n_elem,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        base_length,
        base_radius,
        density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )


class MultiSnakeReferenceSimulator(
    ea.BaseSystemCollection, ea.Forcing, ea.Damping, ea.Contact
):
    pass


class MultiSnakeJAXBlockSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    pass


def _uniform_index_matrix(
    start_idx: np.ndarray,
    end_idx: np.ndarray,
) -> np.ndarray:
    widths = end_idx - start_idx
    assert np.all(widths == widths[0]), "All rods must share the same discretization."
    offsets = np.arange(int(widths[0]), dtype=np.int32)
    return start_idx[:, None].astype(np.int32) + offsets[None, :]


def _gather_vector_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.moveaxis(jnp.take(array, indices, axis=-1), 1, 0)


def _gather_tensor_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.moveaxis(jnp.take(array, indices, axis=-1), 2, 0)


def _gather_scalar_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.take(array, indices, axis=-1)


def _scatter_set_vector_batch(
    array: jax.Array, indices: jax.Array, values: jax.Array
) -> jax.Array:
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


class SnakeMuscleTorquesBlockJax(eaj.NoBlockOpJax):
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
        template = ea.MuscleTorques(
            base_length=base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / float(b_coeff[-1]),
            phase_shift=0.0,
            direction=np.array([0.0, 1.0, 0.0]),
            rest_lengths=np.asarray(
                _system.rest_lengths[
                    _system.start_idx_in_rod_elems[0] : _system.end_idx_in_rod_elems[0]
                ]
            ),
            ramp_up_time=period,
            with_spline=True,
        )
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
        gravitational_acc: float,
        _system,
    ) -> None:
        del static_mu_array
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
        self.gravity = jnp.asarray(
            np.array([0.0, gravitational_acc, 0.0], dtype=np.float64)
        )
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
        unitized_total_velocity = (
            unitized_total_velocity
            / (jnp.linalg.norm(unitized_total_velocity + 1.0e-14, axis=1)[:, None, :])
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


def snake_start(index: int, spacing: float) -> np.ndarray:
    return np.array([index * spacing, 0.0, 0.0], dtype=np.float64)


def build_cpu_sim(
    *,
    n_snakes: int,
    n_elem: int = DEFAULT_N_ELEM,
    period: float = DEFAULT_PERIOD,
    base_length: float = DEFAULT_BASE_LENGTH,
    density: float = DEFAULT_DENSITY,
    youngs_modulus: float = DEFAULT_YOUNGS_MODULUS,
    poisson_ratio: float = DEFAULT_POISSON_RATIO,
    gravitational_acc: float = DEFAULT_GRAVITY,
    time_step: float = DEFAULT_DT,
    include_external_loads: bool = True,
) -> tuple[MultiSnakeReferenceSimulator, list[ea.CosseratRod]]:
    b_coeff = default_b_coeff()
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    wave_length = float(b_coeff[-1])
    mu = base_length / (period * period * np.abs(gravitational_acc) * DEFAULT_FROUDE)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)
    spacing = 1.5 * base_length

    sim = MultiSnakeReferenceSimulator()
    rods: list[ea.CosseratRod] = []
    if include_external_loads:
        ground_plane = ea.Plane(
            plane_origin=np.array([0.0, -base_length * 0.011, 0.0], dtype=np.float64),
            plane_normal=normal,
        )
        sim.append(ground_plane)

    for idx in range(n_snakes):
        rod = build_rod(
            n_elem=n_elem,
            base_length=base_length,
            density=density,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
        )
        start = snake_start(idx, spacing)
        rod.position_collection[...] = rod.position_collection + start[:, None]
        sim.append(rod)
        if include_external_loads:
            sim.add_forcing_to(rod).using(
                ea.GravityForces,
                acc_gravity=np.array([0.0, gravitational_acc, 0.0], dtype=np.float64),
            )
            sim.add_forcing_to(rod).using(
                ea.MuscleTorques,
                base_length=base_length,
                b_coeff=b_coeff[:-1],
                period=period,
                wave_number=2.0 * np.pi / wave_length,
                phase_shift=0.0,
                rest_lengths=rod.rest_lengths,
                ramp_up_time=period,
                direction=normal,
                with_spline=True,
            )
            sim.detect_contact_between(rod, ground_plane).using(
                ea.RodPlaneContactWithAnisotropicFriction,
                k=1.0,
                nu=1.0e-6,
                slip_velocity_tol=1.0e-8,
                static_mu_array=static_mu_array,
                kinetic_mu_array=kinetic_mu_array,
            )
            sim.dampen(rod).using(
                ea.AnalyticalLinearDamper,
                damping_constant=DEFAULT_DAMPING,
                time_step=time_step,
            )
        rods.append(rod)

    sim.finalize()
    return sim, rods


def build_jax_sim(
    *,
    device: jax.Device,
    device_dtype: np.dtype,
    n_snakes: int,
    n_elem: int = DEFAULT_N_ELEM,
    period: float = DEFAULT_PERIOD,
    base_length: float = DEFAULT_BASE_LENGTH,
    density: float = DEFAULT_DENSITY,
    youngs_modulus: float = DEFAULT_YOUNGS_MODULUS,
    poisson_ratio: float = DEFAULT_POISSON_RATIO,
    gravitational_acc: float = DEFAULT_GRAVITY,
    time_step: float = DEFAULT_DT,
    include_external_loads: bool = True,
) -> tuple[MultiSnakeJAXBlockSimulator, eaj._CosseratRodMemoryBlock]:
    b_coeff = default_b_coeff()
    mu = base_length / (period * period * np.abs(gravitational_acc) * DEFAULT_FROUDE)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)
    spacing = 1.5 * base_length

    sim = MultiSnakeJAXBlockSimulator()
    rod_block = eaj.configure_rod_block(
        device=device,
        device_dtype=np.dtype(device_dtype),
    )
    sim.enable_block_supports(ea.CosseratRod, rod_block)
    for idx in range(n_snakes):
        rod = build_rod(
            n_elem=n_elem,
            base_length=base_length,
            density=density,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
        )
        start = snake_start(idx, spacing)
        rod.position_collection[...] = rod.position_collection + start[:, None]
        sim.append(rod)

    if include_external_loads:
        sim.operate_block(ea.CosseratRod).using(
            SnakeMuscleTorquesBlockJax,
            b_coeff=b_coeff,
            period=period,
            base_length=base_length,
        )
        sim.operate_block(ea.CosseratRod).using(
            GravityPlaneContactBlockJax,
            plane_origin=np.array([0.0, -base_length * 0.011, 0.0], dtype=np.float64),
            plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            slip_velocity_tol=1.0e-8,
            k=1.0,
            nu=1.0e-6,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
            gravitational_acc=gravitational_acc,
        )
        sim.operate_block(ea.CosseratRod).using(
            eaj.AnalyticalLinearDamperJax,
            time_step=np.float64(time_step),
            damping_constant=DEFAULT_DAMPING,
        )

    sim.finalize()
    return sim, rod_block


def time_average(n_iter: int, fn) -> float:  # type: ignore[no-untyped-def]
    assert n_iter > 0, "n_iter must be positive."
    start = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - start) / n_iter


def snapshot_jax_state_to_host(
    state: dict[str, jax.Array],
) -> dict[str, np.ndarray]:
    host_state = jax.device_get(state)
    return {key: np.asarray(value).copy() for key, value in host_state.items()}


def restore_jax_state_from_host(
    host_state: dict[str, np.ndarray],
    device: jax.Device,
) -> dict[str, jax.Array]:
    return {
        key: jax.device_put(np.asarray(value), device=device)
        for key, value in host_state.items()
    }


def save_jax_state_npz(path: Path, state: dict[str, np.ndarray]) -> None:
    np.savez(path, **state)


def load_jax_state_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(value).copy() for key, value in data.items()}


def emit_report(lines: list[str], log_path: Path | None) -> None:
    report = "\n".join(lines)
    print(report)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(report + "\n", encoding="utf-8")


type BenchmarkTiming = tuple[float, float]


def run_pyelastica_rollout(
    *,
    n_snakes: int,
    steps: int,
    warmup_runs: int,
    n_elem: int = DEFAULT_N_ELEM,
    dt: float = DEFAULT_DT,
    include_external_loads: bool = True,
) -> BenchmarkTiming:
    """Build a PyElastica simulator and time a fixed-length Position Verlet rollout."""
    dt_value = np.float64(dt)
    final_time = np.float64(steps * dt)

    instantiate_start = time.perf_counter()
    cpu_sim, _cpu_rods = build_cpu_sim(
        n_snakes=n_snakes,
        n_elem=n_elem,
        time_step=dt,
        include_external_loads=include_external_loads,
    )
    instantiate_seconds = time.perf_counter() - instantiate_start

    stepper = ea.PositionVerlet()
    if warmup_runs > 0:
        warmup_sim, _ = build_cpu_sim(
            n_snakes=n_snakes,
            n_elem=n_elem,
            time_step=dt,
            include_external_loads=include_external_loads,
        )
        warmup_stepper = ea.PositionVerlet()
        for _ in range(warmup_runs):
            warmup_time = np.float64(0.0)
            for _ in range(steps):
                warmup_time = warmup_stepper.step(warmup_sim, warmup_time, dt_value)

    time_value = np.float64(0.0)
    rollout_start = time.perf_counter()
    for _ in range(steps):
        time_value = stepper.step(cpu_sim, time_value, dt_value)
    rollout_seconds = time.perf_counter() - rollout_start
    assert np.isclose(time_value, final_time), (
        "PyElastica rollout did not end at final_time."
    )
    return instantiate_seconds, rollout_seconds


def run_jax_rollout(
    *,
    backend: str,
    n_snakes: int,
    steps: int,
    warmup_runs: int,
    transfer_guard: str,
    n_elem: int = DEFAULT_N_ELEM,
    dt: float = DEFAULT_DT,
    include_external_loads: bool = True,
) -> BenchmarkTiming:
    """Build a JAX block simulator and time a fixed-length Position Verlet rollout."""
    dt_value = np.float64(dt)
    final_time = np.float64(steps * dt)
    dtype = np.dtype(np.float64)
    device = eaj.resolve_backend_devices(backend)[0]

    with jax.default_device(device):
        instantiate_start = time.perf_counter()
        jax_sim, jax_block = build_jax_sim(
            device=device,
            device_dtype=dtype,
            n_snakes=n_snakes,
            n_elem=n_elem,
            time_step=dt,
            include_external_loads=include_external_loads,
        )
        jax.block_until_ready(jax_block)
        instantiate_seconds = time.perf_counter() - instantiate_start

        stepper = eaj.PositionVerletJAX()
        guard_context = (
            jax.transfer_guard(transfer_guard)
            if transfer_guard != "allow"
            else nullcontext()
        )
        with guard_context:
            for _ in range(warmup_runs):
                initial_state = dict(jax_block.jax_get_state())
                stepper.integrate(
                    jax_sim,
                    time=np.float64(0.0),
                    final_time=final_time,
                    dt=dt_value,
                )
                jax.block_until_ready(jax_block)
                jax_block.jax_set_state(initial_state)

            rollout_start = time.perf_counter()
            stepper.integrate(
                jax_sim,
                time=np.float64(0.0),
                final_time=final_time,
                dt=dt_value,
            )
            jax.block_until_ready(jax_block)
            rollout_seconds = time.perf_counter() - rollout_start

    return instantiate_seconds, rollout_seconds
