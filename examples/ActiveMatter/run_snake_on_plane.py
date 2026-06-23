"""GPU JAX reproduction of the snakes-on-plane active-matter case."""

from __future__ import annotations

import click
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from contact_kernels import (
    apply_rod_contacts,
    apply_wall_contacts,
    spline_actuation_amplitude,
)
from simulation_runtime import (
    integrate_with_frame_capture,
    override_parameters,
    resolve_output_dir,
    select_device,
    simulation_run_options,
)

import elastica as ea
import elastica_jax as eaj
import jax
import jax.numpy as jnp
from jax import config as jax_config


jax_config.update("jax_enable_x64", True)

CASE_NAME = "snake-on-plane"
GRAVITY_AXIS = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class SnakeOnPlaneParameters:
    n_elements: int = 50
    n_snakes: int = 4
    length: float = 0.35
    radius_ratio: float = 0.011
    density: float = 1000.0
    youngs_modulus: float = 1.0e6
    time_period: float = 2.0
    wave_length: float = 1.0
    contact_stiffness: float = 1.0e4
    contact_damping: float = 1.0e-3
    gravitational_acc: float = -9.80665
    damping_rate: float = 0.01
    time_step: float = 5.0e-5
    final_time: float = 2.0

    @property
    def radius(self) -> float:
        return self.radius_ratio * self.length


class SnakeOnPlaneSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    pass


class _ConfiguredRodBlock(eaj.MemoryBlockCosseratRodJax):
    device = None
    device_dtype = np.dtype(np.float64)

    def __init__(self, systems, system_idx_list):
        super().__init__(
            systems,
            system_idx_list,
            device=self.device,
            device_dtype=self.device_dtype,
        )


class SnakeOnPlaneForcingAndContactJax(eaj.NoBlockOpJax):
    """Gravity, traveling-wave actuation, and capsule contact for snakes on a floor."""

    def __init__(self, *, parameters: SnakeOnPlaneParameters, _system) -> None:
        block = _system
        widths = block.end_idx_in_rod_elems - block.start_idx_in_rod_elems
        assert np.all(
            widths == widths[0]
        ), "Snake-on-plane JAX contact requires equal element counts for all rods."
        assert (
            int(widths[0]) == parameters.n_elements
        ), "Packed rod width must match the snake-on-plane element count."
        self.parameters = parameters
        offsets = np.arange(parameters.n_elements, dtype=np.int32)
        self.element_indices = (
            block.start_idx_in_rod_elems[:, None].astype(np.int32) + offsets[None, :]
        )
        self.node_indices = (
            block.start_idx_in_rod_nodes[:, None].astype(np.int32)
            + np.arange(parameters.n_elements + 1, dtype=np.int32)[None, :]
        )

        rod_id = np.repeat(np.arange(parameters.n_snakes), parameters.n_elements)
        first, second = np.triu_indices(rod_id.size, k=1)
        keep = rod_id[first] != rod_id[second]
        self.pair_first = first[keep].astype(np.int32)
        self.pair_second = second[keep].astype(np.int32)
        self.spline_amplitude = spline_actuation_amplitude(parameters.n_elements)
        self.wall_origins, self.wall_normals = make_floor()

    def jax_block_operate_synchronize(self, state, time):
        dtype = state["position_collection"].dtype
        elem = jnp.asarray(self.element_indices).reshape(-1)
        nodes = jnp.asarray(self.node_indices)

        # Build per-element capsule kinematics from packed rod nodes.
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

        # Uniform nodal gravity along the floor normal (+y).
        gravity_axis = jnp.asarray(GRAVITY_AXIS, dtype=dtype)
        gravity = self.parameters.gravitational_acc * gravity_axis
        external_forces = state["mass"][None, :] * gravity[:, None]
        external_torques = jnp.zeros_like(state["external_torques"])

        # Traveling-wave actuation with a linear ramp over one time period.
        s = jnp.arange(self.parameters.n_elements, dtype=dtype) + 0.5
        s /= self.parameters.n_elements
        wave = jnp.sin(
            2.0 * jnp.pi * time / self.parameters.time_period
            - 2.0 * jnp.pi * s / self.parameters.wave_length
        )
        ramp = jnp.minimum(1.0, time / self.parameters.time_period)
        torque_magnitude = ramp * jnp.asarray(self.spline_amplitude, dtype=dtype) * wave
        torque_world = torque_magnitude[None, None, :] * gravity_axis[None, :, None]
        torque_world = jnp.broadcast_to(
            torque_world, (self.parameters.n_snakes, 3, self.parameters.n_elements)
        )
        torque_field = jnp.einsum(
            "neij,nje->nei",
            directors.reshape(
                self.parameters.n_snakes, self.parameters.n_elements, 3, 3
            ),
            torque_world,
        )
        torque_couple = jnp.zeros_like(torque_field)
        torque_couple = torque_couple.at[:, 1:, :].add(torque_field[:, 1:, :])
        torque_couple = torque_couple.at[:, :-1, :].add(-torque_field[:, 1:, :])
        external_torques = external_torques.at[:, self.element_indices].add(
            jnp.moveaxis(torque_couple, -1, 0)
        )

        # Rod-rod capsule contact and broad-phase candidate cache update.
        (
            external_forces,
            external_torques,
            candidate_mask,
            last_detection_time,
        ) = apply_rod_contacts(
            pair_first=self.pair_first,
            pair_second=self.pair_second,
            centers=centers,
            velocities=element_velocity,
            axes=axes,
            lengths=lengths,
            radii=radii,
            omega=omega_world,
            directors=directors,
            elem=elem,
            external_forces=external_forces,
            external_torques=external_torques,
            cached_candidates=state["active_matter_candidate_mask"],
            last_detection_time=state["active_matter_last_detection_time"],
            time=time,
            contact_stiffness=self.parameters.contact_stiffness,
            contact_damping=self.parameters.contact_damping,
            steps_between_detection=0,
            time_step=self.parameters.time_step,
        )
        # Floor half-space contact at y = 0.
        external_forces, external_torques = apply_wall_contacts(
            wall_origins=self.wall_origins,
            wall_normals=self.wall_normals,
            centers=centers,
            velocities=element_velocity,
            axes=axes,
            lengths=lengths,
            radii=radii,
            omega=omega_world,
            directors=directors,
            elem=elem,
            external_forces=external_forces,
            external_torques=external_torques,
            contact_stiffness=self.parameters.contact_stiffness,
            contact_damping=self.parameters.contact_damping,
        )
        # Publish external loads and contact-detection cache for the stepper.
        updated = dict(state)
        updated["external_forces"] = external_forces
        updated["external_torques"] = external_torques
        updated["active_matter_candidate_mask"] = candidate_mask
        updated["active_matter_last_detection_time"] = last_detection_time
        return updated


def make_floor() -> tuple[np.ndarray, np.ndarray]:
    return np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])


def initial_rods(
    parameters: SnakeOnPlaneParameters,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.array([3.0 * parameters.radius * i, parameters.radius, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        for i in range(parameters.n_snakes)
    ]


def build_simulator(parameters: SnakeOnPlaneParameters, *, device, dtype):
    assert parameters.n_elements >= 2, "Each snake must contain at least two elements."
    assert parameters.n_snakes >= 1, "The case needs at least one snake."
    assert parameters.time_step > 0.0, "The simulation time step must be positive."
    _ConfiguredRodBlock.device = device
    _ConfiguredRodBlock.device_dtype = np.dtype(dtype)
    simulator = SnakeOnPlaneSimulator()
    simulator.enable_block_supports(ea.CosseratRod, _ConfiguredRodBlock)
    shear_modulus = parameters.youngs_modulus / 1.5
    for start, direction in initial_rods(parameters):
        normal_seed = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(normal_seed, direction)) > 0.9:
            normal_seed = np.array([0.0, 1.0, 0.0])
        normal = normal_seed - np.dot(normal_seed, direction) * direction
        normal /= np.linalg.norm(normal)
        rod = ea.CosseratRod.straight_rod(
            parameters.n_elements,
            start,
            direction,
            normal,
            parameters.length,
            parameters.radius,
            parameters.density,
            youngs_modulus=parameters.youngs_modulus,
            shear_modulus=shear_modulus,
        )
        simulator.append(rod)
    simulator.operate_block(ea.CosseratRod).using(
        SnakeOnPlaneForcingAndContactJax, parameters=parameters
    )
    simulator.operate_block(ea.CosseratRod).using(
        eaj.AnalyticalLinearDamperJax,
        damping_constant=parameters.damping_rate,
        time_step=parameters.time_step,
    )
    simulator.finalize()
    block = tuple(simulator.final_systems())[0]
    assert isinstance(
        block, eaj.MemoryBlockCosseratRodJax
    ), "Snake-on-plane requires a JAX Cosserat-rod memory block."
    state = block.jax_get_state()
    pair_count = sum(
        parameters.n_elements * parameters.n_elements
        for _ in range(parameters.n_snakes * (parameters.n_snakes - 1) // 2)
    )
    state["active_matter_candidate_mask"] = jax.device_put(
        np.zeros(pair_count, dtype=bool), device=device
    )
    state["active_matter_last_detection_time"] = jax.device_put(
        np.asarray(-np.inf, dtype=dtype), device=device
    )
    block.jax_set_state(state)
    return simulator, block


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@simulation_run_options(backends=("auto", "cpu", "gpu", "cuda", "mps"))
def main(
    backend: str,
    n_elements: int | None,
    n_snakes: int | None,
    final_time: float | None,
    time_step: float | None,
    seed: int,
    run_name: str | None,
    fps: float,
    save_workers: int,
) -> None:
    parameters = override_parameters(
        SnakeOnPlaneParameters(),
        n_elements=n_elements,
        n_snakes=n_snakes,
        final_time=final_time,
        time_step=time_step,
    )
    assert parameters.final_time >= 0.0, "Final simulation time must be nonnegative."

    backend_name, device = select_device(backend)
    dtype = np.float64 if device.platform.lower() == "cpu" else np.float32
    simulator, block = build_simulator(parameters, device=device, dtype=dtype)
    wall_origins, wall_normals = make_floor()
    output_dir = resolve_output_dir(run_name)
    integrate_with_frame_capture(
        simulator=simulator,
        block=block,
        parameters=parameters,
        case_name=CASE_NAME,
        n_snakes=parameters.n_snakes,
        radius=parameters.radius,
        final_time=parameters.final_time,
        time_step=parameters.time_step,
        output_dir=output_dir,
        fps=fps,
        save_workers=save_workers,
        wall_origins=wall_origins,
        wall_normals=wall_normals,
        seed=seed,
        run_name=run_name,
    )
    print(f"backend={backend_name} device={device} dtype={dtype}")


if __name__ == "__main__":
    main()
