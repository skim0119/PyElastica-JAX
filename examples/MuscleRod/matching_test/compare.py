from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import jax
from tqdm import tqdm

import elastica_jax as eaj
from elastica import (
    BaseSystemCollection,
    Constraints,
    CosseratRod,
    Forcing,
    OneEndFixedBC,
    PositionVerlet,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from old_impl_batch_muscle import BatchMuscle, ApplyMuscleActuations
from memory_block_muscle_rods_jax import muscle_block_with
from muscle_rod import MuscleArm, MuscleConfig

N_TM = 1
N_LM = 4
N_OM = 6
N_ELEMENTS = 50
FINAL_TIME = 2.0
TIME_STEP = 1.0e-4
ACTIVATION_FREQUENCY = 1.0


class BaseSimulator(BaseSystemCollection, Constraints, Forcing):
    pass


def sinusoidal_activation(time: float) -> float:
    return 0.5 * (1.0 + np.sin(2.0 * np.pi * ACTIVATION_FREQUENCY * time))


def make_rod() -> CosseratRod:
    base_length = 0.5
    radius_base = 0.012
    radius_tip = 0.0012
    radius = np.linspace(radius_base, radius_tip, N_ELEMENTS + 1)
    radius_mean = (radius[:-1] + radius[1:]) / 2

    return CosseratRod.straight_rod(
        n_elements=N_ELEMENTS,
        start=np.zeros((3,)),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, -1.0]),
        base_length=base_length,
        base_radius=radius_mean.copy(),
        density=1050,
        youngs_modulus=10_000,
    )


def force_length_weight(normalized_length: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - 5.0 * (normalized_length - 1.0) ** 2)


def make_muscle_rod(batch_muscle) -> MuscleArm:
    base_length = 0.5
    radius_base = 0.012
    radius_tip = 0.0012
    radius = np.linspace(radius_base, radius_tip, N_ELEMENTS + 1)
    radius_mean = (radius[:-1] + radius[1:]) / 2

    muscle_config = MuscleConfig.from_batch_muscle(
        batch_muscle,
        activation_offset=0.5,
        activation_amplitude=0.5,
        activation_frequency=ACTIVATION_FREQUENCY,
    )

    arm = MuscleArm.straight_rod(
        n_elements=N_ELEMENTS,
        start=np.zeros((3,)),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, -1.0]),
        base_length=base_length,
        base_radius=radius_mean.copy(),
        density=1050,
        youngs_modulus=10_000.0,
        muscle_config=muscle_config,
    )
    # TODO: use muscle parameters to set arm variables
    return arm


def muscle_parameters(base_radius: float, rod: CosseratRod) -> dict:
    lm_ratio_muscle_position = 0.0075 / base_radius
    om_ratio_muscle_position = 0.01125 / base_radius

    an_ratio_radius = 0.002 / base_radius
    tm_ratio_radius = 0.0045 / base_radius
    lm_ratio_radius = 0.003 / base_radius
    om_ratio_radius = 0.00075 / base_radius

    shearable_rod_area = np.pi * rod.radius**2
    return dict(
        lm_ratio_muscle_position=lm_ratio_muscle_position,
        om_ratio_muscle_position=om_ratio_muscle_position,
        tm_rest_muscle_area=shearable_rod_area
        * (tm_ratio_radius**2 - an_ratio_radius**2),
        lm_rest_muscle_area=shearable_rod_area * (lm_ratio_radius**2),
        om_rest_muscle_area=shearable_rod_area * (om_ratio_radius**2),
        tm_max_muscle_stress=1000.0,
        lm_max_muscle_stress=1000.0,
        om_max_muscle_stress=1000.0,
        om_rotation_number=6,
    )


def build_batch_muscle(
    rod: CosseratRod,
    params: dict,
) -> tuple[BatchMuscle, list[np.ndarray]]:
    batch = BatchMuscle(num_elements=rod.n_elems).configure(
        force_length_weight=force_length_weight
    )
    activations: list[np.ndarray] = []

    for _ in range(N_TM):
        activation = np.zeros(rod.n_elems)
        activations.append(activation)
        batch.add_transverse_muscle(
            rest_muscle_area=params["tm_rest_muscle_area"],
            max_muscle_stress=params["tm_max_muscle_stress"],
            activation=activation,
        )

    for k in range(N_LM):
        activation = np.zeros(rod.n_elems)
        activations.append(activation)
        batch.add_longitudinal_muscle(
            muscle_init_angle=2.0 * np.pi * k / N_LM,
            ratio_muscle_position=params["lm_ratio_muscle_position"],
            rest_muscle_area=params["lm_rest_muscle_area"],
            max_muscle_stress=params["lm_max_muscle_stress"],
            activation=activation,
        )

    for k in range(N_OM):
        activation = np.zeros(rod.n_elems)
        activations.append(activation)
        batch.add_oblique_muscle(
            muscle_init_angle=2.0 * np.pi * k / N_OM,
            ratio_muscle_position=params["om_ratio_muscle_position"],
            rotation_number=(
                params["om_rotation_number"]
                if k % 2 == 0
                else -params["om_rotation_number"]
            ),
            rest_muscle_area=params["om_rest_muscle_area"],
            max_muscle_stress=params["om_max_muscle_stress"],
            activation=activation,
        )

    return batch, activations


def set_batch_activations(activations: list[np.ndarray], value: float) -> None:
    for activation in activations:
        activation[:] = value


def tip_relative_error(tip_a: np.ndarray, tip_b: np.ndarray) -> float:
    tip_diff = np.linalg.norm(tip_a - tip_b)
    tip_ref = max(np.linalg.norm(tip_a), np.linalg.norm(tip_b), 1e-12)
    return tip_diff / tip_ref


def run_simulation(
    rod: CosseratRod,
    activations: list[np.ndarray] | None,
    batch_muscle: BatchMuscle | None,
) -> tuple[float, np.ndarray]:
    # Setup
    simulator = BaseSimulator()
    simulator.append(rod)
    simulator.constrain(rod).using(
        OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    simulator.add_forcing_to(rod).using(
        ApplyMuscleActuations, batch_muscle=batch_muscle
    )

    simulator.finalize()
    stepper = PositionVerlet()
    total_steps = int(FINAL_TIME / TIME_STEP)

    # Run
    sim_time = np.float64(0.0)
    start = time.perf_counter()
    for n_step in tqdm(range(total_steps)):
        activation = sinusoidal_activation(float(sim_time))
        set_batch_activations(activations, activation)

        sim_time = stepper.step(simulator, sim_time, TIME_STEP)
    elapsed = time.perf_counter() - start

    # Return
    tip_position = rod.position_collection[:, -1].copy()
    print(f"COOMM: {elapsed:8.3f} s  ({total_steps / elapsed:,.0f} steps/s)")
    return elapsed, tip_position


def run_jax_simulation(
    muscle_rod: CosseratRod,
) -> tuple[float, np.ndarray]:
    # Setup
    simulator = eaj.Simulator()
    simulator.enable_block_supports(MuscleArm, muscle_block_with("cpu", "float64"))

    simulator.append(muscle_rod)
    simulator.operate(muscle_rod).using(
        eaj.OneEndFixedJax,
    )

    simulator.finalize()
    stepper = eaj.PositionVerletJAX()
    total_steps = int(FINAL_TIME / TIME_STEP)

    # Run
    start = time.perf_counter()
    stepper.integrate(
        simulator,
        time=np.float64(0.0),
        final_time=FINAL_TIME,
        dt=TIME_STEP,
    )
    elapsed = time.perf_counter() - start

    block = next(iter(simulator.final_systems()))
    block.from_device(attrs=("position_collection",), update_rods=True)
    tip_position = muscle_rod.position_collection[:, -1].copy()

    # Return
    print(f"JAX:   {elapsed:8.3f} s  ({total_steps / elapsed:,.0f} steps/s)")
    return elapsed, tip_position


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    n_muscles = N_TM + N_LM + N_OM
    total_steps = int(FINAL_TIME / TIME_STEP)

    print("COOMM muscle actuation benchmark")
    print(f"  muscles: {N_TM} TM + {N_LM} LM + {N_OM} OM = {n_muscles}")
    print(f"  rod elements: {N_ELEMENTS}")
    print(
        f"  simulation: {FINAL_TIME} s physical time, dt={TIME_STEP}, {total_steps} steps"
    )
    print(f"  activation: sin(2π·{ACTIVATION_FREQUENCY}·t) mapped to [0, 1]")
    print()

    base_radius = 0.012

    rod_orig = make_rod()
    params = muscle_parameters(base_radius, rod_orig)
    batch_muscle, activations = build_batch_muscle(
        rod_orig,
        params,
    )
    batch_muscle.blocking(rod_orig)
    batch_elapsed, batch_tip = run_simulation(
        rod_orig,
        activations,
        batch_muscle=batch_muscle,
    )
    print(batch_tip)

    # TODO: need to not wire through batch_muscle
    muscle_rod = make_muscle_rod(batch_muscle)
    jax_elapsed, jax_tip = run_jax_simulation(
        muscle_rod,
    )
    print(jax_tip)

    print()
    print(f"COOMM vs JAX: {batch_elapsed / jax_elapsed:.2f}x")
    print()
    print(
        f"Tip rel. error COOMM vs JAX at t={FINAL_TIME} s: {tip_relative_error(jax_tip, batch_tip):.3e}"
    )


if __name__ == "__main__":
    main()
