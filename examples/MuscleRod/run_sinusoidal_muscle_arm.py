"""Run a MuscleArm with sinusoidal activation inside a compiled JAX rollout."""

from __future__ import annotations

import argparse

import numpy as np

import elastica as ea

from batch_muscle import BatchMuscle
from memory_block_muscle_rods_jax import muscle_block_with
from muscle_rod import MuscleArm, MuscleConfig

import jax


class MuscleArmSimulator(ea.BaseSystemCollection, ea.JAXOps):
    pass


def build_arm(n_elements: int, frequency: float) -> MuscleArm:
    arm = MuscleArm.straight_rod(
        n_elements,
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        0.5,
        np.linspace(0.012, 0.0012, n_elements),
        1050.0,
        youngs_modulus=10_000.0,
    )
    activation_placeholder = np.zeros(n_elements)
    area = np.pi * arm.radius**2
    batch = (
        BatchMuscle(n_elements)
        .configure(
            force_length_weight=lambda length: np.maximum(
                0.0, 1.0 - 5.0 * (length - 1.0) ** 2
            )
        )
        .add_longitudinal_muscle(
            muscle_init_angle=0.0,
            ratio_muscle_position=0.65,
            rest_muscle_area=0.04 * area,
            max_muscle_stress=1_000.0,
            activation=activation_placeholder,
        )
        .blocking(arm)
    )
    arm.configure_muscles(
        MuscleConfig.from_batch_muscle(
            batch,
            activation_offset=0.5,
            activation_amplitude=0.5,
            activation_frequency=frequency,
        )
    )
    return arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=24)
    parser.add_argument("--final-time", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--frequency", type=float, default=2.0)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)

    simulator = MuscleArmSimulator()
    simulator.enable_block_supports(
        MuscleArm, muscle_block_with(jax.devices("cpu")[0], "float64")
    )
    arm = build_arm(args.elements, args.frequency)
    simulator.append(arm)
    simulator.using(arm).operate(ea.OneEndFixedJax)
    simulator.finalize()

    stepper = ea.PositionVerletJAX()
    final_time = stepper.integrate(
        simulator,
        time=np.float64(0.0),
        final_time=np.float64(args.final_time),
        dt=np.float64(args.dt),
    )
    block = next(iter(simulator.final_systems()))
    block.from_device(attrs=("position_collection",), update_rods=True)
    print(f"time={final_time:.6f}, tip={arm.position_collection[:, -1]}")


if __name__ == "__main__":
    main()
