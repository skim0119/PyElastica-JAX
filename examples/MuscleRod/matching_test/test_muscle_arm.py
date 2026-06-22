"""Match embedded MuscleArm loads against the legacy BatchMuscle actuator."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

import sys

sys.path.append("/Users/skim0119/github/PyElastica-gpu/examples/MuscleRod")

from batch_muscle import BatchMuscle
from muscle_rod import MuscleArm, MuscleConfig


N_ELEMENTS = 8


def _make_rod() -> MuscleArm:
    return MuscleArm.straight_rod(
        N_ELEMENTS,
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        0.5,
        np.linspace(0.012, 0.0012, N_ELEMENTS),
        1050.0,
        youngs_modulus=10_000.0,
    )


def _build_batch(rod: MuscleArm, activation: np.ndarray) -> BatchMuscle:
    area = np.pi * rod.radius**2
    return (
        BatchMuscle(rod.n_elems)
        .configure(
            force_length_weight=lambda length: np.maximum(
                0.0, 1.0 - 5.0 * (length - 1.0) ** 2
            )
        )
        .add_transverse_muscle(
            rest_muscle_area=0.08 * area,
            max_muscle_stress=1_000.0,
            activation=activation,
        )
        .add_longitudinal_muscle(
            muscle_init_angle=0.25 * np.pi,
            ratio_muscle_position=0.65,
            rest_muscle_area=0.04 * area,
            max_muscle_stress=1_200.0,
            activation=activation.copy(),
        )
        .add_oblique_muscle(
            muscle_init_angle=0.0,
            ratio_muscle_position=0.8,
            rotation_number=2.0,
            rest_muscle_area=0.02 * area,
            max_muscle_stress=900.0,
            activation=activation.copy(),
        )
        .blocking(rod)
    )


def _deform_rod(rod: MuscleArm) -> None:
    s = np.linspace(0.0, 1.0, rod.n_nodes)
    rod.position_collection[1] += 0.02 * s**2


@pytest.mark.parametrize("time", [0.0, 0.125, 0.375])
def test_muscle_arm_matches_batch_muscle_loads(time: float) -> None:
    activation_value = 0.5 * (1.0 + np.sin(2.0 * np.pi * time))

    legacy_rod = _make_rod()
    legacy_activation = np.full(N_ELEMENTS, activation_value)
    legacy_batch = _build_batch(legacy_rod, legacy_activation)
    _deform_rod(legacy_rod)
    legacy_rod.compute_internal_forces_and_torques(np.float64(time))
    legacy_batch.forward(legacy_rod)

    muscle_arm = _make_rod()
    config_batch = _build_batch(muscle_arm, np.zeros(N_ELEMENTS))
    muscle_arm.configure_muscles(
        MuscleConfig.from_batch_muscle(
            config_batch,
            activation_offset=0.5,
            activation_amplitude=0.5,
            activation_frequency=1.0,
        )
    )
    _deform_rod(muscle_arm)
    muscle_arm.compute_internal_forces_and_torques(np.float64(time))

    assert_allclose(
        muscle_arm.internal_forces,
        legacy_rod.internal_forces + legacy_rod.external_forces,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert_allclose(
        muscle_arm.internal_torques,
        legacy_rod.internal_torques + legacy_rod.external_torques,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_jax_muscle_arm_block_matches_cpu() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from ..memory_block_muscle_rods_jax import MemoryBlockMuscleArmJax

    rod = _make_rod()
    batch = _build_batch(rod, np.zeros(N_ELEMENTS))
    rod.configure_muscles(
        MuscleConfig.from_batch_muscle(
            batch,
            activation_offset=0.5,
            activation_amplitude=0.5,
            activation_frequency=1.0,
        )
    )
    _deform_rod(rod)
    time = np.float64(0.125)
    rod.compute_internal_forces_and_torques(time)

    cpu = jax.devices("cpu")[0]
    block = MemoryBlockMuscleArmJax([rod], [0], device=cpu, device_dtype=np.float64)
    with jax.default_device(cpu):
        with jax.disable_jit():
            state = block.jax_compute_internal_forces_and_torques(
                block.jax_get_state(), time
            )

    assert_allclose(state["internal_forces"], rod.internal_forces, rtol=1.0e-10)
    assert_allclose(state["internal_torques"], rod.internal_torques, rtol=1.0e-10)
