"""Multi-device (shard_map) vertical CapsuleContact tests.

Public seams:
- CapsuleContactOp on a rod-sharded vertical block spans cross-shard pairs
- Cross-shard contact forces match a single-device vertical reference
- Short PositionVerlet rollouts succeed on a two-device vertical block
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402
from elastica_jax.contact.capsule_metadata import (  # noqa: E402
    CONTACT_STATE_KEYS,
    build_block_capsule_metadata,
    install_capsule_contact_state,
)

N_ELEMENTS = 4
CONTACT_STIFFNESS = 1.0e4
CONTACT_DAMPING = 1.0e-2
TIME_STEP = 1.0e-4


def _overlapping_rod(*, start: np.ndarray) -> ea.CosseratRod:
    return ea.CosseratRod.straight_rod(
        N_ELEMENTS,
        start,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        0.35,
        0.02,
        1000.0,
        youngs_modulus=1.0e5,
    )


def _contact_external_forces(
    *,
    device: jax.Device | tuple[jax.Device, ...],
    starts: tuple[np.ndarray, ...],
) -> np.ndarray:
    primary = device[0] if isinstance(device, tuple) else device
    with jax.default_device(primary):
        simulator = eaj.Simulator()
        rod_block = eaj.configure_rod_block(
            device=device,
            inner_block_cls=eaj._CosseratRodVerticalMemoryBlock,
        )
        simulator.enable_block_supports(ea.CosseratRod, rod_block)
        for start in starts:
            simulator.append(_overlapping_rod(start=start))
        simulator.operate_block(rod_block).using(
            eaj.CapsuleContactOp,
            n_elements_per_rod=N_ELEMENTS,
            contact_stiffness=CONTACT_STIFFNESS,
            contact_damping=CONTACT_DAMPING,
            steps_between_detection=0,
            time_step=TIME_STEP,
            broad_phase="all_pairs",
        )
        simulator.finalize()
        metadata = build_block_capsule_metadata(
            rod_block,
            n_elements_per_rod=N_ELEMENTS,
            broad_phase="all_pairs",
        )
        install_capsule_contact_state(
            rod_block,
            metadata,
            device=primary,
            dtype=rod_block.device_dtype,
        )
        for key in CONTACT_STATE_KEYS:
            assert key in rod_block.jax_get_state()
        state = rod_block.jax_compute_internal_forces_and_torques(
            rod_block.jax_get_state(),
            np.float64(0.0),
        )
        state = rod_block.jax_zero_external_loads(state, np.float64(0.0))
        updated = simulator.jax_synchronize((state,), np.float64(0.0))[0]
        return np.asarray(updated["external_forces"])


def test_multidevice_vertical_contact_forces_match_single_device() -> None:
    devices = tuple(jax.devices("cpu")[:2])
    if len(devices) < 2:
        pytest.skip("requires at least two CPU devices")
    # Two rods => one rod per shard; any contact pair is cross-shard.
    starts = (np.zeros(3), np.array([0.03, 0.0, 0.0]))
    single = _contact_external_forces(device=devices[0], starts=starts)
    multi = _contact_external_forces(device=devices, starts=starts)
    assert single.shape == (2, 3, N_ELEMENTS + 1)
    assert float(np.linalg.norm(single)) > 0.0
    assert_allclose(multi, single, rtol=1.0e-10, atol=1.0e-10)


def test_multidevice_vertical_capsule_contact_rollout() -> None:
    devices = tuple(jax.devices("cpu")[:2])
    if len(devices) < 2:
        pytest.skip("requires at least two CPU devices")
    starts = (np.zeros(3), np.array([0.03, 0.0, 0.0]))
    with jax.default_device(devices[0]):
        simulator = eaj.Simulator()
        rod_block = eaj.configure_rod_block(
            device=devices,
            inner_block_cls=eaj._CosseratRodVerticalMemoryBlock,
        )
        simulator.enable_block_supports(ea.CosseratRod, rod_block)
        for start in starts:
            simulator.append(_overlapping_rod(start=start))
        simulator.operate_block(rod_block).using(
            eaj.CapsuleContactOp,
            n_elements_per_rod=N_ELEMENTS,
            contact_stiffness=CONTACT_STIFFNESS,
            contact_damping=CONTACT_DAMPING,
            steps_between_detection=0,
            time_step=TIME_STEP,
            broad_phase="all_pairs",
        )
        simulator.finalize()
        metadata = build_block_capsule_metadata(
            rod_block,
            n_elements_per_rod=N_ELEMENTS,
            broad_phase="all_pairs",
        )
        install_capsule_contact_state(
            rod_block,
            metadata,
            device=devices[0],
            dtype=rod_block.device_dtype,
        )
        stepper = eaj.PositionVerletJAX()
        time_value = stepper.integrate(
            simulator,
            time=np.float64(0.0),
            final_time=np.float64(5 * TIME_STEP),
            dt=np.float64(TIME_STEP),
        )
        assert float(time_value) == pytest.approx(5 * TIME_STEP)
