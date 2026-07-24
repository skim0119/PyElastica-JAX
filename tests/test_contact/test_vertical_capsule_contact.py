"""Tests for capsule contact on stacked (vertical) Cosserat rod blocks."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402
from elastica_jax.contact.capsule_metadata import (  # noqa: E402
    CONTACT_STATE_KEYS,
    build_block_capsule_metadata,
    capsule_kinematics_from_block_state,
    install_capsule_contact_state,
)

CPU_DEVICE = jax.devices("cpu")[0]
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


def _build_block(
    *,
    vertical: bool,
    starts: tuple[np.ndarray, ...],
) -> eaj._CosseratRodMemoryBlock | eaj._CosseratRodVerticalMemoryBlock:
    with jax.default_device(CPU_DEVICE):
        simulator = eaj.Simulator()
        inner_block_cls = (
            eaj._CosseratRodVerticalMemoryBlock
            if vertical
            else eaj._CosseratRodMemoryBlock
        )
        rod_block = eaj.configure_rod_block(
            device=CPU_DEVICE,
            inner_block_cls=inner_block_cls,
        )
        simulator.enable_block_supports(ea.CosseratRod, rod_block)
        for start in starts:
            simulator.append(_overlapping_rod(start=start))
        simulator.finalize()
        return rod_block


def test_build_block_capsule_metadata_accepts_vertical_block() -> None:
    block = _build_block(
        vertical=True,
        starts=(np.zeros(3), np.array([0.03, 0.0, 0.0])),
    )
    metadata = build_block_capsule_metadata(block, n_elements_per_rod=N_ELEMENTS)

    assert metadata.layout == "stacked"
    assert metadata.n_rods == 2
    assert metadata.n_elements_per_rod == N_ELEMENTS
    assert metadata.n_capsules == 2 * N_ELEMENTS
    assert_array_equal(
        metadata.rod_ids,
        np.repeat(np.arange(2, dtype=np.int32), N_ELEMENTS),
    )
    assert_array_equal(
        metadata.block_element_indices,
        np.tile(np.arange(N_ELEMENTS, dtype=np.int32), 2),
    )
    assert metadata.element_indices.shape == (2, N_ELEMENTS)
    assert metadata.node_indices.shape == (2, N_ELEMENTS + 1)


def test_install_capsule_contact_state_on_vertical_block() -> None:
    block = _build_block(
        vertical=True,
        starts=(np.zeros(3), np.array([0.03, 0.0, 0.0])),
    )
    metadata = build_block_capsule_metadata(block, n_elements_per_rod=N_ELEMENTS)
    install_capsule_contact_state(
        block,
        metadata,
        device=CPU_DEVICE,
        dtype=block.device_dtype,
    )
    state = block.jax_get_state()
    for key in CONTACT_STATE_KEYS:
        assert key in state, f"missing contact state key {key}"


def test_capsule_kinematics_from_vertical_block_state() -> None:
    block = _build_block(
        vertical=True,
        starts=(np.zeros(3), np.array([0.03, 0.0, 0.0])),
    )
    metadata = build_block_capsule_metadata(block, n_elements_per_rod=N_ELEMENTS)
    state = block.jax_compute_internal_forces_and_torques(
        block.jax_get_state(),
        np.float64(0.0),
    )
    kinematics = capsule_kinematics_from_block_state(state, metadata)

    assert kinematics["centers"].shape == (2 * N_ELEMENTS, 3)
    assert kinematics["axes"].shape == (2 * N_ELEMENTS, 3)
    assert kinematics["lengths"].shape == (2 * N_ELEMENTS,)
    assert kinematics["radii"].shape == (2 * N_ELEMENTS,)
    assert kinematics["directors"].shape == (2 * N_ELEMENTS, 3, 3)
    assert kinematics["omega"].shape == (2 * N_ELEMENTS, 3)
    assert kinematics["block_element_indices"].shape == (2 * N_ELEMENTS,)


def test_vertical_capsule_contact_forces_match_horizontal() -> None:
    starts = (np.zeros(3), np.array([0.03, 0.0, 0.0]))

    def _contact_external_forces(*, vertical: bool) -> np.ndarray:
        with jax.default_device(CPU_DEVICE):
            simulator = eaj.Simulator()
            inner_block_cls = (
                eaj._CosseratRodVerticalMemoryBlock
                if vertical
                else eaj._CosseratRodMemoryBlock
            )
            rod_block = eaj.configure_rod_block(
                device=CPU_DEVICE,
                inner_block_cls=inner_block_cls,
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
                device=CPU_DEVICE,
                dtype=rod_block.device_dtype,
            )
            state = rod_block.jax_compute_internal_forces_and_torques(
                rod_block.jax_get_state(),
                np.float64(0.0),
            )
            state = rod_block.jax_zero_external_loads(state, np.float64(0.0))
            updated = simulator.jax_synchronize((state,), np.float64(0.0))[0]
            forces = np.asarray(updated["external_forces"])
            if vertical:
                return forces
            per_rod = []
            for rod_idx in range(rod_block.n_rods):
                node_start = int(rod_block.start_idx_in_rod_nodes[rod_idx])
                node_end = int(rod_block.end_idx_in_rod_nodes[rod_idx])
                per_rod.append(forces[:, node_start:node_end])
            return np.stack(per_rod, axis=0)

    vertical_forces = _contact_external_forces(vertical=True)
    horizontal_forces = _contact_external_forces(vertical=False)
    assert vertical_forces.shape == (2, 3, N_ELEMENTS + 1)
    assert horizontal_forces.shape == (2, 3, N_ELEMENTS + 1)
    assert float(np.linalg.norm(vertical_forces)) > 0.0
    assert_allclose(vertical_forces, horizontal_forces, rtol=1.0e-10, atol=1.0e-10)


def test_vertical_block_registers_capsule_contact_op() -> None:
    with jax.default_device(CPU_DEVICE):
        simulator = eaj.Simulator()
        rod_block = eaj.configure_rod_block(
            device=CPU_DEVICE,
            inner_block_cls=eaj._CosseratRodVerticalMemoryBlock,
        )
        simulator.enable_block_supports(ea.CosseratRod, rod_block)
        simulator.append(_overlapping_rod(start=np.zeros(3)))
        simulator.append(_overlapping_rod(start=np.array([0.03, 0.0, 0.0])))
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
            device=CPU_DEVICE,
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
