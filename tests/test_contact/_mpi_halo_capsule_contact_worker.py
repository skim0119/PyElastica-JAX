"""MPI worker: cross-rank CapsuleContact force parity vs single-rank reference.

Environment:
    ELASTICA_MPI_VERTICAL=0|1 — use vertical inner block when 1.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import numpy as np
from mpi4py import MPI
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

import elastica as ea
import elastica_jax as eaj
from elastica_jax.contact.capsule_metadata import (
    CONTACT_STATE_KEYS,
    build_block_capsule_metadata,
    install_capsule_contact_state,
)

N_ELEMENTS = 4
CONTACT_STIFFNESS = 1.0e4
CONTACT_DAMPING = 1.0e-2
TIME_STEP = 1.0e-4
STARTS = (
    np.zeros(3, dtype=np.float64),
    np.array([0.03, 0.0, 0.0], dtype=np.float64),
)


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


def _single_rank_forces(*, vertical: bool) -> np.ndarray:
    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        simulator = eaj.Simulator()
        inner = (
            eaj._CosseratRodVerticalMemoryBlock
            if vertical
            else eaj._CosseratRodMemoryBlock
        )
        rod_block = eaj.configure_rod_block(device=device, inner_block_cls=inner)
        simulator.enable_block_supports(ea.CosseratRod, rod_block)
        for start in STARTS:
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
            device=device,
            dtype=rod_block.device_dtype,
        )
        state = rod_block.jax_compute_internal_forces_and_torques(
            rod_block.jax_get_state(),
            np.float64(0.0),
        )
        state = rod_block.jax_zero_external_loads(state, np.float64(0.0))
        updated = simulator.jax_synchronize((state,), np.float64(0.0))[0]
        return np.asarray(updated["external_forces"])


def _mpi_local_forces(*, vertical: bool, comm: MPI.Intracomm) -> np.ndarray:
    device = jax.devices("cpu")[0]
    with jax.default_device(device):
        simulator = eaj.Simulator()
        inner = (
            eaj._CosseratRodVerticalMemoryBlock
            if vertical
            else eaj._CosseratRodMemoryBlock
        )
        rod_block = eaj.configure_rod_block_mpi(
            comm=comm,
            device=device,
            inner_block_cls=inner,
        )
        simulator.enable_block_supports(ea.CosseratRod, rod_block)
        for rod_index, start in enumerate(STARTS):
            if rod_block.owns_rod(rod_index):
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
            device=device,
            dtype=rod_block.device_dtype,
        )
        for key in CONTACT_STATE_KEYS:
            assert key in rod_block.jax_get_state()
        # jax_synchronize must run under JIT for mpi4jax collectives.
        state = rod_block.jax_compute_internal_forces_and_torques(
            rod_block.jax_get_state(),
            np.float64(0.0),
        )
        state = rod_block.jax_zero_external_loads(state, np.float64(0.0))

        @jax.jit
        def _synchronize(states, time):
            return simulator.jax_synchronize(states, time)

        updated = _synchronize((state,), np.float64(0.0))[0]
        return np.asarray(updated["external_forces"])


def main() -> None:
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "This worker expects exactly two MPI ranks."
    vertical = os.environ.get("ELASTICA_MPI_VERTICAL", "0") == "1"
    reference = _single_rank_forces(vertical=vertical)
    local = _mpi_local_forces(vertical=vertical, comm=comm)

    # Round-robin: rank 0 owns global rod 0, rank 1 owns global rod 1.
    if vertical:
        ref_local = reference[comm.Get_rank()]
        local_cmp = np.asarray(local[0])
    else:
        # Packed layout: extract the owned rod's node/element slice from reference.
        # Single-rank packed block concatenates rods with ghost separators; use
        # rod starts from a throwaway single-rank block of the same layout.
        device = jax.devices("cpu")[0]
        with jax.default_device(device):
            probe = eaj.Simulator()
            block = eaj.configure_rod_block(device=device)
            probe.enable_block_supports(ea.CosseratRod, block)
            for start in STARTS:
                probe.append(_overlapping_rod(start=start))
            probe.finalize()
            start_nodes = int(block.start_idx_in_rod_nodes[comm.Get_rank()])
            end_nodes = int(block.end_idx_in_rod_nodes[comm.Get_rank()])
            ref_local = reference[:, start_nodes:end_nodes]
            local_cmp = local
            local_nodes = local.shape[-1]
            assert end_nodes - start_nodes == local_nodes, (
                f"packed slice width mismatch: ref {end_nodes - start_nodes} "
                f"vs local {local_nodes}"
            )

    assert_allclose(local_cmp, ref_local, rtol=1.0e-6, atol=1.0e-8)
    # Non-trivial contact: owned rod must feel a force.
    assert np.any(np.abs(local_cmp) > 1.0e-8), "expected nonzero contact force"
    if comm.Get_rank() == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    main()
