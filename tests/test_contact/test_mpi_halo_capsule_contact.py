"""Tests for MPI halo CapsuleContact (HALO_READ).

Public seams:
- Owned-capsule mask zeros ghost loads before scatter
- Cross-rank contact forces match a single-rank reference (packed + vertical)
- CapsuleContactOp declares HALO_READ; gravity stays LOCAL
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

import elastica_jax as eaj  # noqa: E402
from elastica_jax.block_operation import CommunicationScope  # noqa: E402
from elastica_jax.contact.kernels import apply_capsule_pair_forces  # noqa: E402
from elastica_jax.operations import GravityAnalyticalDamperJax  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MPI_HELPER = Path(__file__).resolve().parent / "_mpi_halo_capsule_contact_worker.py"


def test_capsule_contact_declares_halo_read_scope() -> None:
    assert eaj.CapsuleContactOp.communication_scope is CommunicationScope.HALO_READ
    assert eaj.WallContactOp.communication_scope is CommunicationScope.LOCAL
    # Gravity is a per-rod NoOpsJax operator: rank-local, no halo exchange.
    assert issubclass(GravityAnalyticalDamperJax, eaj.NoOpsJax)


def test_apply_capsule_pair_forces_owned_mask_skips_ghost_scatter() -> None:
    """Ghost capsules may participate in pairs but must not write local loads."""
    centers = jnp.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    velocities = jnp.zeros_like(centers)
    axes = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), centers.shape)
    lengths = jnp.full((3,), 0.2, dtype=jnp.float64)
    radii = jnp.full((3,), 0.02, dtype=jnp.float64)
    omega = jnp.zeros_like(centers)
    directors = jnp.broadcast_to(jnp.eye(3), (3, 3, 3))
    # Capsules map to disjoint element slots so owned writes cannot alias ghosts.
    block_element_indices = jnp.asarray([0, 2, 4], dtype=jnp.int32)
    external_forces = jnp.zeros((3, 6), dtype=jnp.float64)
    external_torques = jnp.zeros((3, 5), dtype=jnp.float64)

    pair_first = jnp.asarray([0], dtype=jnp.int32)
    pair_second = jnp.asarray([1], dtype=jnp.int32)
    pair_active = jnp.asarray([True])
    # Capsule 1 is a ghost: contact still resolved, but only capsule 0 writes.
    owned_mask = jnp.asarray([True, False, False])

    forces, torques, _, _ = apply_capsule_pair_forces(
        pair_first=pair_first,
        pair_second=pair_second,
        pair_active=pair_active,
        centers=centers,
        velocities=velocities,
        axes=axes,
        lengths=lengths,
        radii=radii,
        omega=omega,
        directors=directors,
        block_element_indices=block_element_indices,
        external_forces=external_forces,
        external_torques=external_torques,
        contact_stiffness=1.0e4,
        contact_damping=0.0,
        cached_candidates=jnp.asarray([True]),
        last_detection_time=jnp.asarray(-np.inf),
        time=0.0,
        steps_between_detection=0,
        time_step=1.0e-4,
        owned_mask=owned_mask,
    )
    # Owned capsule 0 writes elements 0-1; ghost capsule 1 must not write 2-3.
    assert jnp.any(jnp.abs(forces[:, 0]) > 0.0)
    assert_allclose(np.asarray(forces[:, 2:4]), 0.0, atol=1.0e-12)
    assert_allclose(np.asarray(torques[:, 2:3]), 0.0, atol=1.0e-12)


@pytest.mark.parametrize("vertical", [False, True])
def test_mpi_halo_contact_forces_match_single_rank(vertical: bool) -> None:
    """Two ranks with one overlapping rod each match a single-rank two-rod block."""
    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "1")
    env["ELASTICA_MPI_VERTICAL"] = "1" if vertical else "0"
    result = subprocess.run(
        [
            "mpirun",
            "-n",
            "2",
            sys.executable,
            str(MPI_HELPER),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"MPI halo contact worker failed (vertical={vertical}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PASS" in result.stdout
