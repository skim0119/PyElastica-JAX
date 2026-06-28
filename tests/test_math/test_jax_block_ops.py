from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica_jax as eaj

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "snake-self-activate-single-node"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from snake_operation import (  # noqa: E402
    GravityPlaneContactBlockJax,
    SnakeMuscleTorquesBlockJax,
)
from examples.ContinuumSnakeGPUCase.run_continuum_snake_gpu import (
    SnakeMuscleTorquesJax,
    SnakePlaneContactJax,
    build_rod,
    default_b_coeff,
)
from elastica_jax.memory_block.memory_block_rod_jax import (  # noqa: E402
    _CosseratRodMemoryBlock,
    JAXRodView,
    JAXRodViewMetadata,
)


def _single_rod_metadata(block: _CosseratRodMemoryBlock) -> JAXRodViewMetadata:
    return JAXRodViewMetadata(
        0,
        slice(
            int(block.start_idx_in_rod_nodes[0]),
            int(block.end_idx_in_rod_nodes[0]),
        ),
        slice(
            int(block.start_idx_in_rod_elems[0]),
            int(block.end_idx_in_rod_elems[0]),
        ),
        slice(
            int(block.start_idx_in_rod_voronoi[0]),
            int(block.end_idx_in_rod_voronoi[0]),
        ),
    )


def test_block_snake_ops_match_single_rod_sequence() -> None:
    rod = build_rod(n_elem=10)
    with jax.default_device(jax.devices("cpu")[0]):
        block = eaj.configure_rod_block(device_dtype=np.float64)([rod], [0])
    base_state = block.jax_compute_internal_forces_and_torques(
        block.jax_get_state(),
        np.float64(0.0),
    )
    metadata = _single_rod_metadata(block)

    b_coeff = default_b_coeff()
    kinetic_mu = np.array([0.89226666, 1.3384, 1.7845333], dtype=np.float64)
    static_mu = np.zeros(3, dtype=np.float64)
    time = np.float64(0.3)

    rod_state = (
        SnakeMuscleTorquesJax(
            b_coeff=b_coeff,
            period=2.0,
            base_length=0.35,
            gravitational_acc=-9.80665,
            _system=rod,
        )
        .jax_operate_synchronize(JAXRodView(base_state, metadata), time)
        .commit()
    )
    rod_state = (
        SnakePlaneContactJax(
            plane_origin=np.array([0.0, -0.35 * 0.011, 0.0], dtype=np.float64),
            plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            slip_velocity_tol=1.0e-8,
            k=1.0,
            nu=1.0e-6,
            kinetic_mu_array=kinetic_mu,
            static_mu_array=static_mu,
            _system=rod,
        )
        .jax_operate_synchronize(JAXRodView(rod_state, metadata), time)
        .commit()
    )
    rod_state = (
        eaj.AnalyticalLinearDamperJax(
            time_step=np.float64(1.0e-4),
            damping_constant=2.0e-3,
            _system=rod,
        )
        .jax_operate_constrain_rates(JAXRodView(rod_state, metadata), time)
        .commit()
    )

    block_state = SnakeMuscleTorquesBlockJax(
        b_coeff=b_coeff,
        period=2.0,
        base_length=0.35,
        _system=block,
    ).jax_block_operate_synchronize(base_state, time)
    block_state = GravityPlaneContactBlockJax(
        plane_origin=np.array([0.0, -0.35 * 0.011, 0.0], dtype=np.float64),
        plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        slip_velocity_tol=1.0e-8,
        k=1.0,
        nu=1.0e-6,
        kinetic_mu_array=kinetic_mu,
        static_mu_array=static_mu,
        _system=block,
    ).jax_block_operate_synchronize(block_state, time)
    block_state = (
        eaj.AnalyticalLinearDamperJax(
            time_step=np.float64(1.0e-4),
            damping_constant=2.0e-3,
            _system=rod,
        )
        .jax_operate_constrain_rates(JAXRodView(block_state, metadata), time)
        .commit()
    )

    node_slice = metadata.node_slice
    elem_slice = metadata.element_slice

    np.testing.assert_allclose(
        np.asarray(block_state["external_forces"])[:, node_slice],
        np.asarray(rod_state["external_forces"]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(block_state["external_torques"])[:, elem_slice],
        np.asarray(rod_state["external_torques"]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(block_state["velocity_collection"])[:, node_slice],
        np.asarray(rod_state["velocity_collection"]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(block_state["omega_collection"])[:, elem_slice],
        np.asarray(rod_state["omega_collection"]),
        atol=1.0e-14,
    )
