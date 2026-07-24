"""Unit tests for rod-contact MPI sweep helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "rod-contact-scaling"
# Append so ``_rod_contact_common`` does not pop this entry from ``sys.path[0]``.
if str(BENCH_DIR) not in sys.path:
    sys.path.append(str(BENCH_DIR))

from sweep_jax_rod_contact_mpi_throughput import (  # noqa: E402
    build_mpiexec_command,
    parse_rollout_walltimes,
)
from _rod_contact_common import (  # noqa: E402
    mpi_global_rod_count,
    mpi_rods_per_rank,
)


def test_mpi_rods_per_rank_scales_multiplier() -> None:
    assert mpi_rods_per_rank(rods_per_rank_exp=3, rods_per_rank_multiplier=2) == 16


def test_mpi_global_rod_count_is_weak_scaling() -> None:
    assert (
        mpi_global_rod_count(
            rods_per_rank_exp=2,
            rods_per_rank_multiplier=1,
            comm_size=4,
        )
        == 16
    )


def test_parse_rollout_walltimes_reads_csv_line() -> None:
    stdout = "\n".join(
        [
            "backend=cuda",
            "vertical=0",
            "mpi_size=2",
            "rollout_walltimes=1.250000000000000000e-01,2.500000000000000000e-01",
        ]
    )
    walltimes = parse_rollout_walltimes(stdout)
    assert walltimes.shape == (2,)
    np.testing.assert_allclose(walltimes, [0.125, 0.25])


def test_parse_rollout_walltimes_rejects_missing_marker() -> None:
    with pytest.raises(AssertionError, match="Could not parse rollout walltimes"):
        parse_rollout_walltimes("mpi_size=1\n")


def test_build_mpiexec_command_includes_vertical_flag() -> None:
    command = build_mpiexec_command(
        mpi_size=4,
        python_executable="/usr/bin/python",
        rods_per_rank_exp=3,
        rods_per_rank_multiplier=1,
        steps=200,
        warmup_runs=1,
        backend="cuda",
        vertical=True,
        n_elements=10,
        steps_between_detection=0,
        broad_phase="spatial_hash",
    )
    assert command[:3] == ["ibrun", "-n", "4"]
    assert "--vertical" in command
    assert mpi_rods_per_rank(rods_per_rank_exp=3, rods_per_rank_multiplier=1) == 8
    assert (
        mpi_global_rod_count(
            rods_per_rank_exp=3,
            rods_per_rank_multiplier=1,
            comm_size=4,
        )
        == 32
    )
