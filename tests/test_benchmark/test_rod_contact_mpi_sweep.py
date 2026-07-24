"""Unit tests for rod-contact MPI sweep helpers."""

from __future__ import annotations

import csv
import os
import subprocess
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
    export_scaling_csv,
    flatten_rollout_samples,
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


def test_export_scaling_csv_is_long_form_with_vertical(tmp_path: Path) -> None:
    points = [
        (1, 8, 8, np.asarray([0.10], dtype=np.float64)),
        (2, 8, 16, np.asarray([0.11, 0.12], dtype=np.float64)),
    ]
    output = tmp_path / "mpi_scaling.csv"
    export_scaling_csv(
        points,
        backend="cuda",
        rods_per_rank_exp=3,
        rods_per_rank_multiplier=1,
        steps=200,
        vertical=True,
        n_elements=10,
        steps_between_detection=0,
        broad_phase="spatial_hash",
        output=output,
    )
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mpi_size"] for row in rows] == ["1", "2", "2"]
    assert [row["rank"] for row in rows] == ["0", "0", "1"]
    assert rows[0]["vertical"] == "1"
    assert rows[0]["backend"] == "cuda"
    assert rows[2]["n_rods"] == "16"
    assert rows[2]["rollout_walltime_s"] == "0.12"


def test_flatten_rollout_samples_expands_ranks() -> None:
    points = [(2, 4, 8, np.asarray([0.2, 0.3], dtype=np.float64))]
    assert flatten_rollout_samples(points) == [
        (2, 4, 8, 0, 0.2),
        (2, 4, 8, 1, 0.3),
    ]


@pytest.mark.parametrize("vertical", [False, True])
def test_mpi_worker_smoke_prints_rollout_walltimes(vertical: bool) -> None:
    """Tiny two-rank worker smoke (halo CapsuleContact path)."""
    worker = BENCH_DIR / "jax_rod_contact_mpi_throughput.py"
    command = [
        "mpirun",
        "-n",
        "2",
        sys.executable,
        str(worker),
        "--backend",
        "cpu",
        "--rods-per-rank-exp",
        "1",
        "--steps",
        "2",
        "--warmup-runs",
        "0",
        "--n-elements",
        "4",
        "--broad-phase",
        "all_pairs",
    ]
    if vertical:
        command.append("--vertical")
    result = subprocess.run(
        command,
        cwd=BENCH_DIR.parent.parent,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "JAX_ENABLE_X64": "1",
        },
    )
    assert result.returncode == 0, (
        f"MPI worker smoke failed (vertical={vertical}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    walltimes = parse_rollout_walltimes(result.stdout)
    assert walltimes.shape == (2,)
    assert np.all(walltimes > 0.0)
