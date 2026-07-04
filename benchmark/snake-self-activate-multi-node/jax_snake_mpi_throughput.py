"""Benchmark multi-snake MPI rollout throughput on one MPI world size."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_NODE_DIR = SCRIPT_DIR.parent / "snake-self-activate-single-node"
if str(SINGLE_NODE_DIR) not in sys.path:
    sys.path.insert(0, str(SINGLE_NODE_DIR))

import click
from mpi4py import MPI

from _jax_snake_common import run_jax_rollout_mpi


def global_snake_count(*, snakes_per_rank_exp: int, comm_size: int) -> int:
    """Return the weak-scaling global snake count for an MPI world size."""
    snakes_per_rank = 2**snakes_per_rank_exp
    return snakes_per_rank * comm_size


def run(
    *,
    snakes_per_rank_exp: int,
    steps: int,
    warmup_runs: int,
    comm: MPI.Intracomm | None = None,
) -> float:
    """
    Run one weak-scaling MPI rollout and return the max rank rollout time.

    Each rank owns ``2 ** snakes_per_rank_exp`` snakes. The global snake count
    grows linearly with ``comm_size``.

    Parameters
    ----------
    snakes_per_rank_exp
        Exponent for the per-rank snake count ``snakes_per_rank = 2 ** exp``.
    steps
        Number of timed integration steps.
    warmup_runs
        Number of warmup integration chunks before timing.
    comm
        MPI communicator. Defaults to ``MPI.COMM_WORLD``.

    Returns
    -------
    float
        Maximum rollout walltime across MPI ranks in seconds.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    n_snakes_total = global_snake_count(
        snakes_per_rank_exp=snakes_per_rank_exp,
        comm_size=comm.Get_size(),
    )
    _, rollout_walltime = run_jax_rollout_mpi(
        comm=comm,
        n_snakes_total=n_snakes_total,
        steps=steps,
        warmup_runs=warmup_runs,
    )
    return rollout_walltime


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--snakes-per-rank-exp",
    type=int,
    default=4,
    show_default=True,
    help="Exponent for snakes owned by each MPI rank (2 ** exp).",
)
@click.option("--steps", type=int, default=1000, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
def main(
    snakes_per_rank_exp: int,
    steps: int,
    warmup_runs: int,
) -> None:
    comm = MPI.COMM_WORLD
    snakes_per_rank = 2**snakes_per_rank_exp
    n_snakes_total = global_snake_count(
        snakes_per_rank_exp=snakes_per_rank_exp,
        comm_size=comm.Get_size(),
    )
    rollout_walltime = run(
        snakes_per_rank_exp=snakes_per_rank_exp,
        steps=steps,
        warmup_runs=warmup_runs,
        comm=comm,
    )
    if comm.Get_rank() == 0:
        print(f"mpi_size={comm.Get_size()}")
        print(f"snakes_per_rank={snakes_per_rank}")
        print(f"n_snakes={n_snakes_total}")
        print(f"rollout_walltime={rollout_walltime:.6f}")


if __name__ == "__main__":
    main()
