"""Benchmark multi-snake MPI PyElastica (Numba) rollout throughput."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_NODE_DIR = SCRIPT_DIR.parent / "snake-self-activate-single-node"
if str(SINGLE_NODE_DIR) not in sys.path:
    sys.path.insert(0, str(SINGLE_NODE_DIR))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import click  # noqa: E402
from mpi4py import MPI  # noqa: E402

from _jax_snake_common import (  # noqa: E402
    mpi_global_snake_count,
    mpi_snakes_per_rank,
    run_pyelastica_rollout_mpi,
)


def run(
    *,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    comm: MPI.Intracomm | None = None,
) -> list[float] | None:
    """
    Run one weak-scaling PyElastica MPI rollout and return per-rank times.

    Each rank owns ``multiplier * 2 ** snakes_per_rank_exp`` snakes and runs
    the same Numba PositionVerlet setup as the single-node PyElastica benchmark.

    Parameters
    ----------
    snakes_per_rank_exp
        Exponent for the base per-rank snake count ``2 ** exp``.
    snakes_per_rank_multiplier
        Scalar applied to the base per-rank snake count.
    steps
        Number of timed integration steps.
    warmup_runs
        Number of warmup steps before timing.
    comm
        MPI communicator. Defaults to ``MPI.COMM_WORLD``.

    Returns
    -------
    list[float] | None
        Gathered per-rank rollout walltimes in seconds on rank 0. Non-root
        ranks receive ``None``.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    snakes_per_rank = mpi_snakes_per_rank(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
    )
    _, rollout_walltimes = run_pyelastica_rollout_mpi(
        comm=comm,
        snakes_per_rank=snakes_per_rank,
        steps=steps,
        warmup_runs=warmup_runs,
    )
    return rollout_walltimes


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--snakes-per-rank-exp",
    type=int,
    default=4,
    show_default=True,
    help="Exponent for the base per-rank snake count (2 ** exp).",
)
@click.option(
    "--snakes-per-rank-multiplier",
    type=int,
    default=1,
    show_default=True,
    help="Scale the base per-rank snake count before weak scaling.",
)
@click.option("--steps", type=int, default=1000, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
def main(
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
) -> None:
    comm = MPI.COMM_WORLD
    snakes_per_rank = mpi_snakes_per_rank(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
    )
    n_snakes_total = mpi_global_snake_count(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        comm_size=comm.Get_size(),
    )
    rollout_walltimes = run(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        steps=steps,
        warmup_runs=warmup_runs,
        comm=comm,
    )
    if comm.Get_rank() == 0:
        assert rollout_walltimes is not None, "Rank 0 must receive gathered timings."
        rollout_walltimes_csv = ",".join(
            f"{rollout_walltime:.18e}" for rollout_walltime in rollout_walltimes
        )
        print("backend=pyelastica")
        print(f"mpi_size={comm.Get_size()}")
        print(f"snakes_per_rank_multiplier={snakes_per_rank_multiplier}")
        print(f"snakes_per_rank={snakes_per_rank}")
        print(f"n_snakes={n_snakes_total}")
        print(f"rollout_walltimes={rollout_walltimes_csv}")


if __name__ == "__main__":
    main()
