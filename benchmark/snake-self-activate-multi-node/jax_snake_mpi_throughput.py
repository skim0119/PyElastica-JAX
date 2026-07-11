"""Benchmark multi-snake MPI rollout throughput on one MPI world size."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _read_cli_option(option: str, default: str) -> str:
    prefix = f"{option}="
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == option and index < len(sys.argv) - 1:
            return sys.argv[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return default


BACKEND = _read_cli_option("--backend", "cpu")
if BACKEND == "cpu":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_NODE_DIR = SCRIPT_DIR.parent / "snake-self-activate-single-node"
if str(SINGLE_NODE_DIR) not in sys.path:
    sys.path.insert(0, str(SINGLE_NODE_DIR))

import click
from mpi4py import MPI

comm = MPI.COMM_WORLD
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    f"/tmp/pyelastica_jax_mpi_cache_{BACKEND}_rank{comm.Get_rank()}",
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from _jax_snake_common import mpi_global_snake_count, mpi_snakes_per_rank, run_jax_rollout_mpi


def run(
    *,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    backend: str,
    vertical: bool = False,
    comm: MPI.Intracomm | None = None,
) -> list[float] | None:
    """
    Run one weak-scaling MPI rollout and return all per-rank rollout times.

    Each rank owns ``multiplier * 2 ** snakes_per_rank_exp`` snakes. The global
    snake count grows linearly with ``comm_size``.

    Parameters
    ----------
    snakes_per_rank_exp
        Exponent for the base per-rank snake count ``2 ** exp``.
    snakes_per_rank_multiplier
        Scalar applied to the base per-rank snake count.
    steps
        Number of timed integration steps.
    warmup_runs
        Number of warmup integration chunks before timing.
    backend
        JAX backend for the MPI-local rod block, such as ``"cpu"`` or ``"cuda"``.
    vertical
        If True, pack rods with the vertical memory block.
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
    n_snakes_total = mpi_global_snake_count(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        comm_size=comm.Get_size(),
    )
    _, rollout_walltimes = run_jax_rollout_mpi(
        comm=comm,
        n_snakes_total=n_snakes_total,
        steps=steps,
        warmup_runs=warmup_runs,
        backend=backend,
        vertical=vertical,
    )
    return rollout_walltimes


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="JAX backend for the MPI-local rod block.",
)
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
@click.option(
    "--vertical",
    is_flag=True,
    help="Use vertical (stacked-axis) rod memory block packing.",
)
def main(
    backend: str,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    vertical: bool,
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
        backend=backend,
        vertical=vertical,
        comm=comm,
    )
    if comm.Get_rank() == 0:
        assert rollout_walltimes is not None, "Rank 0 must receive gathered timings."
        rollout_walltimes_csv = ",".join(
            f"{rollout_walltime:.18e}" for rollout_walltime in rollout_walltimes
        )
        print(f"backend={backend}")
        print(f"vertical={int(vertical)}")
        print(f"mpi_size={comm.Get_size()}")
        print(f"snakes_per_rank_multiplier={snakes_per_rank_multiplier}")
        print(f"snakes_per_rank={snakes_per_rank}")
        print(f"n_snakes={n_snakes_total}")
        print(f"rollout_walltimes={rollout_walltimes_csv}")


if __name__ == "__main__":
    main()
