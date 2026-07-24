"""Benchmark multi-node rod-rod contact rollout throughput on one MPI world."""

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


BACKEND = _read_cli_option("--backend", "cuda")
if BACKEND == "cpu":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import click
from mpi4py import MPI

comm = MPI.COMM_WORLD
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    f"/tmp/pyelastica_jax_rod_contact_mpi_cache_{BACKEND}_rank{comm.Get_rank()}",
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from _rod_contact_common import (  # noqa: E402
    N_ELEMENTS,
    STEPS_BETWEEN_DETECTION,
    mpi_global_rod_count,
    mpi_rods_per_rank,
    run_rollout_mpi,
)


def run(
    *,
    rods_per_rank_exp: int,
    rods_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    backend: str,
    vertical: bool = False,
    n_elements: int = N_ELEMENTS,
    steps_between_detection: int = STEPS_BETWEEN_DETECTION,
    broad_phase: str = "spatial_hash",
    comm: MPI.Intracomm | None = None,
) -> list[float] | None:
    """Run one weak-scaling MPI contact rollout; gather walltimes on rank 0.

    Parameters
    ----------
    rods_per_rank_exp
        Exponent for the base per-rank rod count ``2 ** exp``.
    rods_per_rank_multiplier
        Scalar applied to the base per-rank rod count.
    steps
        Number of timed integration steps.
    warmup_runs
        Number of warmup integration chunks before timing.
    backend
        JAX backend for the MPI-local rod block, such as ``"cpu"`` or ``"cuda"``.
    vertical
        If True, pack rods with the vertical memory block.
    n_elements
        Elements per rod.
    steps_between_detection
        Broad-phase refresh interval (0 = every step).
    broad_phase
        Capsule contact broad-phase strategy.
    comm
        MPI communicator. Defaults to ``MPI.COMM_WORLD``.

    Returns
    -------
    list[float] | None
        Gathered per-rank rollout walltimes on rank 0; ``None`` elsewhere.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    n_rods_total = mpi_global_rod_count(
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
        comm_size=comm.Get_size(),
    )
    _, rollout_walltimes = run_rollout_mpi(
        comm=comm,
        n_rods_total=n_rods_total,
        steps=steps,
        warmup_runs=warmup_runs,
        backend=backend,
        vertical=vertical,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
    )
    return rollout_walltimes


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cuda",
    show_default=True,
    help="JAX backend for the MPI-local rod block.",
)
@click.option(
    "--rods-per-rank-exp",
    type=int,
    default=4,
    show_default=True,
    help="Exponent for the base per-rank rod count (2 ** exp).",
)
@click.option(
    "--rods-per-rank-multiplier",
    type=int,
    default=1,
    show_default=True,
    help="Scale the base per-rank rod count before weak scaling.",
)
@click.option("--steps", type=int, default=200, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
@click.option("--n-elements", type=int, default=N_ELEMENTS, show_default=True)
@click.option(
    "--steps-between-detection",
    type=int,
    default=STEPS_BETWEEN_DETECTION,
    show_default=True,
)
@click.option(
    "--broad-phase",
    type=click.Choice(["spatial_hash", "all_pairs"]),
    default="spatial_hash",
    show_default=True,
)
@click.option(
    "--vertical",
    is_flag=True,
    help="Use vertical (stacked-axis) rod memory block packing.",
)
def main(
    backend: str,
    rods_per_rank_exp: int,
    rods_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    vertical: bool,
) -> None:
    """Print parseable weak-scaling timings for one MPI world size."""
    rods_per_rank = mpi_rods_per_rank(
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
    )
    n_rods_total = mpi_global_rod_count(
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
        comm_size=comm.Get_size(),
    )
    rollout_walltimes = run(
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
        steps=steps,
        warmup_runs=warmup_runs,
        backend=backend,
        vertical=vertical,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
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
        print(f"rods_per_rank_multiplier={rods_per_rank_multiplier}")
        print(f"rods_per_rank={rods_per_rank}")
        print(f"n_rods={n_rods_total}")
        print(f"n_elements={n_elements}")
        print(f"steps_between_detection={steps_between_detection}")
        print(f"broad_phase={broad_phase}")
        print(f"rollout_walltimes={rollout_walltimes_csv}")


if __name__ == "__main__":
    main()
