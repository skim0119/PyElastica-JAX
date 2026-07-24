"""Sweep single-node two-GPU rod-contact throughput (horizontal and vertical)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import click
import jax
import numpy as np
from tqdm import tqdm

from _rod_contact_common import N_ELEMENTS
from _rod_contact_scaling_sweep import (
    ScalingCase,
    export_scaling_csv,
    export_scaling_plot,
    run_scaling_benchmark,
    series_label,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
JAX_MPI_WORKER = SCRIPT_DIR / "jax_rod_contact_mpi_throughput.py"
N_DEVICES = 2

type SweepPoint = tuple[int, int, float, float]


def _backend_available(name: str, *, n_devices: int = N_DEVICES) -> bool:
    try:
        return len(jax.devices(name)) >= n_devices
    except RuntimeError:
        return False


def _parse_rollout_walltimes(output: str) -> np.ndarray:
    match = re.search(r"rollout_walltimes=([0-9eE+.,-]+)", output)
    assert match is not None, f"Could not parse rollout walltimes from:\n{output}"
    values = tuple(
        float(item) for item in match.group(1).split(",") if item.strip() != ""
    )
    assert values, f"Parsed empty rollout walltime list from:\n{output}"
    return np.asarray(values, dtype=np.float64)


def _mpi_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    return env


def _build_horizontal_mpiexec_command(
    *,
    n_rods_exp: int,
    python_executable: str,
    steps: int,
    warmup_runs: int,
    backend: str,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
) -> list[str]:
    """Build ``ibrun -n 2`` for one horizontal two-GPU sample.

    Global ``n_rods = 2 ** n_rods_exp`` is split evenly across two ranks, so
    each rank owns ``2 ** (n_rods_exp - 1)`` rods.
    """
    assert n_rods_exp >= 1, "horizontal gpu2x requires n_rods_exp >= 1."
    return [
        "ibrun",
        "-n",
        str(N_DEVICES),
        python_executable,
        str(JAX_MPI_WORKER),
        "--backend",
        backend,
        "--rods-per-rank-exp",
        str(n_rods_exp - 1),
        "--rods-per-rank-multiplier",
        "1",
        "--steps",
        str(steps),
        "--warmup-runs",
        str(warmup_runs),
        "--n-elements",
        str(n_elements),
        "--steps-between-detection",
        str(steps_between_detection),
        "--broad-phase",
        broad_phase,
    ]


def sweep_horizontal_mpi(
    backend: str,
    min_exp: int,
    max_exp: int,
    *,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    python_executable: str,
    verbose: bool,
) -> list[SweepPoint]:
    """Sweep rod count on two MPI ranks with halo CapsuleContact (horizontal)."""
    assert min_exp >= 1, "horizontal gpu2x requires min_exp >= 1 (n_rods >= 2)."
    assert max_exp >= min_exp, "max_exp must be >= min_exp."
    label = series_label(backend, vertical=False, n_devices=N_DEVICES)
    results: list[SweepPoint] = []
    for exponent in tqdm(
        range(min_exp, max_exp + 1),
        desc=f"{label} rod-rod contact",
        disable=not verbose,
    ):
        command = _build_horizontal_mpiexec_command(
            n_rods_exp=exponent,
            python_executable=python_executable,
            steps=steps,
            warmup_runs=warmup_runs,
            backend=backend,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase=broad_phase,
        )
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=_mpi_worker_env(),
        )
        walltimes = _parse_rollout_walltimes(completed.stdout)
        rollout_seconds = float(np.max(walltimes))
        n_rods = 2**exponent
        print(
            f"{label} n_rods={n_rods} (2^{exponent}): "
            f"rollout={rollout_seconds:.6f}s (max over {walltimes.size} ranks)"
        )
        # Instantiation is not separately reported by the MPI worker; store 0.
        results.append((exponent, n_rods, 0.0, rollout_seconds))
    return results


def _export_horizontal_case(
    points: list[SweepPoint],
    *,
    backend: str,
    steps: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    output_plot: Path,
    output_csv: Path | None,
) -> None:
    case = ScalingCase(
        backend=backend,
        vertical=False,
        n_devices=N_DEVICES,
        points=points,
    )
    csv_path = output_csv if output_csv is not None else output_plot.with_suffix(".csv")
    export_scaling_csv(
        [case],
        steps=steps,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
        output=csv_path,
    )
    export_scaling_plot(
        [case],
        steps=steps,
        n_elements=n_elements,
        output=output_plot,
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cuda",
    show_default=True,
    help="JAX backend; requires at least two devices.",
)
@click.option(
    "--layout",
    type=click.Choice(["vertical", "horizontal", "both"], case_sensitive=False),
    default="both",
    show_default=True,
    help=(
        "vertical: shard_map stacked block; "
        "horizontal: MPI 2 ranks + halo CapsuleContact; "
        "both: run each and write separate CSV/PNG."
    ),
)
@click.option("--min-exp", type=int, default=1, show_default=True)
@click.option("--max-exp", type=int, default=8, show_default=True)
@click.option("--steps", type=int, default=200, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
@click.option("--n-elements", type=int, default=N_ELEMENTS, show_default=True)
@click.option(
    "--steps-between-detection",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--broad-phase",
    type=click.Choice(["spatial_hash", "all_pairs"]),
    default="spatial_hash",
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output/rod_contact_gpu2x.png"),
    show_default=True,
    help="Base output path; layout suffix is added when --layout both.",
)
@click.option(
    "--csv-output",
    type=click.Path(path_type=Path),
    default=None,
    help="CSV path (default: plot path with .csv suffix).",
)
@click.option(
    "--python",
    "python_executable",
    type=str,
    default=sys.executable,
    show_default=True,
    help="Python executable passed to ibrun for the horizontal path.",
)
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    backend: str,
    layout: str,
    min_exp: int,
    max_exp: int,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    output: Path,
    csv_output: Path | None,
    python_executable: str,
    quiet: bool,
) -> None:
    """Sweep rod count on one node with two GPUs (horizontal and/or vertical)."""
    backend = backend.lower()
    layout = layout.lower()
    assert _backend_available(backend), (
        f"Backend {backend!r} needs at least {N_DEVICES} devices for gpu2x."
    )
    assert min_exp >= 1, "gpu2x sweeps require min_exp >= 1 (n_rods >= 2)."

    verbose = not quiet
    run_vertical = layout in {"vertical", "both"}
    run_horizontal = layout in {"horizontal", "both"}

    if run_vertical:
        vertical_plot = (
            output
            if layout == "vertical"
            else output.with_name(f"{output.stem}_vertical{output.suffix}")
        )
        vertical_csv = None
        if csv_output is not None and layout == "vertical":
            vertical_csv = csv_output
        elif csv_output is not None:
            vertical_csv = csv_output.with_name(
                f"{csv_output.stem}_vertical{csv_output.suffix}"
            )
        run_scaling_benchmark(
            backend=backend,
            min_exp=min_exp,
            max_exp=max_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase=broad_phase,
            vertical=True,
            n_devices=N_DEVICES,
            output_plot=vertical_plot,
            output_csv=vertical_csv,
            verbose=verbose,
        )

    if run_horizontal:
        horizontal_plot = (
            output
            if layout == "horizontal"
            else output.with_name(f"{output.stem}_horizontal{output.suffix}")
        )
        horizontal_csv = None
        if csv_output is not None and layout == "horizontal":
            horizontal_csv = csv_output
        elif csv_output is not None:
            horizontal_csv = csv_output.with_name(
                f"{csv_output.stem}_horizontal{csv_output.suffix}"
            )
        points = sweep_horizontal_mpi(
            backend,
            min_exp,
            max_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase=broad_phase,
            python_executable=python_executable,
            verbose=verbose,
        )
        _export_horizontal_case(
            points,
            backend=backend,
            steps=steps,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase=broad_phase,
            output_plot=horizontal_plot,
            output_csv=horizontal_csv,
        )


if __name__ == "__main__":
    main()
