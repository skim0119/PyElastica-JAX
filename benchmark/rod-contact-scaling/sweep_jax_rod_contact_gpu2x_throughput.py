"""Sweep single-node two-GPU rod-contact throughput (horizontal and vertical)."""

from __future__ import annotations

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

from sweep_jax_rod_contact_mpi_throughput import run_mpi_point

N_DEVICES = 2

type SweepPoint = tuple[int, int, float, float]


def _backend_available(name: str, *, n_devices: int = N_DEVICES) -> bool:
    try:
        return len(jax.devices(name)) >= n_devices
    except RuntimeError:
        return False


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
        _, _, n_rods, walltimes = run_mpi_point(
            mpi_size=N_DEVICES,
            rods_per_rank_exp=exponent - 1,
            rods_per_rank_multiplier=1,
            vertical=False,
            python_executable=python_executable,
            steps=steps,
            warmup_runs=warmup_runs,
            backend=backend,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase=broad_phase,
        )
        rollout_seconds = float(np.max(walltimes))
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

    layouts = ("vertical", "horizontal") if layout == "both" else (layout,)
    for current_layout in layouts:
        plot_path = (
            output.with_name(f"{output.stem}_{current_layout}{output.suffix}")
            if layout == "both"
            else output
        )
        csv_path = csv_output
        if csv_output is not None and layout == "both":
            csv_path = csv_output.with_name(
                f"{csv_output.stem}_{current_layout}{csv_output.suffix}"
            )
        if current_layout == "vertical":
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
                output_plot=plot_path,
                output_csv=csv_path,
                verbose=not quiet,
            )
        else:
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
                verbose=not quiet,
            )
            _export_horizontal_case(
                points,
                backend=backend,
                steps=steps,
                n_elements=n_elements,
                steps_between_detection=steps_between_detection,
                broad_phase=broad_phase,
                output_plot=plot_path,
                output_csv=csv_path,
            )


if __name__ == "__main__":
    main()
