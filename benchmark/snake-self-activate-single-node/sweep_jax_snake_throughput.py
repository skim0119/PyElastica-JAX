"""Sweep multi-snake rollout throughput and export scaling CSV and plot."""

from __future__ import annotations

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import csv
from pathlib import Path
from typing import TypeAlias

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from jax_snake_throughput import run

SweepPoint: TypeAlias = tuple[int, int, float]


def _sweep_backend(
    backend: str,
    min_exp: int,
    max_exp: int,
    *,
    steps: int,
    warmup_runs: int,
    verbose: bool,
) -> list[SweepPoint]:
    """Run rollout timing for ``n_snakes = 2**exp`` across an exponent range."""
    assert min_exp >= 0, "min exponent must be nonnegative."
    assert max_exp >= min_exp, "max exponent must be greater than or equal to min."

    results: list[SweepPoint] = []
    for exponent in tqdm(
        range(min_exp, max_exp + 1),
        desc=backend,
        disable=not verbose,
    ):
        n_snakes = 2**exponent
        rollout_walltime = run(
            backend=backend,
            n_snakes_exp=exponent,
            steps=steps,
            warmup_runs=warmup_runs,
        )
        print(
            f"{backend} n_snakes={n_snakes} (2^{exponent}): "
            f"rollout_walltime={rollout_walltime:.6f}s"
        )
        results.append((exponent, n_snakes, rollout_walltime))
    return results


def _export_scaling_plot(
    series: dict[str, list[SweepPoint]],
    *,
    steps: int,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for label, points in series.items():
        n_snakes = np.asarray([point[1] for point in points], dtype=np.float64)
        walltimes = np.asarray([point[2] for point in points], dtype=np.float64)
        ax.loglog(n_snakes, walltimes, marker="o", label=label)

    ax.set_xlabel("number of snakes")
    ax.set_ylabel("rollout walltime (s)")
    ax.set_title(f"Multi-snake rollout scaling ({steps} steps) (20 elements/snake)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote plot: {output}")


def _export_scaling_csv(
    series: dict[str, list[SweepPoint]],
    *,
    steps: int,
    output: Path,
) -> None:
    """Write sweep results in long-form CSV for later replotting."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("backend", "exponent", "n_snakes", "rollout_walltime_s", "steps")
        )
        for backend, points in series.items():
            for exponent, n_snakes, walltime in points:
                writer.writerow((backend, exponent, n_snakes, walltime, steps))
    print(f"wrote csv: {output}")


def _load_scaling_csv(csv_path: Path) -> tuple[dict[str, list[SweepPoint]], int]:
    """Load sweep results written by :func:`_export_scaling_csv`."""
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"CSV {csv_path} is empty."

    steps = int(rows[0]["steps"])
    series: dict[str, list[SweepPoint]] = {}
    for row in rows:
        backend = row["backend"]
        exponent = int(row["exponent"])
        n_snakes = int(row["n_snakes"])
        walltime = float(row["rollout_walltime_s"])
        series.setdefault(backend, []).append((exponent, n_snakes, walltime))

    for points in series.values():
        points.sort(key=lambda point: point[0])
    return series, steps


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--steps", type=int, default=1000, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
@click.option("--cuda-min-exp", type=int, default=1, show_default=True)
@click.option("--cuda-max-exp", type=int, default=14, show_default=True)
@click.option("--cpu-min-exp", type=int, default=1, show_default=True)
@click.option("--cpu-max-exp", type=int, default=12, show_default=True)
@click.option("--gpu2x-min-exp", type=int, default=1, show_default=True)
@click.option("--gpu2x-max-exp", type=int, default=8, show_default=True)
@click.option("--gpu2x-sharded-min-exp", type=int, default=1, show_default=True)
@click.option("--gpu2x-sharded-max-exp", type=int, default=8, show_default=True)
@click.option("--pyelastica-min-exp", type=int, default=1, show_default=True)
@click.option("--pyelastica-max-exp", type=int, default=12, show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("snake_throughput_scaling.png"),
    show_default=True,
    help="Output path for the scaling plot.",
)
@click.option(
    "--csv-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for scaling results CSV (default: plot path with .csv suffix).",
)
@click.option(
    "--from-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Regenerate the scaling plot from a saved CSV instead of running sweeps.",
)
@click.option("--skip-cuda", is_flag=True, help="Skip the CUDA sweep.")
@click.option("--skip-cpu", is_flag=True, help="Skip the JAX CPU sweep.")
@click.option(
    "--skip-gpu2x", is_flag=True, help="Skip the 2-GPU dual-block CUDA sweep."
)
@click.option(
    "--skip-gpu2x-sharded",
    default=False,
    is_flag=True,
    help="Skip the 2-GPU sharded CUDA sweep.",
)
@click.option("--skip-pyelastica", is_flag=True, help="Skip the PyElastica sweep.")
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    steps: int,
    warmup_runs: int,
    cuda_min_exp: int,
    cuda_max_exp: int,
    cpu_min_exp: int,
    cpu_max_exp: int,
    gpu2x_min_exp: int,
    gpu2x_max_exp: int,
    gpu2x_sharded_min_exp: int,
    gpu2x_sharded_max_exp: int,
    pyelastica_min_exp: int,
    pyelastica_max_exp: int,
    output: Path,
    csv_output: Path | None,
    from_csv: Path | None,
    skip_cuda: bool,
    skip_cpu: bool,
    skip_gpu2x: bool,
    skip_gpu2x_sharded: bool,
    skip_pyelastica: bool,
    quiet: bool,
) -> None:
    assert steps > 0, "steps must be positive."

    if from_csv is not None:
        series, steps = _load_scaling_csv(from_csv)
        _export_scaling_plot(series, steps=steps, output=output)
        return

    assert not (
        skip_cuda and skip_cpu and skip_gpu2x and skip_gpu2x_sharded and skip_pyelastica
    ), "At least one backend sweep is required."

    series: dict[str, list[SweepPoint]] = {}

    if not skip_cuda:
        series["cuda"] = _sweep_backend(
            "cuda",
            cuda_min_exp,
            cuda_max_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            verbose=not quiet,
        )
    if not skip_cpu:
        series["jax-cpu"] = _sweep_backend(
            "cpu",
            cpu_min_exp,
            cpu_max_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            verbose=not quiet,
        )
    if not skip_gpu2x:
        series["gpu2x"] = _sweep_backend(
            "gpu2x",
            gpu2x_min_exp,
            gpu2x_max_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            verbose=not quiet,
        )
    if not skip_gpu2x_sharded:
        series["gpu2x_sharded"] = _sweep_backend(
            "gpu2x_sharded",
            gpu2x_sharded_min_exp,
            gpu2x_sharded_max_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            verbose=not quiet,
        )
    if not skip_pyelastica:
        series["pyelastica"] = _sweep_backend(
            "pyelastica",
            pyelastica_min_exp,
            pyelastica_max_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            verbose=not quiet,
        )

    csv_path = csv_output if csv_output is not None else output.with_suffix(".csv")
    _export_scaling_csv(series, steps=steps, output=csv_path)
    _export_scaling_plot(series, steps=steps, output=output)


if __name__ == "__main__":
    main()
