"""Sweep MPI world size for weak-scaling multi-snake rollout throughput."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
WORKER = SCRIPT_DIR / "jax_snake_mpi_throughput.py"

RolloutPoint = tuple[int, int, int, float]


def _parse_rollout_walltime(output: str) -> float:
    match = re.search(r"rollout_walltime=([0-9.eE+-]+)", output)
    assert match is not None, f"Could not parse rollout walltime from:\n{output}"
    return float(match.group(1))


def _global_snake_count(*, snakes_per_rank_exp: int, mpi_size: int) -> int:
    return (2**snakes_per_rank_exp) * mpi_size


def _run_mpi_point(
    *,
    mpi_size: int,
    snakes_per_rank_exp: int,
    steps: int,
    warmup_runs: int,
    python_executable: str,
) -> RolloutPoint:
    command = [
        "mpiexec",
        "-n",
        str(mpi_size),
        python_executable,
        str(WORKER),
        "--snakes-per-rank-exp",
        str(snakes_per_rank_exp),
        "--steps",
        str(steps),
        "--warmup-runs",
        str(warmup_runs),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    snakes_per_rank = 2**snakes_per_rank_exp
    n_snakes_total = _global_snake_count(
        snakes_per_rank_exp=snakes_per_rank_exp,
        mpi_size=mpi_size,
    )
    return (
        mpi_size,
        snakes_per_rank,
        n_snakes_total,
        _parse_rollout_walltime(completed.stdout),
    )


def _sweep_mpi_sizes(
    mpi_sizes: tuple[int, ...],
    *,
    snakes_per_rank_exp: int,
    steps: int,
    warmup_runs: int,
    python_executable: str,
    verbose: bool,
) -> list[RolloutPoint]:
    snakes_per_rank = 2**snakes_per_rank_exp
    results: list[RolloutPoint] = []
    for mpi_size in tqdm(mpi_sizes, desc="mpi", disable=not verbose):
        assert mpi_size > 0, "MPI world size must be positive."
        point = _run_mpi_point(
            mpi_size=mpi_size,
            snakes_per_rank_exp=snakes_per_rank_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            python_executable=python_executable,
        )
        _, _, n_snakes_total, rollout_walltime = point
        print(
            f"mpi_size={mpi_size} snakes_per_rank={snakes_per_rank} "
            f"n_snakes={n_snakes_total}: rollout_walltime={rollout_walltime:.6f}s"
        )
        results.append(point)
    return results


def _export_scaling_plot(
    points: list[RolloutPoint],
    *,
    snakes_per_rank_exp: int,
    steps: int,
    output: Path,
) -> None:
    mpi_sizes = np.asarray([point[0] for point in points], dtype=np.float64)
    walltimes = np.asarray([point[3] for point in points], dtype=np.float64)
    snakes_per_rank = 2**snakes_per_rank_exp

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(mpi_sizes, walltimes, marker="o")
    ax.set_xlabel("MPI world size")
    ax.set_ylabel("rollout walltime (s)")
    ax.set_title(
        "MPI weak scaling "
        f"(snakes_per_rank={snakes_per_rank}, {steps} steps, 20 elements/snake)"
    )
    ax.grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote plot: {output}")


def _export_scaling_csv(
    points: list[RolloutPoint],
    *,
    snakes_per_rank_exp: int,
    steps: int,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "mpi_size",
                "snakes_per_rank",
                "n_snakes",
                "rollout_walltime_s",
                "snakes_per_rank_exp",
                "steps",
            )
        )
        for mpi_size, snakes_per_rank, n_snakes, walltime in points:
            writer.writerow(
                (
                    mpi_size,
                    snakes_per_rank,
                    n_snakes,
                    walltime,
                    snakes_per_rank_exp,
                    steps,
                )
            )
    print(f"wrote csv: {output}")


def _load_scaling_csv(
    csv_path: Path,
) -> tuple[list[RolloutPoint], int, int]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"CSV {csv_path} is empty."

    if "snakes_per_rank_exp" in rows[0]:
        snakes_per_rank_exp = int(rows[0]["snakes_per_rank_exp"])
    else:
        legacy_exp = int(rows[0]["n_snakes_exp"])
        mpi_size = int(rows[0]["mpi_size"])
        snakes_per_rank_exp = max(0, legacy_exp - int(np.log2(mpi_size)))

    steps = int(rows[0]["steps"])
    points = [
        (
            int(row["mpi_size"]),
            int(row.get("snakes_per_rank", 2**snakes_per_rank_exp)),
            int(row["n_snakes"]),
            float(row["rollout_walltime_s"]),
        )
        for row in rows
    ]
    points.sort(key=lambda item: item[0])
    return points, snakes_per_rank_exp, steps


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--snakes-per-rank-exp",
    type=int,
    default=4,
    show_default=True,
    help="Fixed snakes per MPI rank (2 ** exp) for weak scaling.",
)
@click.option("--steps", type=int, default=1000, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
@click.option(
    "--mpi-sizes",
    type=str,
    default="1,2,4",
    show_default=True,
    help="Comma-separated MPI world sizes to benchmark.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("snake_mpi_throughput_scaling.png"),
    show_default=True,
)
@click.option(
    "--csv-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output CSV path (default: plot path with .csv suffix).",
)
@click.option(
    "--from-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Regenerate the plot from a saved CSV instead of running sweeps.",
)
@click.option(
    "--python",
    "python_executable",
    type=str,
    default=sys.executable,
    show_default=True,
    help="Python executable passed to mpiexec.",
)
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    snakes_per_rank_exp: int,
    steps: int,
    warmup_runs: int,
    mpi_sizes: str,
    output: Path,
    csv_output: Path | None,
    from_csv: Path | None,
    python_executable: str,
    quiet: bool,
) -> None:
    assert steps > 0, "steps must be positive."

    if from_csv is not None:
        points, snakes_per_rank_exp, steps = _load_scaling_csv(from_csv)
        _export_scaling_plot(
            points,
            snakes_per_rank_exp=snakes_per_rank_exp,
            steps=steps,
            output=output,
        )
        return

    parsed_mpi_sizes = tuple(
        int(item.strip()) for item in mpi_sizes.split(",") if item.strip()
    )
    assert parsed_mpi_sizes, "At least one MPI world size is required."

    points = _sweep_mpi_sizes(
        parsed_mpi_sizes,
        snakes_per_rank_exp=snakes_per_rank_exp,
        steps=steps,
        warmup_runs=warmup_runs,
        python_executable=python_executable,
        verbose=not quiet,
    )
    csv_path = csv_output if csv_output is not None else output.with_suffix(".csv")
    _export_scaling_csv(
        points,
        snakes_per_rank_exp=snakes_per_rank_exp,
        steps=steps,
        output=csv_path,
    )
    _export_scaling_plot(
        points,
        snakes_per_rank_exp=snakes_per_rank_exp,
        steps=steps,
        output=output,
    )


if __name__ == "__main__":
    main()
