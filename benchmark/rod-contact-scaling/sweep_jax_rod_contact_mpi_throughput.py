"""Sweep MPI world size for weak-scaling rod-rod contact throughput."""

from __future__ import annotations

import csv
import itertools
import os
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
JAX_WORKER = SCRIPT_DIR / "jax_rod_contact_mpi_throughput.py"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _rod_contact_common import (  # noqa: E402
    mpi_global_rod_count,
    mpi_rods_per_rank,
)

RolloutPoint = tuple[int, int, int, np.ndarray]
RolloutSample = tuple[int, int, int, int, float]


def parse_rollout_walltimes(output: str) -> np.ndarray:
    """Parse ``rollout_walltimes=...`` from an MPI worker stdout blob."""
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


def build_mpiexec_command(
    *,
    mpi_size: int,
    python_executable: str,
    rods_per_rank_exp: int,
    rods_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    backend: str,
    vertical: bool,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
) -> list[str]:
    """Build the ``ibrun`` command for one weak-scaling worker launch."""
    command = [
        "ibrun",
        "-n",
        str(mpi_size),
        python_executable,
        str(JAX_WORKER),
        "--backend",
        backend,
        "--rods-per-rank-exp",
        str(rods_per_rank_exp),
        "--rods-per-rank-multiplier",
        str(rods_per_rank_multiplier),
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
    if vertical:
        command.append("--vertical")
    return command


def _format_mpi_worker_failure(
    exc: subprocess.CalledProcessError,
    *,
    mpi_size: int,
    command: list[str],
) -> str:
    command_text = " ".join(command)
    stdout = exc.stdout.strip() if exc.stdout else ""
    stderr = exc.stderr.strip() if exc.stderr else ""
    sections = [
        f"MPI weak-scaling worker failed for mpi_size={mpi_size} "
        f"(exit code {exc.returncode}).",
        f"Command: {command_text}",
    ]
    if stdout:
        sections.append(f"stdout:\n{stdout}")
    if stderr:
        sections.append(f"stderr:\n{stderr}")
    if not stdout and not stderr:
        sections.append("stdout/stderr: (empty)")
    return "\n\n".join(sections)


def run_mpi_point(
    *,
    mpi_size: int,
    rods_per_rank_exp: int,
    rods_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    backend: str,
    python_executable: str,
    vertical: bool,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
) -> RolloutPoint:
    """Launch one MPI world size and parse gathered rollout walltimes."""
    command = build_mpiexec_command(
        mpi_size=mpi_size,
        python_executable=python_executable,
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
        steps=steps,
        warmup_runs=warmup_runs,
        backend=backend,
        vertical=vertical,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=_mpi_worker_env(),
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            _format_mpi_worker_failure(exc, mpi_size=mpi_size, command=command)
        ) from exc
    return (
        mpi_size,
        mpi_rods_per_rank(
            rods_per_rank_exp=rods_per_rank_exp,
            rods_per_rank_multiplier=rods_per_rank_multiplier,
        ),
        mpi_global_rod_count(
            rods_per_rank_exp=rods_per_rank_exp,
            rods_per_rank_multiplier=rods_per_rank_multiplier,
            comm_size=mpi_size,
        ),
        parse_rollout_walltimes(completed.stdout),
    )


def sweep_mpi_sizes(
    mpi_sizes: tuple[int, ...],
    *,
    rods_per_rank_exp: int,
    rods_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    backend: str,
    python_executable: str,
    vertical: bool,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    verbose: bool,
) -> list[RolloutPoint]:
    """Sweep MPI world sizes for one layout/backend configuration."""
    per_rank = mpi_rods_per_rank(
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
    )
    results: list[RolloutPoint] = []
    for mpi_size in tqdm(mpi_sizes, desc="mpi", disable=not verbose):
        assert mpi_size > 0, "MPI world size must be positive."
        point = run_mpi_point(
            mpi_size=mpi_size,
            rods_per_rank_exp=rods_per_rank_exp,
            rods_per_rank_multiplier=rods_per_rank_multiplier,
            steps=steps,
            warmup_runs=warmup_runs,
            backend=backend,
            python_executable=python_executable,
            vertical=vertical,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase=broad_phase,
        )
        _, _, n_rods_total, rollout_walltimes = point
        print(
            f"mpi_size={mpi_size} rods_per_rank={per_rank} "
            f"n_rods={n_rods_total}: gathered {rollout_walltimes.size} "
            "rollout walltimes"
        )
        results.append(point)
    return results


def summarize_weak_scaling(points: list[RolloutPoint]) -> None:
    """Print max-per-rank weak-scaling efficiency versus the first point."""
    baseline_max: float | None = None
    print("\nWeak-scaling summary (max per-rank rollout time):")
    for mpi_size, per_rank, n_rods, rollout_walltimes in points:
        max_walltime = float(np.max(rollout_walltimes))
        if baseline_max is None:
            baseline_max = max_walltime
        efficiency = baseline_max / max_walltime if max_walltime > 0.0 else 0.0
        print(
            f"  mpi_size={mpi_size:>2d}  rods_per_rank={per_rank:>4d}  "
            f"n_rods={n_rods:>4d}  max={max_walltime:.4f}s  "
            f"weak_eff={efficiency:.3f}"
        )


def flatten_rollout_samples(points: list[RolloutPoint]) -> list[RolloutSample]:
    """Flatten gathered per-rank timings into long-form CSV rows."""
    return [
        (mpi_size, per_rank, n_rods, rank, float(walltime))
        for mpi_size, per_rank, n_rods, rollout_walltimes in points
        for rank, walltime in enumerate(rollout_walltimes.tolist())
    ]


def export_scaling_csv(
    points: list[RolloutPoint],
    *,
    backend: str,
    rods_per_rank_exp: int,
    rods_per_rank_multiplier: int,
    steps: int,
    vertical: bool,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    output: Path,
) -> None:
    """Write long-form CSV for later combined plotting."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "backend",
                "mpi_size",
                "rods_per_rank",
                "n_rods",
                "rank",
                "rollout_walltime_s",
                "rods_per_rank_exp",
                "rods_per_rank_multiplier",
                "steps",
                "vertical",
                "n_elements",
                "steps_between_detection",
                "broad_phase",
            )
        )
        for mpi_size, per_rank, n_rods, rank, walltime in flatten_rollout_samples(
            points
        ):
            writer.writerow(
                (
                    backend,
                    mpi_size,
                    per_rank,
                    n_rods,
                    rank,
                    walltime,
                    rods_per_rank_exp,
                    rods_per_rank_multiplier,
                    steps,
                    int(vertical),
                    n_elements,
                    steps_between_detection,
                    broad_phase,
                )
            )
    print(f"wrote csv: {output}")


def export_scaling_plot(
    points: list[RolloutPoint],
    *,
    backend: str,
    rods_per_rank_exp: int,
    rods_per_rank_multiplier: int,
    steps: int,
    vertical: bool,
    n_elements: int,
    output: Path,
) -> None:
    """Write a weak-scaling plot of max per-rank rollout time versus MPI size."""
    per_rank = mpi_rods_per_rank(
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
    )
    mpi_sizes = np.asarray([point[0] for point in points], dtype=np.float64)
    max_walltimes = np.asarray(
        [float(np.max(point[3])) for point in points],
        dtype=np.float64,
    )
    baseline = float(max_walltimes[0]) if max_walltimes.size else 0.0
    layout_label = "vertical" if vertical else "horizontal"

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(mpi_sizes, max_walltimes, marker="o", label="max per-rank rollout")
    if baseline > 0.0:
        ax.axhline(
            baseline,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            label="first mpi_size baseline",
        )
    ax.set_xlabel("MPI world size")
    ax.set_ylabel("rollout walltime (s)")
    ax.set_title(
        f"MPI weak scaling ({backend}, {layout_label}, "
        f"rods_per_rank={per_rank}, {steps} steps, {n_elements} elems/rod)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote plot: {output}")


def load_scaling_csv(
    csv_path: Path,
) -> tuple[list[RolloutPoint], str, int, int, int, bool, int, int, str]:
    """Load sweep results written by :func:`export_scaling_csv`."""
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"CSV {csv_path} is empty."

    backend = rows[0]["backend"]
    rods_per_rank_exp = int(rows[0]["rods_per_rank_exp"])
    rods_per_rank_multiplier = int(rows[0]["rods_per_rank_multiplier"])
    steps = int(rows[0]["steps"])
    vertical = bool(int(rows[0]["vertical"]))
    n_elements = int(rows[0]["n_elements"])
    steps_between_detection = int(rows[0]["steps_between_detection"])
    broad_phase = rows[0]["broad_phase"]

    grouped_rows: list[tuple[int, list[dict[str, str]]]] = []
    for mpi_size, mpi_size_rows_iter in itertools.groupby(
        sorted(rows, key=lambda row: (int(row["mpi_size"]), int(row["rank"]))),
        key=lambda row: int(row["mpi_size"]),
    ):
        grouped_rows.append((mpi_size, list(mpi_size_rows_iter)))

    points: list[RolloutPoint] = []
    for mpi_size, mpi_size_rows in grouped_rows:
        first_row = mpi_size_rows[0]
        points.append(
            (
                mpi_size,
                int(first_row["rods_per_rank"]),
                int(first_row["n_rods"]),
                np.asarray(
                    [
                        float(row["rollout_walltime_s"])
                        for row in sorted(
                            mpi_size_rows, key=lambda row: int(row["rank"])
                        )
                    ],
                    dtype=np.float64,
                ),
            )
        )
    return (
        points,
        backend,
        rods_per_rank_exp,
        rods_per_rank_multiplier,
        steps,
        vertical,
        n_elements,
        steps_between_detection,
        broad_phase,
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--rods-per-rank-exp",
    type=int,
    default=4,
    show_default=True,
    help="Fixed base rods per MPI rank (2 ** exp) for weak scaling.",
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
@click.option("--n-elements", type=int, default=10, show_default=True)
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
    "--mpi-sizes",
    type=str,
    default="1,2,4",
    show_default=True,
    help="Comma-separated MPI world sizes to benchmark.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("rod_contact_mpi_throughput_scaling.png"),
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
    help="Python executable passed to ibrun.",
)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cuda",
    show_default=True,
)
@click.option(
    "--vertical",
    is_flag=True,
    help="Use vertical (stacked-axis) rod memory block packing.",
)
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    rods_per_rank_exp: int,
    rods_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    mpi_sizes: str,
    output: Path,
    csv_output: Path | None,
    from_csv: Path | None,
    python_executable: str,
    backend: str,
    vertical: bool,
    quiet: bool,
) -> None:
    """Sweep MPI world size and export CSV + plot for rod-rod contact."""
    assert steps > 0, "steps must be positive."
    assert rods_per_rank_multiplier > 0, "rods_per_rank_multiplier must be positive."

    if from_csv is not None:
        (
            points,
            backend,
            rods_per_rank_exp,
            rods_per_rank_multiplier,
            steps,
            vertical,
            n_elements,
            _steps_between_detection,
            _broad_phase,
        ) = load_scaling_csv(from_csv)
        export_scaling_plot(
            points,
            backend=backend,
            rods_per_rank_exp=rods_per_rank_exp,
            rods_per_rank_multiplier=rods_per_rank_multiplier,
            steps=steps,
            vertical=vertical,
            n_elements=n_elements,
            output=output,
        )
        return

    parsed_mpi_sizes = tuple(
        int(item.strip()) for item in mpi_sizes.split(",") if item.strip()
    )
    assert parsed_mpi_sizes, "At least one MPI world size is required."

    points = sweep_mpi_sizes(
        parsed_mpi_sizes,
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
        steps=steps,
        warmup_runs=warmup_runs,
        backend=backend,
        python_executable=python_executable,
        vertical=vertical,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
        verbose=not quiet,
    )
    summarize_weak_scaling(points)
    csv_path = csv_output if csv_output is not None else output.with_suffix(".csv")
    export_scaling_csv(
        points,
        backend=backend,
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
        steps=steps,
        vertical=vertical,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
        output=csv_path,
    )
    export_scaling_plot(
        points,
        backend=backend,
        rods_per_rank_exp=rods_per_rank_exp,
        rods_per_rank_multiplier=rods_per_rank_multiplier,
        steps=steps,
        vertical=vertical,
        n_elements=n_elements,
        output=output,
    )


if __name__ == "__main__":
    main()
