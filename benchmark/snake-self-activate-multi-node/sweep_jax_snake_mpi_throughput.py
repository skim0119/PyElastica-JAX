"""Sweep MPI world size for weak-scaling multi-snake rollout throughput."""

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
JAX_WORKER = SCRIPT_DIR / "jax_snake_mpi_throughput.py"
PYELASTICA_WORKER = SCRIPT_DIR / "pyelastica_snake_mpi_throughput.py"

RolloutPoint = tuple[int, int, int, np.ndarray]
RolloutSample = tuple[int, int, int, int, float]


def _worker_script(backend: str) -> Path:
    if backend == "pyelastica":
        return PYELASTICA_WORKER
    return JAX_WORKER


def _parse_rollout_walltimes(output: str) -> np.ndarray:
    match = re.search(r"rollout_walltimes=([0-9eE+.,-]+)", output)
    assert match is not None, f"Could not parse rollout walltimes from:\n{output}"
    values = tuple(
        float(item) for item in match.group(1).split(",") if item.strip() != ""
    )
    assert values, f"Parsed empty rollout walltime list from:\n{output}"
    return np.asarray(values, dtype=np.float64)


def _snakes_per_rank(
    *,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
) -> int:
    return snakes_per_rank_multiplier * (2**snakes_per_rank_exp)


def _global_snake_count(
    *,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    mpi_size: int,
) -> int:
    return (
        _snakes_per_rank(
            snakes_per_rank_exp=snakes_per_rank_exp,
            snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        )
        * mpi_size
    )


def _mpi_worker_env(*, mpi_size: int) -> dict[str, str]:
    """Environment for one weak-scaling MPI worker launch."""
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


def _default_mpi_bind_to_core() -> bool:
    return sys.platform != "darwin"


def _build_mpiexec_command(
    *,
    mpi_size: int,
    python_executable: str,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    backend: str,
    bind_to_core: bool,
    vertical: bool,
) -> list[str]:
    worker = _worker_script(backend)
    command = [
        "ibrun",
        "-n",
        str(mpi_size),
        python_executable,
        str(worker),
        "--snakes-per-rank-exp",
        str(snakes_per_rank_exp),
        "--snakes-per-rank-multiplier",
        str(snakes_per_rank_multiplier),
        "--steps",
        str(steps),
        "--warmup-runs",
        str(warmup_runs),
    ]
    if backend != "pyelastica":
        command.extend(["--backend", backend])
        if vertical:
            command.append("--vertical")
    return command


def _summarize_weak_scaling(points: list[RolloutPoint]) -> None:
    baseline_max: float | None = None
    print("\nWeak-scaling summary (max per-rank rollout time):")
    for mpi_size, snakes_per_rank, n_snakes, rollout_walltimes in points:
        max_walltime = float(np.max(rollout_walltimes))
        if baseline_max is None:
            baseline_max = max_walltime
        efficiency = baseline_max / max_walltime if max_walltime > 0.0 else 0.0
        print(
            f"  mpi_size={mpi_size:>2d}  snakes_per_rank={snakes_per_rank:>4d}  "
            f"n_snakes={n_snakes:>4d}  max={max_walltime:.4f}s  "
            f"weak_eff={efficiency:.3f}"
        )


def _format_mpi_worker_failure(
    exc: subprocess.CalledProcessError,
    *,
    mpi_size: int,
    command: list[str],
) -> str:
    """Build a readable error report for a failed MPI worker launch."""
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


def _run_mpi_point(
    *,
    mpi_size: int,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    backend: str,
    python_executable: str,
    bind_to_core: bool,
    vertical: bool,
) -> RolloutPoint:
    command = _build_mpiexec_command(
        mpi_size=mpi_size,
        python_executable=python_executable,
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        steps=steps,
        warmup_runs=warmup_runs,
        backend=backend,
        bind_to_core=bind_to_core,
        vertical=vertical,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=_mpi_worker_env(mpi_size=mpi_size),
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            _format_mpi_worker_failure(exc, mpi_size=mpi_size, command=command)
        ) from exc
    snakes_per_rank = _snakes_per_rank(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
    )
    n_snakes_total = _global_snake_count(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        mpi_size=mpi_size,
    )
    return (
        mpi_size,
        snakes_per_rank,
        n_snakes_total,
        _parse_rollout_walltimes(completed.stdout),
    )


def _sweep_mpi_sizes(
    mpi_sizes: tuple[int, ...],
    *,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    backend: str,
    python_executable: str,
    bind_to_core: bool,
    vertical: bool,
    verbose: bool,
) -> list[RolloutPoint]:
    snakes_per_rank = _snakes_per_rank(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
    )
    results: list[RolloutPoint] = []
    for mpi_size in tqdm(mpi_sizes, desc="mpi", disable=not verbose):
        assert mpi_size > 0, "MPI world size must be positive."
        point = _run_mpi_point(
            mpi_size=mpi_size,
            snakes_per_rank_exp=snakes_per_rank_exp,
            snakes_per_rank_multiplier=snakes_per_rank_multiplier,
            steps=steps,
            warmup_runs=warmup_runs,
            backend=backend,
            python_executable=python_executable,
            bind_to_core=bind_to_core,
            vertical=vertical,
        )
        _, _, n_snakes_total, rollout_walltimes = point
        print(
            f"mpi_size={mpi_size} snakes_per_rank={snakes_per_rank} "
            f"n_snakes={n_snakes_total}: gathered {rollout_walltimes.size} "
            "rollout walltimes"
        )
        results.append(point)
    return results


def _flatten_rollout_samples(points: list[RolloutPoint]) -> list[RolloutSample]:
    return [
        (mpi_size, snakes_per_rank, n_snakes, rank, float(walltime))
        for mpi_size, snakes_per_rank, n_snakes, rollout_walltimes in points
        for rank, walltime in enumerate(rollout_walltimes.tolist())
    ]


def _export_scaling_plot(
    points: list[RolloutPoint],
    *,
    backend: str,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    vertical: bool,
    output: Path,
) -> None:
    snakes_per_rank = _snakes_per_rank(
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
    )
    mpi_sizes = np.asarray([point[0] for point in points], dtype=np.float64)
    max_walltimes = np.asarray(
        [float(np.max(point[3])) for point in points],
        dtype=np.float64,
    )
    baseline = float(max_walltimes[0]) if max_walltimes.size else 0.0
    if backend == "pyelastica":
        layout_label = "numba"
    else:
        layout_label = "vertical" if vertical else "horizontal"

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(mpi_sizes, max_walltimes, marker="o", label="max per-rank rollout")
    if baseline > 0.0:
        ax.axhline(
            baseline,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            label="mpi_size=1 baseline",
        )
    ax.set_xlabel("MPI world size")
    ax.set_ylabel("rollout walltime (s)")
    ax.set_title(
        f"MPI weak scaling ({backend}, {layout_label}, "
        f"snakes_per_rank={snakes_per_rank}, {steps} steps, 20 elements/snake)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote plot: {output}")


def _export_scaling_csv(
    points: list[RolloutPoint],
    *,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    vertical: bool,
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
                "rank",
                "rollout_walltime_s",
                "snakes_per_rank_exp",
                "snakes_per_rank_multiplier",
                "steps",
                "vertical",
            )
        )
        for (
            mpi_size,
            snakes_per_rank,
            n_snakes,
            rank,
            walltime,
        ) in _flatten_rollout_samples(points):
            writer.writerow(
                (
                    mpi_size,
                    snakes_per_rank,
                    n_snakes,
                    rank,
                    walltime,
                    snakes_per_rank_exp,
                    snakes_per_rank_multiplier,
                    steps,
                    int(vertical),
                )
            )
    print(f"wrote csv: {output}")


def _load_scaling_csv(
    csv_path: Path,
) -> tuple[list[RolloutPoint], int, int, int, bool]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"CSV {csv_path} is empty."

    if "snakes_per_rank_exp" in rows[0]:
        snakes_per_rank_exp = int(rows[0]["snakes_per_rank_exp"])
    else:
        legacy_exp = int(rows[0]["n_snakes_exp"])
        mpi_size = int(rows[0]["mpi_size"])
        snakes_per_rank_exp = max(0, legacy_exp - int(np.log2(mpi_size)))

    snakes_per_rank_multiplier = int(rows[0].get("snakes_per_rank_multiplier", 1))
    steps = int(rows[0]["steps"])
    vertical = bool(int(rows[0].get("vertical", 0)))
    grouped_rows: list[tuple[int, list[dict[str, str]]]] = []
    for mpi_size, mpi_size_rows_iter in itertools.groupby(
        sorted(rows, key=lambda row: (int(row["mpi_size"]), int(row.get("rank", 0)))),
        key=lambda row: int(row["mpi_size"]),
    ):
        grouped_rows.append((mpi_size, list(mpi_size_rows_iter)))

    points = []
    for mpi_size, mpi_size_rows in grouped_rows:
        first_row = mpi_size_rows[0]
        points.append(
            (
                mpi_size,
                int(
                    first_row.get(
                        "snakes_per_rank",
                        _snakes_per_rank(
                            snakes_per_rank_exp=snakes_per_rank_exp,
                            snakes_per_rank_multiplier=snakes_per_rank_multiplier,
                        ),
                    )
                ),
                int(first_row["n_snakes"]),
                np.asarray(
                    [
                        float(row["rollout_walltime_s"])
                        for row in sorted(
                            mpi_size_rows, key=lambda row: int(row.get("rank", 0))
                        )
                    ],
                    dtype=np.float64,
                ),
            )
        )

    return points, snakes_per_rank_exp, snakes_per_rank_multiplier, steps, vertical


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--snakes-per-rank-exp",
    type=int,
    default=4,
    show_default=True,
    help="Fixed base snakes per MPI rank (2 ** exp) for weak scaling.",
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
@click.option(
    "--bind-to-core/--no-bind-to-core",
    default=_default_mpi_bind_to_core(),
    show_default=True,
    help="Pin each MPI rank to a CPU core via mpiexec (unsupported on macOS).",
)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda", "pyelastica"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Rollout backend: JAX cpu/cuda or PyElastica (numba).",
)
@click.option(
    "--vertical",
    is_flag=True,
    help="Use vertical (stacked-axis) rod memory block packing (JAX only).",
)
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int,
    steps: int,
    warmup_runs: int,
    mpi_sizes: str,
    output: Path,
    csv_output: Path | None,
    from_csv: Path | None,
    python_executable: str,
    backend: str,
    bind_to_core: bool,
    vertical: bool,
    quiet: bool,
) -> None:
    assert steps > 0, "steps must be positive."
    assert snakes_per_rank_multiplier > 0, (
        "snakes_per_rank_multiplier must be positive."
    )
    if backend == "pyelastica":
        assert not vertical, "--vertical is only supported for JAX backends."

    if from_csv is not None:
        (
            points,
            snakes_per_rank_exp,
            snakes_per_rank_multiplier,
            steps,
            vertical,
        ) = _load_scaling_csv(from_csv)
        _export_scaling_plot(
            points,
            backend=backend,
            snakes_per_rank_exp=snakes_per_rank_exp,
            snakes_per_rank_multiplier=snakes_per_rank_multiplier,
            steps=steps,
            vertical=vertical,
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
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        steps=steps,
        warmup_runs=warmup_runs,
        backend=backend,
        python_executable=python_executable,
        bind_to_core=bind_to_core,
        vertical=vertical,
        verbose=not quiet,
    )
    _summarize_weak_scaling(points)
    csv_path = csv_output if csv_output is not None else output.with_suffix(".csv")
    _export_scaling_csv(
        points,
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        steps=steps,
        vertical=vertical,
        output=csv_path,
    )
    _export_scaling_plot(
        points,
        backend=backend,
        snakes_per_rank_exp=snakes_per_rank_exp,
        snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        steps=steps,
        vertical=vertical,
        output=output,
    )


if __name__ == "__main__":
    main()
