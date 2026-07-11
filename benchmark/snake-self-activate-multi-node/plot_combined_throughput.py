"""Combine GG CPU and GH200 GPU MPI weak-scaling CSVs into one figure."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent

MACHINE_STYLES = {
    "GH200": {
        "linestyle": "-",
        "marker": "o",
        "markersize": 7.5,
        "linewidth": 2.2,
        "filled": True,
    },
    "GG": {
        "linestyle": "--",
        "marker": "s",
        "markersize": 6.5,
        "linewidth": 2.0,
        "filled": False,
    },
}

MACHINE_FULL_NAMES = {
    "GH200": "GH200 (GPU)",
    "GG": "GG (CPU) - 2x72=144 cores",
}

MACHINE_ORDER = ("GH200", "GG")


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """One weak-scaling CSV mapped to display metadata."""

    path: Path
    machine: str
    backend_label: str
    color: str
    # CPU MPI ranks represented by one device rank on the plot x-axis.
    # GG CPU ranks map 1:1; GH200 maps 1 GPU -> 144 (or 72 if halfed).
    cpu_ranks_per_device: int


# GPU rank work is expressed in GG CPU-rank equivalents:
#   full  = 144 ranks/GPU x 64 rods/rank = 9216 rods/GPU
#   halfed =  72 ranks/GPU x 64 rods/rank = 4608 rods/GPU
SERIES_SPECS = (
    SeriesSpec(
        path=SCRIPT_DIR / "scaling_plot_gpu_N8.csv",
        machine="GH200",
        backend_label="JAX (GPU-CUDA), 144 ranks/GPU",
        color="#15803D",
        cpu_ranks_per_device=144,
    ),
    SeriesSpec(
        path=SCRIPT_DIR / "scaling_plot_gpu_N8_halfed.csv",
        machine="GH200",
        backend_label="JAX (GPU-CUDA), halfed, 72 ranks/GPU",
        color="#D97706",
        cpu_ranks_per_device=72,
    ),
    SeriesSpec(
        path=SCRIPT_DIR / "scaling_plot_N28.csv",
        machine="GG",
        backend_label="JAX (CPU), 64 rods/rank",
        color="#1D4ED8",
        cpu_ranks_per_device=1,
    ),
    SeriesSpec(
        path=SCRIPT_DIR / "snake_mpi_scaline_4N.csv",
        machine="GG",
        backend_label="JAX (CPU), 256 rods/rank",
        color="#9F1239",
        cpu_ranks_per_device=1,
    ),
    SeriesSpec(
        path=SCRIPT_DIR / "snake_mpi_scaling.csv",
        machine="GG",
        backend_label="JAX (CPU), 32 rods/rank",
        color="#7C3AED",
        cpu_ranks_per_device=1,
    ),
)

BACKEND_ORDER = tuple(spec.backend_label for spec in SERIES_SPECS)
BACKEND_COLORS = {spec.backend_label: spec.color for spec in SERIES_SPECS}

# machine, backend_label, mpi_size, effective_cpu_ranks, snakes_per_rank,
# n_snakes, walltime, steps, throughput
Row = tuple[str, str, int, int, int, int, float, int, float]


def _aggregate_csv(
    csv_path: Path,
    machine: str,
    backend_label: str,
    *,
    cpu_ranks_per_device: int,
) -> list[Row]:
    """Load per-rank walltimes and reduce each MPI size to max walltime."""
    assert machine in MACHINE_FULL_NAMES, f"Unknown machine {machine!r}."
    assert cpu_ranks_per_device > 0, "cpu_ranks_per_device must be positive."
    with csv_path.open(newline="") as handle:
        reader = list(csv.DictReader(handle))
    assert reader, f"CSV {csv_path} is empty."

    grouped: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    for raw in reader:
        key = (
            int(raw["mpi_size"]),
            int(raw["snakes_per_rank"]),
            int(raw["n_snakes"]),
            int(raw["steps"]),
        )
        grouped[key].append(float(raw["rollout_walltime_s"]))

    rows: list[Row] = []
    for (mpi_size, snakes_per_rank, n_snakes, steps), walltimes in grouped.items():
        max_walltime = float(np.max(walltimes))
        assert max_walltime > 0.0, f"Non-positive walltime in {csv_path}."
        throughput = n_snakes * steps / max_walltime
        effective_cpu_ranks = mpi_size * cpu_ranks_per_device
        rows.append(
            (
                machine,
                backend_label,
                mpi_size,
                effective_cpu_ranks,
                snakes_per_rank,
                n_snakes,
                max_walltime,
                steps,
                throughput,
            )
        )
    return rows


def combine_csvs(specs: tuple[SeriesSpec, ...] = SERIES_SPECS) -> list[Row]:
    """Merge configured machine CSVs into one long-form table."""
    combined: list[Row] = []
    for spec in specs:
        assert spec.path.is_file(), f"Missing CSV: {spec.path}"
        combined.extend(
            _aggregate_csv(
                spec.path,
                spec.machine,
                spec.backend_label,
                cpu_ranks_per_device=spec.cpu_ranks_per_device,
            )
        )

    backend_rank = {label: index for index, label in enumerate(BACKEND_ORDER)}
    machine_rank = {label: index for index, label in enumerate(MACHINE_ORDER)}
    combined.sort(
        key=lambda row: (
            machine_rank[row[0]],
            backend_rank[row[1]],
            row[3],
        )
    )
    return combined


def plot_scaling(rows: list[Row], *, output: Path, steps: int) -> None:
    """Plot log-log walltime and throughput weak-scaling curves."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    series: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for (
        machine,
        backend_label,
        _mpi_size,
        effective_cpu_ranks,
        _spr,
        _n,
        walltime,
        _steps,
        thr,
    ) in rows:
        series[(machine, backend_label)].append((effective_cpu_ranks, walltime, thr))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2))
    walltime_ax, throughput_ax = axes

    for machine in MACHINE_ORDER:
        style = MACHINE_STYLES[machine]
        for backend in BACKEND_ORDER:
            points = series.get((machine, backend))
            if not points:
                continue
            points = sorted(points, key=lambda item: item[0])
            x_ranks = np.asarray([item[0] for item in points], dtype=np.float64)
            walltimes = np.asarray([item[1] for item in points], dtype=np.float64)
            throughputs = np.asarray([item[2] for item in points], dtype=np.float64)
            color = BACKEND_COLORS[backend]
            common = {
                "color": color,
                "linestyle": style["linestyle"],
                "marker": style["marker"],
                "markersize": style["markersize"],
                "linewidth": style["linewidth"],
                "markeredgewidth": 1.4,
                "markeredgecolor": color,
                "markerfacecolor": color if style["filled"] else "none",
            }
            walltime_ax.loglog(x_ranks, walltimes, **common)
            throughput_ax.loglog(x_ranks, throughputs, **common)

    for ax in axes:
        ax.set_xlabel("effective CPU ranks (1 GPU = 144, halfed = 72)")
        ax.grid(True, which="major", color="#D0D5DD", linewidth=0.9)
        ax.grid(True, which="minor", color="#EEF0F3", linewidth=0.6)
        ax.set_axisbelow(True)

    walltime_ax.set_ylabel("max per-rank rollout walltime (s)")
    walltime_ax.set_title("Weak-scaling walltime")
    throughput_ax.set_ylabel("throughput (rod-steps / s)")
    throughput_ax.set_title("Weak-scaling throughput")

    backend_handles = [
        Line2D(
            [0],
            [0],
            color=BACKEND_COLORS[backend],
            linewidth=2.5,
            # marker="o",
            # markersize=7,
            label=backend,
        )
        for backend in BACKEND_ORDER
        if any(row[1] == backend for row in rows)
    ]
    machine_handles = [
        Line2D(
            [0],
            [0],
            color="#4A5568",
            linestyle=MACHINE_STYLES[machine]["linestyle"],
            marker=MACHINE_STYLES[machine]["marker"],
            markersize=MACHINE_STYLES[machine]["markersize"],
            linewidth=MACHINE_STYLES[machine]["linewidth"],
            markerfacecolor=(
                "#4A5568" if MACHINE_STYLES[machine]["filled"] else "none"
            ),
            markeredgecolor="#4A5568",
            markeredgewidth=1.4,
            label=MACHINE_FULL_NAMES[machine],
        )
        for machine in MACHINE_ORDER
        if any(row[0] == machine for row in rows)
    ]

    backend_legend = fig.legend(
        handles=backend_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.05, 0.20),
        handlelength=2.4,
        labelspacing=0.55,
        title="Workload",
        title_fontsize=11,
        borderaxespad=0.0,
        alignment="left",
    )
    fig.add_artist(backend_legend)
    fig.legend(
        handles=machine_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.48, 0.20),
        handlelength=2.8,
        labelspacing=0.55,
        title="Machine",
        title_fontsize=11,
        borderaxespad=0.0,
        alignment="left",
    )

    fig.suptitle(
        f"Weak scaling (snake-activated)\n"
        f"({steps} steps, 20 elements/rod; "
        f"1 GPU = 144 CPU ranks, halfed = 72)",
        fontsize=15,
        fontweight="semibold",
        x=0.02,
        y=0.99,
        ha="left",
        linespacing=1.3,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.82,
        bottom=0.32,
        wspace=0.13,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote plot: {output}")


@click.command()
@click.option(
    "--csv-output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=SCRIPT_DIR / "snake_mpi_throughput_scaling_combined.csv",
    show_default=True,
    help="Combined long-form CSV output path.",
)
@click.option(
    "--plot-output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=SCRIPT_DIR / "snake_mpi_throughput_scaling_combined.png",
    show_default=True,
    help="Combined weak-scaling plot output path.",
)
def main(csv_output: Path, plot_output: Path) -> None:
    """Combine machine CSVs and write a two-panel weak-scaling figure."""
    combined = combine_csvs()
    steps = combined[0][7]
    assert all(row[7] == steps for row in combined), "Mixed step counts across CSVs."

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "machine",
                "backend_label",
                "mpi_size",
                "effective_cpu_ranks",
                "snakes_per_rank",
                "n_snakes",
                "rollout_walltime_s",
                "steps",
                "throughput_rod_steps_per_s",
            )
        )
        writer.writerows(combined)
    print(f"wrote csv: {csv_output}")

    plot_scaling(combined, output=plot_output, steps=steps)


if __name__ == "__main__":
    main()
