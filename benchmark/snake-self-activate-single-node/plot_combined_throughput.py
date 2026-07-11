"""Combine GH200 and Kole snake-throughput CSVs and plot scaling curves."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BACKEND_LABELS = {
    "pyelastica": "PyElastica (numba)",
    "jax-cpu": "JAX (CPU)",
    "cuda": "JAX (GPU-CUDA)",
    "gpu2x": "JAX (GPU-CUDA 2x)",
}

BACKEND_COLORS = {
    "PyElastica (numba)": "#C45C26",
    "JAX (CPU)": "#3B6EA5",
    "JAX (GPU-CUDA)": "#1B7F5A",
    "JAX (GPU-CUDA 2x)": "#6D28D9",
}

MACHINE_STYLES = {
    "GH200": {
        "linestyle": "-",
        "marker": "o",
        "markersize": 7.5,
        "linewidth": 2.2,
        "filled": True,
    },
    "Kole": {
        "linestyle": "--",
        "marker": "s",
        "markersize": 6.5,
        "linewidth": 2.0,
        "filled": False,
    },
}

MACHINE_FULL_NAMES = {
    "GH200": "GH200",
    "Kole": "i7-12700K + RTX 3060",
}

BACKEND_ORDER = (
    "PyElastica (numba)",
    "JAX (CPU)",
    "JAX (GPU-CUDA)",
    "JAX (GPU-CUDA 2x)",
)
MACHINE_ORDER = ("GH200", "Kole")

Row = tuple[str, str, str, int, int, float, int, float]


def _load_machine_csv(csv_path: Path, machine: str) -> list[Row]:
    """Load one machine CSV and attach display labels."""
    assert machine in MACHINE_FULL_NAMES, f"Unknown machine {machine!r}."
    with csv_path.open(newline="") as handle:
        reader = list(csv.DictReader(handle))
    assert reader, f"CSV {csv_path} is empty."

    rows: list[Row] = []
    for raw in reader:
        backend = raw["backend"]
        assert backend in BACKEND_LABELS, f"Unknown backend {backend!r} in {csv_path}."
        n_snakes = int(raw["n_snakes"])
        steps = int(raw["steps"])
        walltime = float(raw["rollout_walltime_s"])
        throughput = n_snakes * steps / walltime
        rows.append(
            (
                machine,
                backend,
                BACKEND_LABELS[backend],
                int(raw["exponent"]),
                n_snakes,
                walltime,
                steps,
                throughput,
            )
        )
    return rows


def combine_csvs(gh_csv: Path, kole_csv: Path) -> list[Row]:
    """Merge GH200 and Kole benchmark CSVs into one long-form table."""
    combined = _load_machine_csv(gh_csv, "GH200") + _load_machine_csv(kole_csv, "Kole")
    backend_rank = {label: index for index, label in enumerate(BACKEND_ORDER)}
    machine_rank = {label: index for index, label in enumerate(MACHINE_ORDER)}
    combined.sort(
        key=lambda row: (
            machine_rank[row[0]],
            backend_rank[row[2]],
            row[4],
        )
    )
    return combined


def plot_scaling(rows: list[Row], *, output: Path, steps: int) -> None:
    """Plot log-log walltime and throughput scaling for both machines."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    series: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for machine, _backend, backend_label, _exp, n_snakes, walltime, _steps, thr in rows:
        series[(machine, backend_label)].append((n_snakes, walltime, thr))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    walltime_ax, throughput_ax = axes

    for machine in MACHINE_ORDER:
        style = MACHINE_STYLES[machine]
        for backend in BACKEND_ORDER:
            points = series.get((machine, backend))
            if not points:
                continue
            points = sorted(points, key=lambda item: item[0])
            n_snakes = np.asarray([item[0] for item in points], dtype=np.float64)
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
            walltime_ax.loglog(n_snakes, walltimes, **common)
            throughput_ax.loglog(n_snakes, throughputs, **common)

    for ax in axes:
        ax.set_xlabel("number of rods")
        ax.grid(True, which="major", color="#D0D5DD", linewidth=0.9)
        ax.grid(True, which="minor", color="#EEF0F3", linewidth=0.6)
        ax.set_axisbelow(True)

    walltime_ax.set_ylabel("rollout walltime (s)")
    walltime_ax.set_title("Walltime scaling")
    throughput_ax.set_ylabel("throughput (rod-steps / s)")
    throughput_ax.set_title("Throughput scaling")

    backend_handles = [
        Line2D(
            [0],
            [0],
            color=BACKEND_COLORS[backend],
            linewidth=2.2,
            marker="o",
            markersize=7,
            label=backend,
        )
        for backend in BACKEND_ORDER
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
    ]

    backend_legend = fig.legend(
        handles=backend_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.05, 0.15),
        handlelength=2.4,
        labelspacing=0.55,
        title="Backend",
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
        bbox_to_anchor=(0.25, 0.15),
        handlelength=2.8,
        labelspacing=0.55,
        title="Machine",
        title_fontsize=11,
        borderaxespad=0.0,
        alignment="left",
    )

    fig.suptitle(
        f"Multi-rods (snake-activated) rollout scaling\n({steps} steps, 20 elements/rod)",
        fontsize=16,
        fontweight="semibold",
        x=0.02,
        y=0.99,
        ha="left",
        linespacing=1.3,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.84,
        bottom=0.26,
        wspace=0.13,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote plot: {output}")


@click.command()
@click.option(
    "--gh-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("snake_throughput_scaling_GH.csv"),
    show_default=True,
    help="GH200 benchmark CSV.",
)
@click.option(
    "--kole-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("snake_throughput_scaling_kole.csv"),
    show_default=True,
    help="Kole (i7-12700K + RTX 3060) benchmark CSV.",
)
@click.option(
    "--csv-output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("snake_throughput_scaling_combined.csv"),
    show_default=True,
    help="Combined long-form CSV output path.",
)
@click.option(
    "--plot-output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("snake_throughput_scaling_combined.png"),
    show_default=True,
    help="Combined scaling plot output path.",
)
def main(
    gh_csv: Path,
    kole_csv: Path,
    csv_output: Path,
    plot_output: Path,
) -> None:
    """Combine machine CSVs and write a two-panel scaling figure."""
    combined = combine_csvs(gh_csv, kole_csv)
    steps = combined[0][6]
    assert all(row[6] == steps for row in combined), "Mixed step counts across CSVs."

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "machine",
                "backend",
                "backend_label",
                "exponent",
                "n_snakes",
                "rollout_walltime_s",
                "steps",
                "throughput_snakes_per_s",
            )
        )
        writer.writerows(combined)
    print(f"wrote csv: {csv_output}")

    plot_scaling(combined, output=plot_output, steps=steps)


if __name__ == "__main__":
    main()
