"""Combine all multi-node weak-scaling CSVs into one figure."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import click
from matplotlib.patches import bbox_artist
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent

MACHINE_STYLES = {
    "GH200": {
        "linestyle": "-",
        "marker": "o",
        "markersize": 5.5,
        "linewidth": 1.8,
        "filled": True,
    },
    "GG": {
        "linestyle": "--",
        "marker": "o",
        "markersize": 5.0,
        "linewidth": 1.6,
        "filled": False,
    },
    "GG_half": {
        "linestyle": "--",
        "marker": "^",
        "markersize": 5.0,
        "linewidth": 1.6,
        "filled": False,
    },
}

MACHINE_FULL_NAMES = {
    "GH200": "GH200 (GPU)",
    "GG": "GG (CPU, 144 ranks/node)",
    "GG_half": "GG (CPU, 72 ranks/node)",
}

MACHINE_ORDER = ("GH200", "GG", "GG_half")

# machine, series_label, mpi_size, n_nodes, snakes_per_rank, n_snakes,
# walltime, steps, thr
Row = tuple[str, str, int, float, int, int, float, int, float]

GG_RANKS_PER_NODE = 144
GG_RANKS_PER_NODE_HALF = 72


def _is_source_csv(path: Path) -> bool:
    if path.suffix != ".csv" or "combined" in path.name:
        return False
    # Drop superseded small GPU sweeps; N16 is the primary GH200 set.
    if "gpu_N8" in path.name:
        return False
    # GPU half is a reduced rods/GPU load, not CPU affinity; omit from plot.
    name = path.name.lower()
    if "gpu" in name and "half" in name:
        return False
    return True


def _is_halfed_series(path: Path) -> bool:
    """Return True when the CSV name marks a halfed packing."""
    return "half" in path.name.lower()


def _machine_from_path(path: Path) -> str:
    """
    Return the machine style key for a CSV path.

    GG half (name contains ``half`` / ``halfed``) uses a distinct marker.
    GPU half CSVs are filtered out in ``_is_source_csv``.
    """
    if "gpu" in path.name.lower():
        return "GH200"
    return "GG_half" if _is_halfed_series(path) else "GG"


def _ranks_per_node(path: Path, machine: str) -> int:
    """
    MPI ranks represented by one node on the plot x-axis.

    GH200 keeps one GPU rank per node. GG uses 144 ranks/node, or 72 when the
    series name contains ``half`` / ``halfed``.
    """
    if machine.startswith("GH200"):
        return 1
    if machine == "GG_half" or _is_halfed_series(path):
        return GG_RANKS_PER_NODE_HALF
    return GG_RANKS_PER_NODE


def _label_from_path(path: Path, *, snakes_per_rank: int) -> str:
    """
    Build a short legend label from the CSV stem and per-rank workload.

    Machine / half packing is shown in the Machine legend, so strip ``gpu``,
    ``half``, and ``halfed`` from the series name. Keep ``vertical`` as a
    packing qualifier and ``rods/rank`` for workload size.
    """
    stem = path.stem
    if stem.startswith("scaling_plot_"):
        stem = stem[len("scaling_plot_") :]
    if stem.startswith("gpu_"):
        stem = stem[len("gpu_") :]
    if stem.endswith("_halfed"):
        stem = stem[: -len("_halfed")]
    elif stem.endswith("_half"):
        stem = stem[: -len("_half")]
    stem = stem.replace("_vertical", " vertical")
    stem = stem.replace("_", " ")
    return f"{stem} ({snakes_per_rank} rods/rank)"


def _discover_source_csvs(directory: Path = SCRIPT_DIR) -> list[Path]:
    """Return every non-combined CSV in the multi-node benchmark folder."""
    paths = sorted(
        (path for path in directory.iterdir() if _is_source_csv(path)),
        key=lambda path: (0 if "gpu" in path.name.lower() else 1, path.name),
    )
    assert paths, f"No source CSVs found in {directory}."
    return paths


def _aggregate_csv(csv_path: Path) -> list[Row]:
    """Load per-rank walltimes and reduce each MPI size to max walltime."""
    machine = _machine_from_path(csv_path)
    assert machine in MACHINE_FULL_NAMES, f"Unknown machine for {csv_path}."
    ranks_per_node = _ranks_per_node(csv_path, machine)
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

    snakes_per_rank_values = {key[1] for key in grouped}
    assert len(snakes_per_rank_values) == 1, (
        f"Mixed snakes_per_rank in {csv_path}: {sorted(snakes_per_rank_values)}"
    )
    snakes_per_rank = next(iter(snakes_per_rank_values))
    series_label = _label_from_path(csv_path, snakes_per_rank=snakes_per_rank)

    rows: list[Row] = []
    for (mpi_size, snakes_per_rank, n_snakes, steps), walltimes in grouped.items():
        max_walltime = float(np.max(walltimes))
        assert max_walltime > 0.0, f"Non-positive walltime in {csv_path}."
        throughput = n_snakes * steps / max_walltime
        n_nodes = mpi_size / ranks_per_node
        rows.append(
            (
                machine,
                series_label,
                mpi_size,
                n_nodes,
                snakes_per_rank,
                n_snakes,
                max_walltime,
                steps,
                throughput,
            )
        )
    return rows


def combine_csvs(directory: Path = SCRIPT_DIR) -> list[Row]:
    """Load every source CSV into one long-form table."""
    combined: list[Row] = []
    for csv_path in _discover_source_csvs(directory):
        ranks_per_node = _ranks_per_node(csv_path, _machine_from_path(csv_path))
        print(f"loading {csv_path.name} (x = mpi_size / {ranks_per_node} -> nodes)")
        combined.extend(_aggregate_csv(csv_path))
    assert combined, "No benchmark CSVs were loaded."

    machine_rank = {label: index for index, label in enumerate(MACHINE_ORDER)}
    combined.sort(
        key=lambda row: (
            machine_rank[row[0]],
            row[1],
            row[3],
        )
    )
    return combined


def plot_scaling(rows: list[Row], *, output: Path, steps: int) -> None:
    """Plot log-log walltime-per-rod vs node count."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    series: dict[tuple[str, str], list[tuple[float, float, int]]] = defaultdict(list)
    for (
        machine,
        series_label,
        _mpi_size,
        n_nodes,
        snakes_per_rank,
        _n,
        walltime,
        _steps,
        _thr,
    ) in rows:
        series[(machine, series_label)].append((n_nodes, walltime, snakes_per_rank))

    series_labels = sorted({label for _machine, label in series})
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series_colors = {
        label: default_cycle[index % len(default_cycle)]
        for index, label in enumerate(series_labels)
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    walltime_ax = axes[0]
    blank_ax = axes[1]
    blank_ax.set_visible(False)

    for machine in MACHINE_ORDER:
        style = MACHINE_STYLES[machine]
        for series_label in series_labels:
            if "pyelastica" in series_label.lower():
                continue
            points = series.get((machine, series_label))
            if not points:
                continue
            points = sorted(points, key=lambda item: item[0])
            n_nodes = np.asarray([item[0] for item in points], dtype=np.float64)
            walltimes = np.asarray([item[1] for item in points], dtype=np.float64)
            snakes_per_rank = np.asarray([item[2] for item in points], dtype=np.float64)
            walltime_per_rod = walltimes / snakes_per_rank
            color = series_colors[series_label]
            walltime_ax.loglog(
                n_nodes,
                walltime_per_rod,
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=style["markersize"],
                linewidth=style["linewidth"],
                markeredgewidth=1.2,
                markeredgecolor=color,
                markerfacecolor=color if style["filled"] else "none",
            )

    # PyElastica: single first-point marker (not a scaling curve).
    pyelastica_anchor: tuple[float, float] | None = None
    for series_label in series_labels:
        if "pyelastica" not in series_label.lower():
            continue
        if "half" in series_label.lower():
            continue
        for machine in MACHINE_ORDER:
            points = series.get((machine, series_label))
            if not points:
                continue
            ordered = sorted(points, key=lambda item: item[0])
            n_nodes_value, walltime, snakes_per_rank = ordered[0]
            pyelastica_anchor = (n_nodes_value, walltime / snakes_per_rank)
            break
        if pyelastica_anchor is not None:
            break
    if pyelastica_anchor is not None:
        anchor_x, anchor_y = pyelastica_anchor
        walltime_ax.plot(
            [anchor_x],
            [anchor_y],
            linestyle="none",
            marker="*",
            markersize=16,
            color="#DC2626",
            markeredgecolor="#DC2626",
            zorder=6,
        )
        walltime_ax.annotate(
            "PyElastica",
            xy=(anchor_x, anchor_y),
            xytext=(anchor_x * 1.0, anchor_y * 2.5),
            textcoords="data",
            fontsize=12,
            fontweight="semibold",
            color="k",
            ha="left",
            va="bottom",
            arrowprops={
                "arrowstyle": "->",
                "color": "k",
                "lw": 1.6,
                "connectionstyle": "arc3,rad=0.15",
            },
        )

    walltime_ax.set_xlabel("number of nodes")
    walltime_ax.set_ylabel("max walltime (s / rod)")
    walltime_ax.set_title("Weak-scaling walltime per rod")
    walltime_ax.grid(True, which="major", color="#D0D5DD", linewidth=0.9)
    walltime_ax.grid(True, which="minor", color="#EEF0F3", linewidth=0.6)
    walltime_ax.set_axisbelow(True)

    series_handles = [
        Line2D(
            [0],
            [0],
            color=series_colors[label],
            linewidth=2.2,
            label=label,
        )
        for label in series_labels
        if "pyelastica" not in label.lower()
    ]
    if pyelastica_anchor is not None:
        series_handles.append(
            Line2D(
                [0],
                [0],
                color="#DC2626",
                linestyle="none",
                marker="*",
                markersize=12,
                label="PyElastica",
            )
        )
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
            markeredgewidth=1.2,
            label=MACHINE_FULL_NAMES[machine],
        )
        for machine in MACHINE_ORDER
        if any(row[0] == machine for row in rows)
    ]

    series_legend = fig.legend(
        handles=series_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.05, 0.15),
        handlelength=2.0,
        labelspacing=0.35,
        columnspacing=1.0,
        title="Series",
        title_fontsize=10,
        borderaxespad=0.0,
        alignment="left",
        fontsize=7.5,
    )
    fig.add_artist(series_legend)
    fig.legend(
        handles=machine_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.25, 0.15),
        handlelength=2.6,
        labelspacing=0.40,
        title="Machine",
        title_fontsize=10,
        borderaxespad=0.0,
        alignment="left",
    )

    fig.suptitle(
        f"Weak scaling (snake-activated)\n"
        f"({steps} steps; CPU - 64 rods/rank; GPU - 9216 rods/node)",
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
    """Combine all machine CSVs and write a weak-scaling walltime figure."""
    combined = combine_csvs()
    steps = combined[0][7]
    assert all(row[7] == steps for row in combined), "Mixed step counts across CSVs."

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "machine",
                "series_label",
                "mpi_size",
                "n_nodes",
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
