"""Sweep and plot helpers for rod-rod contact scaling benchmarks."""

from __future__ import annotations

import csv
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from _rod_contact_common import N_ELEMENTS, run_rollout

type SweepPoint = tuple[int, int, float, float]
type RunRolloutFn = Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class ScalingCase:
    """One backend/layout series for CSV export and plotting."""

    backend: str
    vertical: bool
    points: list[SweepPoint]

    @property
    def label(self) -> str:
        return series_label(self.backend, vertical=self.vertical)


def series_label(backend: str, *, vertical: bool) -> str:
    """Return the plot/CSV series label for a backend and layout."""
    if backend == "pyelastica":
        assert not vertical, "PyElastica has no vertical stacked layout."
        return "pyelastica"
    layout = "vertical" if vertical else "horizontal"
    return f"jax-{backend}-{layout}"


def sweep_backend(
    backend: str,
    min_exp: int,
    max_exp: int,
    *,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str = "spatial_hash",
    vertical: bool = False,
    verbose: bool,
    run_rollout_fn: RunRolloutFn = run_rollout,
) -> list[SweepPoint]:
    """Time rollouts for ``n_rods = 2**exp`` across an exponent range."""
    assert min_exp >= 0, "min exponent must be nonnegative."
    assert max_exp >= min_exp, "max exponent must be >= min exponent."

    results: list[SweepPoint] = []
    label = series_label(backend, vertical=vertical)
    for exponent in tqdm(
        range(min_exp, max_exp + 1),
        desc=f"{label} rod-rod contact",
        disable=not verbose,
    ):
        n_rods = 2**exponent
        run_kwargs: dict[str, object] = {
            "n_rods": n_rods,
            "steps": steps,
            "warmup_runs": warmup_runs,
            "n_elements": n_elements,
        }
        rollout_params = inspect.signature(run_rollout_fn).parameters
        if "backend" in rollout_params:
            run_kwargs["backend"] = backend
        if "steps_between_detection" in rollout_params:
            run_kwargs["steps_between_detection"] = steps_between_detection
        if "broad_phase" in rollout_params:
            run_kwargs["broad_phase"] = broad_phase
        if "vertical" in rollout_params:
            run_kwargs["vertical"] = vertical
        instantiate_seconds, rollout_seconds = run_rollout_fn(**run_kwargs)
        print(
            f"{label} n_rods={n_rods} (2^{exponent}): "
            f"instantiate={instantiate_seconds:.3f}s "
            f"rollout={rollout_seconds:.6f}s"
        )
        results.append((exponent, n_rods, instantiate_seconds, rollout_seconds))
    return results


def export_scaling_plot(
    cases: list[ScalingCase],
    *,
    steps: int,
    n_elements: int,
    output: Path,
) -> None:
    """Write a log-log plot of rollout wall time versus rod count."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for case in cases:
        n_rods = np.asarray([point[1] for point in case.points], dtype=np.float64)
        rollout_seconds = np.asarray(
            [point[3] for point in case.points], dtype=np.float64
        )
        ax.loglog(n_rods, rollout_seconds, marker="o", label=case.label)

    ax.set_xlabel("number of rods")
    ax.set_ylabel("rollout wall time (s)")
    ax.set_title(f"Rod-Rod contact scaling ({steps} steps, {n_elements} elements/rod)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote plot: {output}")


def export_scaling_csv(
    cases: list[ScalingCase],
    *,
    steps: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    output: Path,
) -> None:
    """Write sweep results as long-form CSV for later combined plotting."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "backend",
                "vertical",
                "exponent",
                "n_rods",
                "instantiate_s",
                "rollout_walltime_s",
                "steps",
                "n_elements",
                "steps_between_detection",
                "broad_phase",
            )
        )
        for case in cases:
            for exponent, n_rods, instantiate_s, rollout_s in case.points:
                writer.writerow(
                    (
                        case.backend,
                        int(case.vertical),
                        exponent,
                        n_rods,
                        instantiate_s,
                        rollout_s,
                        steps,
                        n_elements,
                        steps_between_detection,
                        broad_phase,
                    )
                )
    print(f"wrote csv: {output}")


def run_scaling_benchmark(
    *,
    backend: str,
    min_exp: int,
    max_exp: int,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str = "spatial_hash",
    vertical: bool = False,
    output_plot: Path,
    output_csv: Path | None,
    verbose: bool,
    run_rollout_fn: RunRolloutFn = run_rollout,
) -> None:
    """Run one backend/layout sweep and export CSV + plot."""
    points = sweep_backend(
        backend,
        min_exp,
        max_exp,
        steps=steps,
        warmup_runs=warmup_runs,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
        vertical=vertical,
        verbose=verbose,
        run_rollout_fn=run_rollout_fn,
    )
    cases = [ScalingCase(backend=backend, vertical=vertical, points=points)]
    csv_path = output_csv if output_csv is not None else output_plot.with_suffix(".csv")
    export_scaling_csv(
        cases,
        steps=steps,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
        output=csv_path,
    )
    export_scaling_plot(
        cases,
        steps=steps,
        n_elements=n_elements,
        output=output_plot,
    )


def scaling_cli(
    backend: str,
    default_plot: str,
    *,
    run_rollout_fn: RunRolloutFn = run_rollout,
    allow_vertical: bool = True,
) -> click.Command:
    """Return a click command configured for one backend."""

    @click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
        help="Broad-phase refresh interval (0 = every step).",
    )
    @click.option(
        "--broad-phase",
        type=click.Choice(["spatial_hash", "all_pairs"]),
        default="spatial_hash",
        show_default=True,
        help="JAX contact broad-phase strategy.",
    )
    @click.option(
        "--vertical",
        is_flag=True,
        help="Use stacked vertical rod memory block (JAX only).",
    )
    @click.option(
        "--output",
        type=click.Path(path_type=Path),
        default=Path(default_plot),
        show_default=True,
        help="Output path for the scaling plot.",
    )
    @click.option(
        "--csv-output",
        type=click.Path(path_type=Path),
        default=None,
        help="CSV path (default: plot path with .csv suffix).",
    )
    @click.option("--quiet", is_flag=True, help="Disable progress bars.")
    def main(
        min_exp: int,
        max_exp: int,
        steps: int,
        warmup_runs: int,
        n_elements: int,
        steps_between_detection: int,
        broad_phase: str,
        vertical: bool,
        output: Path,
        csv_output: Path | None,
        quiet: bool,
    ) -> None:
        """Sweep rod count and plot rollout wall time for rod-rod contact."""
        if vertical:
            assert allow_vertical, f"Backend {backend!r} does not support --vertical."
        run_scaling_benchmark(
            backend=backend,
            min_exp=min_exp,
            max_exp=max_exp,
            steps=steps,
            warmup_runs=warmup_runs,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase=broad_phase,
            vertical=vertical,
            output_plot=output,
            output_csv=csv_output,
            verbose=not quiet,
            run_rollout_fn=run_rollout_fn,
        )

    return main
