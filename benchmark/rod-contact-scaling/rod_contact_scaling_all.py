"""Unified single-node scaling sweep for PyElastica and JAX layouts."""

from __future__ import annotations

from pathlib import Path

import click
import jax

from _rod_contact_common import N_ELEMENTS, run_rollout, run_rollout_pyelastica
from _rod_contact_scaling_sweep import (
    RunRolloutFn,
    ScalingCase,
    export_scaling_csv,
    export_scaling_plot,
    sweep_backend,
)


def _backend_available(name: str) -> bool:
    try:
        return len(jax.devices(name)) > 0
    except RuntimeError:
        return False


def _append_case(
    cases: list[ScalingCase],
    *,
    backend: str,
    vertical: bool,
    min_exp: int,
    max_exp: int,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    verbose: bool,
    run_rollout_fn: RunRolloutFn,
) -> None:
    points = sweep_backend(
        backend=backend,
        min_exp=min_exp,
        max_exp=max_exp,
        steps=steps,
        warmup_runs=warmup_runs,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
        vertical=vertical,
        verbose=verbose,
        run_rollout_fn=run_rollout_fn,
    )
    cases.append(ScalingCase(backend=backend, vertical=vertical, points=points))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--min-exp", type=int, default=1, show_default=True)
@click.option("--max-exp", type=int, default=8, show_default=True)
@click.option(
    "--pyelastica-min-exp",
    type=int,
    default=None,
    help="Override minimum exponent for PyElastica sweep.",
)
@click.option(
    "--pyelastica-max-exp",
    type=int,
    default=None,
    help="Override maximum exponent for PyElastica sweep.",
)
@click.option(
    "--cpu-min-exp",
    type=int,
    default=None,
    help="Override minimum exponent for JAX CPU sweeps.",
)
@click.option(
    "--cpu-max-exp",
    type=int,
    default=None,
    help="Override maximum exponent for JAX CPU sweeps.",
)
@click.option(
    "--gpu-min-exp",
    type=int,
    default=None,
    help="Override minimum exponent for JAX CUDA sweeps.",
)
@click.option(
    "--gpu-max-exp",
    type=int,
    default=None,
    help="Override maximum exponent for JAX CUDA sweeps.",
)
@click.option("--steps", type=int, default=200, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
@click.option("--n-elements", type=int, default=N_ELEMENTS, show_default=True)
@click.option(
    "--steps-between-detection",
    type=int,
    default=0,
    show_default=True,
    help="Broad-phase refresh interval for JAX contact (0 = every step).",
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
    default=Path("output/rod_contact_scaling_all.png"),
    show_default=True,
    help="Output path for the combined scaling plot.",
)
@click.option(
    "--csv-output",
    type=click.Path(path_type=Path),
    default=None,
    help="CSV path (default: plot path with .csv suffix).",
)
@click.option("--skip-pyelastica", is_flag=True, help="Skip the PyElastica sweep.")
@click.option("--skip-cpu", is_flag=True, help="Skip JAX CPU horizontal/vertical.")
@click.option("--skip-gpu", is_flag=True, help="Skip JAX CUDA horizontal/vertical.")
@click.option(
    "--skip-vertical",
    is_flag=True,
    help="Skip stacked vertical JAX layouts.",
)
@click.option(
    "--skip-horizontal",
    is_flag=True,
    help="Skip packed horizontal JAX layouts.",
)
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    min_exp: int,
    max_exp: int,
    pyelastica_min_exp: int | None,
    pyelastica_max_exp: int | None,
    cpu_min_exp: int | None,
    cpu_max_exp: int | None,
    gpu_min_exp: int | None,
    gpu_max_exp: int | None,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    output: Path,
    csv_output: Path | None,
    skip_pyelastica: bool,
    skip_cpu: bool,
    skip_gpu: bool,
    skip_vertical: bool,
    skip_horizontal: bool,
    quiet: bool,
) -> None:
    """Sweep rod count across single-node backends and layouts."""
    assert not (skip_pyelastica and skip_cpu and skip_gpu), (
        "At least one backend must be enabled."
    )
    jax_enabled = (not skip_cpu) or (not skip_gpu)
    assert not (skip_horizontal and skip_vertical and jax_enabled), (
        "At least one JAX layout must be enabled when JAX backends run."
    )

    cases: list[ScalingCase] = []
    verbose = not quiet
    pyelastica_min = min_exp if pyelastica_min_exp is None else pyelastica_min_exp
    pyelastica_max = max_exp if pyelastica_max_exp is None else pyelastica_max_exp
    cpu_min = min_exp if cpu_min_exp is None else cpu_min_exp
    cpu_max = max_exp if cpu_max_exp is None else cpu_max_exp
    gpu_min = min_exp if gpu_min_exp is None else gpu_min_exp
    gpu_max = max_exp if gpu_max_exp is None else gpu_max_exp
    jax_layouts = []
    if not skip_horizontal:
        jax_layouts.append(False)
    if not skip_vertical:
        jax_layouts.append(True)

    if not skip_pyelastica:
        _append_case(
            cases,
            backend="pyelastica",
            vertical=False,
            min_exp=pyelastica_min,
            max_exp=pyelastica_max,
            steps=steps,
            warmup_runs=warmup_runs,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase=broad_phase,
            verbose=verbose,
            run_rollout_fn=run_rollout_pyelastica,
        )

    if not skip_cpu:
        for vertical in jax_layouts:
            _append_case(
                cases,
                backend="cpu",
                vertical=vertical,
                min_exp=cpu_min,
                max_exp=cpu_max,
                steps=steps,
                warmup_runs=warmup_runs,
                n_elements=n_elements,
                steps_between_detection=steps_between_detection,
                broad_phase=broad_phase,
                verbose=verbose,
                run_rollout_fn=run_rollout,
            )

    if not skip_gpu:
        if not _backend_available("cuda"):
            print("cuda unavailable: skipping jax-cuda sweeps")
        else:
            for vertical in jax_layouts:
                _append_case(
                    cases,
                    backend="cuda",
                    vertical=vertical,
                    min_exp=gpu_min,
                    max_exp=gpu_max,
                    steps=steps,
                    warmup_runs=warmup_runs,
                    n_elements=n_elements,
                    steps_between_detection=steps_between_detection,
                    broad_phase=broad_phase,
                    verbose=verbose,
                    run_rollout_fn=run_rollout,
                )

    assert cases, "No scaling cases were collected."
    csv_path = csv_output if csv_output is not None else output.with_suffix(".csv")
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
        output=output,
    )


if __name__ == "__main__":
    main()
