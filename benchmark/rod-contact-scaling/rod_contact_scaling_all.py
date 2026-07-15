"""Unified scaling sweep for PyElastica, JAX CPU, and JAX CUDA."""

from __future__ import annotations

from pathlib import Path

import click
import jax

from _rod_contact_scaling_sweep import (
    export_scaling_csv,
    export_scaling_plot,
    sweep_backend,
)
from _rod_contact_common import (
    N_ELEMENTS,
    run_rollout,
    run_rollout_pairwise,
    run_rollout_pyelastica,
)


def _backend_available(name: str) -> bool:
    try:
        return len(jax.devices(name)) > 0
    except Exception:
        return False


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
    help="Override minimum exponent for JAX CPU sweep.",
)
@click.option(
    "--cpu-max-exp",
    type=int,
    default=None,
    help="Override maximum exponent for JAX CPU sweep.",
)
@click.option(
    "--gpu-min-exp",
    type=int,
    default=None,
    help="Override minimum exponent for JAX CUDA sweep.",
)
@click.option(
    "--gpu-max-exp",
    type=int,
    default=None,
    help="Override maximum exponent for JAX CUDA sweep.",
)
@click.option(
    "--cpu-old-min-exp",
    type=int,
    default=None,
    help="Override minimum exponent for JAX CPU pairwise RodRodContact sweep.",
)
@click.option(
    "--cpu-old-max-exp",
    type=int,
    default=None,
    help="Override maximum exponent for JAX CPU pairwise RodRodContact sweep.",
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
@click.option("--skip-cpu", is_flag=True, help="Skip the JAX CPU sweep.")
@click.option(
    "--skip-cpu-old",
    is_flag=True,
    help="Skip the JAX CPU pairwise RodRodContact (PyElastica-style) sweep.",
)
@click.option("--skip-gpu", is_flag=True, help="Skip the JAX CUDA sweep.")
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    min_exp: int,
    max_exp: int,
    pyelastica_min_exp: int | None,
    pyelastica_max_exp: int | None,
    cpu_min_exp: int | None,
    cpu_max_exp: int | None,
    cpu_old_min_exp: int | None,
    cpu_old_max_exp: int | None,
    gpu_min_exp: int | None,
    gpu_max_exp: int | None,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    output: Path,
    csv_output: Path | None,
    skip_pyelastica: bool,
    skip_cpu: bool,
    skip_cpu_old: bool,
    skip_gpu: bool,
    quiet: bool,
) -> None:
    """Sweep rod count and plot all backends on one figure."""
    assert not (skip_pyelastica and skip_cpu and skip_cpu_old and skip_gpu), (
        "At least one backend must be enabled."
    )

    series: dict[str, list[tuple[int, int, float, float]]] = {}
    verbose = not quiet
    pyelastica_min = min_exp if pyelastica_min_exp is None else pyelastica_min_exp
    pyelastica_max = max_exp if pyelastica_max_exp is None else pyelastica_max_exp
    cpu_min = min_exp if cpu_min_exp is None else cpu_min_exp
    cpu_max = max_exp if cpu_max_exp is None else cpu_max_exp
    cpu_old_min = min_exp if cpu_old_min_exp is None else cpu_old_min_exp
    cpu_old_max = max_exp if cpu_old_max_exp is None else cpu_old_max_exp
    gpu_min = min_exp if gpu_min_exp is None else gpu_min_exp
    gpu_max = max_exp if gpu_max_exp is None else gpu_max_exp

    if not skip_pyelastica:
        series["pyelastica"] = sweep_backend(
            backend="pyelastica",
            min_exp=pyelastica_min,
            max_exp=pyelastica_max,
            steps=steps,
            warmup_runs=warmup_runs,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            verbose=verbose,
            run_rollout_fn=run_rollout_pyelastica,
        )

    if not skip_cpu:
        series["jax-cpu"] = sweep_backend(
            backend="cpu",
            min_exp=cpu_min,
            max_exp=cpu_max,
            steps=steps,
            warmup_runs=warmup_runs,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            broad_phase="spatial_hash",
            verbose=verbose,
            run_rollout_fn=run_rollout,
        )

    if not skip_cpu_old:
        series["jax-cpu-old"] = sweep_backend(
            backend="cpu",
            min_exp=cpu_old_min,
            max_exp=cpu_old_max,
            steps=steps,
            warmup_runs=warmup_runs,
            n_elements=n_elements,
            steps_between_detection=steps_between_detection,
            verbose=verbose,
            run_rollout_fn=run_rollout_pairwise,
        )

    if not skip_gpu:
        if not _backend_available("cuda"):
            print("cuda unavailable: skipping jax-cuda sweep")
        else:
            series["jax-cuda"] = sweep_backend(
                backend="cuda",
                min_exp=gpu_min,
                max_exp=gpu_max,
                steps=steps,
                warmup_runs=warmup_runs,
                n_elements=n_elements,
                steps_between_detection=steps_between_detection,
                broad_phase="spatial_hash",
                verbose=verbose,
                run_rollout_fn=run_rollout,
            )

    csv_path = csv_output if csv_output is not None else output.with_suffix(".csv")
    export_scaling_csv(
        series,
        steps=steps,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        output=csv_path,
    )
    export_scaling_plot(
        series,
        steps=steps,
        n_elements=n_elements,
        output=output,
    )


if __name__ == "__main__":
    main()
