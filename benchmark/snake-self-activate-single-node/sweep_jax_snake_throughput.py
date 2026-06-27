"""Sweep multi-snake rollout throughput and export a scaling plot."""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from jax_snake_throughput import run

type SweepPoint = tuple[int, int, float]


def _sweep_backend(
    backend: str,
    min_exp: int,
    max_exp: int,
    *,
    steps: int,
    warmup_runs: int,
    no_external_loads: bool,
    transfer_guard: str,
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
            no_external_loads=no_external_loads,
            transfer_guard=transfer_guard,
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
    ax.set_title(f"Multi-snake rollout scaling ({steps} steps)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote plot: {output}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--steps", type=int, default=1000, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
@click.option("--cuda-min-exp", type=int, default=6, show_default=True)
@click.option("--cuda-max-exp", type=int, default=14, show_default=True)
@click.option("--cpu-min-exp", type=int, default=6, show_default=True)
@click.option("--cpu-max-exp", type=int, default=12, show_default=True)
@click.option("--pyelastica-min-exp", type=int, default=6, show_default=True)
@click.option("--pyelastica-max-exp", type=int, default=12, show_default=True)
@click.option("--no-external-loads", is_flag=True)
@click.option(
    "--transfer-guard",
    type=click.Choice(
        ("allow", "log", "disallow", "log_explicit", "disallow_explicit"),
        case_sensitive=False,
    ),
    default="allow",
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("snake_throughput_scaling.png"),
    show_default=True,
    help="Output path for the scaling plot.",
)
@click.option("--skip-cuda", is_flag=True, help="Skip the CUDA sweep.")
@click.option("--skip-cpu", is_flag=True, help="Skip the JAX CPU sweep.")
@click.option("--skip-pyelastica", is_flag=True, help="Skip the PyElastica sweep.")
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    steps: int,
    warmup_runs: int,
    cuda_min_exp: int,
    cuda_max_exp: int,
    cpu_min_exp: int,
    cpu_max_exp: int,
    pyelastica_min_exp: int,
    pyelastica_max_exp: int,
    no_external_loads: bool,
    transfer_guard: str,
    output: Path,
    skip_cuda: bool,
    skip_cpu: bool,
    skip_pyelastica: bool,
    quiet: bool,
) -> None:
    assert steps > 0, "steps must be positive."
    assert not (skip_cuda and skip_cpu and skip_pyelastica), (
        "At least one backend sweep is required."
    )

    sweep_kwargs = {
        "steps": steps,
        "warmup_runs": warmup_runs,
        "no_external_loads": no_external_loads,
        "transfer_guard": transfer_guard,
        "verbose": not quiet,
    }
    series: dict[str, list[SweepPoint]] = {}

    if not skip_cuda:
        series["cuda"] = _sweep_backend(
            "cuda",
            cuda_min_exp,
            cuda_max_exp,
            **sweep_kwargs,
        )
    if not skip_cpu:
        series["jax-cpu"] = _sweep_backend(
            "cpu",
            cpu_min_exp,
            cpu_max_exp,
            **sweep_kwargs,
        )
    if not skip_pyelastica:
        series["pyelastica"] = _sweep_backend(
            "pyelastica",
            pyelastica_min_exp,
            pyelastica_max_exp,
            **sweep_kwargs,
        )

    _export_scaling_plot(series, steps=steps, output=output)


if __name__ == "__main__":
    main()
