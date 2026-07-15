"""Benchmark multi-snake rollout throughput."""

from __future__ import annotations

import click

from _jax_snake_common import (
    run_jax_rollout,
    run_jax_rollout_gpu2x,
    run_pyelastica_rollout,
)


def run(
    backend: str,
    n_snakes_exp: int,
    steps: int,
    warmup_runs: int,
) -> float:
    """
    Run a multi-snake rollout throughput benchmark.

    Parameters
    ----------
    backend : str
        "pyelastica", "cpu", "cuda", or "gpu2x"
    n_snakes_exp : int
        Exponent of number of snakes (n_snakes = 2 ** n_snakes_exp)
    steps : int
        Number of steps in the rollout
    warmup_runs : int
        Number of warmup runs before timed rollout

    Returns
    -------
    rollout_walltime : float
        Rollout walltime in seconds
    """
    n_snakes = 2**n_snakes_exp

    match backend:
        case "pyelastica":
            _, rollout_walltime = run_pyelastica_rollout(
                n_snakes=n_snakes,
                steps=steps,
                warmup_runs=warmup_runs,
            )
        case "cpu" | "cuda":
            _, rollout_walltime = run_jax_rollout(
                backend=backend,
                n_snakes=n_snakes,
                steps=steps,
                warmup_runs=warmup_runs,
            )
        case "gpu2x":
            _, rollout_walltime = run_jax_rollout_gpu2x(
                n_snakes=n_snakes,
                steps=steps,
                warmup_runs=warmup_runs,
            )
        case _:
            raise AssertionError(f"Unsupported backend {backend!r}.")
    return rollout_walltime


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(
        ("pyelastica", "cpu", "cuda", "gpu2x"),
        case_sensitive=False,
    ),
    default="cpu",
    show_default=True,
    help="PyElastica (Numba) or JAX rollout backend.",
)
@click.option("--n-snakes-exp", type=int, default=8, show_default=True)
@click.option("--steps", type=int, default=1000, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
def main(
    backend: str,
    n_snakes_exp: int,
    steps: int,
    warmup_runs: int,
) -> None:
    rollout_walltime = run(
        backend=backend,
        n_snakes_exp=n_snakes_exp,
        steps=steps,
        warmup_runs=warmup_runs,
    )
    print(f"rollout_walltime: {rollout_walltime:.6f}")


if __name__ == "__main__":
    main()
