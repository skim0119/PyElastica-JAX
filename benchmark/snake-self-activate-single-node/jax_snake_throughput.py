"""Benchmark multi-snake rollout throughput."""

from __future__ import annotations

import click

from _jax_snake_common import (
    run_jax_rollout,
    run_pyelastica_rollout,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(("pyelastica", "cpu", "cuda"), case_sensitive=False),
    default="cpu",
    show_default=True,
    help="PyElastica (Numba) or JAX rollout backend.",
)
@click.option("--n-snakes-exp", type=int, default=8, show_default=True)
@click.option("--steps", type=int, default=1000, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
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
def main(
    backend: str,
    n_snakes_exp: int,
    steps: int,
    warmup_runs: int,
    no_external_loads: bool,
    transfer_guard: str,
) -> None:
    n_snakes = 2**n_snakes_exp

    match backend:
        case "pyelastica":
            _, rollout_walltime = run_pyelastica_rollout(
                n_snakes=n_snakes,
                steps=steps,
                warmup_runs=warmup_runs,
                include_external_loads=not no_external_loads,
            )
        case "cpu" | "cuda":
            _, rollout_walltime = run_jax_rollout(
                backend=backend,
                n_snakes=n_snakes,
                steps=steps,
                warmup_runs=warmup_runs,
                transfer_guard=transfer_guard,
                include_external_loads=not no_external_loads,
            )
        case _:
            raise AssertionError(f"Unsupported backend {backend!r}.")
    return rollout_walltime


if __name__ == "__main__":
    rollout_walltime = main()
    print(f"rollout_walltime: {rollout_walltime:.6f}")
