"""Sweep single-node two-GPU vertical rod-contact throughput."""

from __future__ import annotations

from pathlib import Path

import click
import jax

from _rod_contact_common import N_ELEMENTS
from _rod_contact_scaling_sweep import run_scaling_benchmark

N_DEVICES = 2


def _backend_available(name: str) -> bool:
    try:
        return len(jax.devices(name)) >= N_DEVICES
    except RuntimeError:
        return False


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cuda",
    show_default=True,
    help="JAX backend; requires at least two devices.",
)
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
    default=Path("output/rod_contact_gpu2x_vertical.png"),
    show_default=True,
)
@click.option(
    "--csv-output",
    type=click.Path(path_type=Path),
    default=None,
    help="CSV path (default: plot path with .csv suffix).",
)
@click.option("--quiet", is_flag=True, help="Disable progress bars.")
def main(
    backend: str,
    min_exp: int,
    max_exp: int,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    output: Path,
    csv_output: Path | None,
    quiet: bool,
) -> None:
    """Sweep rod count on one stacked block sharded across two devices.

    This is the issue-04 vertical two-GPU path (``shard_map``). The horizontal
    two-GPU path (MPI ranks + halo contact) remains blocked on ticket 05.
    """
    backend = backend.lower()
    assert _backend_available(backend), (
        f"Backend {backend!r} needs at least {N_DEVICES} devices for gpu2x."
    )
    assert min_exp >= 1, "gpu2x vertical sweeps require min_exp >= 1 (n_rods >= 2)."

    run_scaling_benchmark(
        backend=backend,
        min_exp=min_exp,
        max_exp=max_exp,
        steps=steps,
        warmup_runs=warmup_runs,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
        vertical=True,
        n_devices=N_DEVICES,
        output_plot=output,
        output_csv=csv_output,
        verbose=not quiet,
    )


if __name__ == "__main__":
    main()
