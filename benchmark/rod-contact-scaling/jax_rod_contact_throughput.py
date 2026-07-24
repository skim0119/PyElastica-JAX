"""Single-case rod-rod contact throughput worker (snake-style)."""

from __future__ import annotations

from dataclasses import dataclass

import click

from _rod_contact_common import (
    N_ELEMENTS,
    STEPS_BETWEEN_DETECTION,
    run_rollout,
    run_rollout_pyelastica,
)


@dataclass(frozen=True, kw_only=True)
class ThroughputConfig:
    """Fixed knobs for one rod-contact throughput sample."""

    n_rods_exp: int
    steps: int
    warmup_runs: int
    n_elements: int = N_ELEMENTS
    steps_between_detection: int = STEPS_BETWEEN_DETECTION
    broad_phase: str = "spatial_hash"
    vertical: bool = False

    @property
    def n_rods(self) -> int:
        """Return ``2 ** n_rods_exp``."""
        assert self.n_rods_exp >= 0, "n_rods_exp must be nonnegative."
        return 2**self.n_rods_exp


def run(*, backend: str, config: ThroughputConfig) -> tuple[float, float]:
    """Time one rod-contact rollout for a backend and layout.

    Parameters
    ----------
    backend :
        ``"pyelastica"``, ``"cpu"``, or ``"cuda"``.
    config :
        Rod count exponent, timing, layout, and contact knobs.

    Returns
    -------
    tuple[float, float]
        Instantiation seconds and rollout wall time seconds.
    """
    n_rods = config.n_rods
    match backend:
        case "pyelastica":
            assert not config.vertical, "PyElastica does not support vertical layout."
            return run_rollout_pyelastica(
                n_rods=n_rods,
                steps=config.steps,
                warmup_runs=config.warmup_runs,
                n_elements=config.n_elements,
            )
        case "cpu" | "cuda":
            return run_rollout(
                backend=backend,
                n_rods=n_rods,
                steps=config.steps,
                warmup_runs=config.warmup_runs,
                n_elements=config.n_elements,
                steps_between_detection=config.steps_between_detection,
                broad_phase=config.broad_phase,
                vertical=config.vertical,
            )
        case _:
            assert False, f"Unsupported backend {backend!r}."


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(("pyelastica", "cpu", "cuda"), case_sensitive=False),
    default="cpu",
    show_default=True,
    help="PyElastica or JAX rollout backend.",
)
@click.option("--n-rods-exp", type=int, default=4, show_default=True)
@click.option("--steps", type=int, default=200, show_default=True)
@click.option("--warmup-runs", type=int, default=1, show_default=True)
@click.option("--n-elements", type=int, default=N_ELEMENTS, show_default=True)
@click.option(
    "--steps-between-detection",
    type=int,
    default=STEPS_BETWEEN_DETECTION,
    show_default=True,
)
@click.option(
    "--broad-phase",
    type=click.Choice(["spatial_hash", "all_pairs"]),
    default="spatial_hash",
    show_default=True,
)
@click.option(
    "--vertical",
    is_flag=True,
    help="Use stacked vertical rod memory block (JAX only).",
)
def main(
    backend: str,
    n_rods_exp: int,
    steps: int,
    warmup_runs: int,
    n_elements: int,
    steps_between_detection: int,
    broad_phase: str,
    vertical: bool,
) -> None:
    """Run one rod-contact throughput sample and print timings."""
    config = ThroughputConfig(
        n_rods_exp=n_rods_exp,
        steps=steps,
        warmup_runs=warmup_runs,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
        broad_phase=broad_phase,
        vertical=vertical,
    )
    instantiate_seconds, rollout_seconds = run(
        backend=backend.lower(),
        config=config,
    )
    print(f"backend={backend.lower()}")
    print(f"vertical={int(vertical)}")
    print(f"n_rods={config.n_rods}")
    print(f"instantiate_s={instantiate_seconds:.6f}")
    print(f"rollout_walltime={rollout_seconds:.6f}")


if __name__ == "__main__":
    main()
