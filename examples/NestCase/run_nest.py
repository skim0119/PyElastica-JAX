"""Run the JAX nest packing case.

Randomly oriented rods settle under gravity inside a cylinder, interacting
through Hertzian rod-rod contact and a Hertzian ground/wall substrate. This
reproduces the C++ ``Nest`` case in its native mm-g-s unit system, differing
only in dissipation (analytical linear damper instead of the C++ force-based
viscous damping).

Configuration lives in :class:`NestParameters` below; ``--smoke`` downscales the
rod count and shortens the run while still exercising every feature.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click
import jax
import numpy as np
from jax import config as jax_config
from tqdm import tqdm

jax_config.update("jax_enable_x64", True)

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

OUTPUT_PATH = _EXAMPLE_DIR / "data" / "nest_diagnostics.npz"

from environment import build_simulation  # noqa: E402
import elastica_jax as eaj  # noqa: E402


@dataclass(frozen=True, kw_only=True)
class NestParameters:
    """Physical and numerical parameters for the nest case (mm-g-s units).

    Defaults reproduce the full-scale C++ ``Nest`` case. Contact stiffness,
    damping, friction coefficients, and time step are copied verbatim from the
    C++ source; ``damping_constant`` replaces the C++ viscous ``nu`` with the
    equivalent analytical decay rate ``nu / (density * area)``.
    """

    num_rods: int
    final_time: float

    n_elements: int = 8
    rod_length: float = 75.74
    rod_radius: float = 1.2075
    density: float = 8.94e-3
    youngs_modulus: float = 12e9
    poisson_ratio: float = 0.305

    gravity: float = 981.0

    cylinder_radius: float = 69.85
    height_range: float = 30.0
    time_step: float = 8.8e-7

    capture_interval: float = 0.02
    contact_stiffness: float = 1.0e6
    contact_damping: float = 1.0e3
    contact_stiffness_soft: float = 1500.0
    contact_damping_soft: float = 100.0

    stiffness_ramp_time: float = 0.2
    contact_friction: float = 0.4
    contact_static_velocity: float = 1.0e-6
    plane_stiffness: float = 1.0e6
    plane_damping: float = 1.0e3
    plane_friction_kinetic: float = 0.2
    plane_static_velocity: float = 1.0e-6
    friction_start_time: float = 0.5

    damping_constant: float = 5.0

    steps_between_detection: int = 1000

    random_seed: int = 42

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (self.poisson_ratio + 1.0)

    @classmethod
    def smoke(cls) -> NestParameters:
        """Return a downscaled copy for a fast smoke test of the full stack.

        Uses few rods and a short horizon while keeping every operator
        registered with its real, physically-tuned coefficients. Like the C++
        case, the soft-to-hard contact ramp (``t=0.2 s``) and friction
        (``t=0.5 s``) only activate once the pack has calmed, so a short smoke
        run exercises the settling regime without the numerical blow-up that
        hard contact on raw initial overlaps would cause.
        """
        return cls(
            num_rods=8,
            final_time=0.01,
            capture_interval=0.001,
            steps_between_detection=200,
        )


def sample_diagnostics(
    block: eaj._CosseratRodMemoryBlock,
    *,
    gravity: float,
) -> tuple[float, float, float]:
    """Return ``(max_z, min_z, total_energy)`` from the current block state."""
    block.from_device(
        variables=("position_collection", "velocity_collection"),
        update_rods=False,
    )
    positions = block.position_collection
    velocities = block.velocity_collection
    mass = block.mass

    max_z = float(np.max(positions[2, :]))
    min_z = float(np.min(positions[2, :]))
    kinetic = 0.5 * float(np.sum(mass * np.sum(velocities**2, axis=0)))
    potential = gravity * float(np.sum(mass * positions[2, :]))
    return max_z, min_z, kinetic + potential


def run_simulation(
    parameters: NestParameters,
    *,
    device: str,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Build the simulator, integrate, and return recorded diagnostics.

    Parameters
    ----------
    parameters : NestParameters
        Case configuration.
    device : str
        ``"cpu"`` or ``"cuda"``.
    verbose : bool, optional
        Show a progress bar and status prints.

    Returns
    -------
    dict[str, numpy.ndarray]
        Diagnostic history keyed by ``time``, ``nest_height_max``,
        ``nest_height_min``, ``total_energy``, and ``rod_positions``.

    Raises
    ------
    FloatingPointError
        If the state becomes non-finite (the simulation has blown up).
    """
    simulator, block = build_simulation(parameters, device=device)
    stepper = eaj.PositionVerletJAX()

    steps_per_capture = max(
        1, round(parameters.capture_interval / parameters.time_step)
    )
    capture_interval = steps_per_capture * parameters.time_step
    n_captures = max(1, round(parameters.final_time / capture_interval))

    if verbose:
        print(f"Backend: {device}")
        print(f"Rods: {parameters.num_rods}, elements/rod: {parameters.n_elements}")
        print(f"Time step: {parameters.time_step:.3e} s")
        print(f"Final time: {n_captures * capture_interval:.4f} s")
        print(f"Steps: {n_captures * steps_per_capture} ({steps_per_capture}/capture)")

    history: dict[str, list] = {
        "time": [],
        "nest_height_max": [],
        "nest_height_min": [],
        "total_energy": [],
        "rod_positions": [],
    }

    def record(current_time: float) -> None:
        jax.block_until_ready(block)

        max_z, min_z, energy = sample_diagnostics(block, gravity=parameters.gravity)
        assert np.isfinite(max_z) and np.isfinite(energy), (
            f"Simulation diverged at t={current_time:.4e} s "
            f"(max_z={max_z}, energy={energy})."
        )
        history["time"].append(current_time)
        history["nest_height_max"].append(max_z)
        history["nest_height_min"].append(min_z)
        history["total_energy"].append(energy)
        rod_positions = [
            np.asarray(rod.position_collection).copy() for rod in block.iterate_rods()
        ]
        history["rod_positions"].append(rod_positions)

    start = time.perf_counter()
    current_time = 0.0
    record(current_time)
    for _ in tqdm(range(n_captures), desc="Nest rollout", disable=not verbose):
        current_time = stepper.integrate(
            simulator,
            time=current_time,
            final_time=current_time + capture_interval,
            dt=parameters.time_step,
        )
        record(current_time)
    elapsed = time.perf_counter() - start

    if verbose:
        print(f"Completed in {elapsed:.2f} s; captured {len(history['time'])} frames.")

    return {
        "time": np.asarray(history["time"]),
        "nest_height_max": np.asarray(history["nest_height_max"]),
        "nest_height_min": np.asarray(history["nest_height_min"]),
        "total_energy": np.asarray(history["total_energy"]),
        "rod_positions": np.asarray(history["rod_positions"], dtype=object),
        "cylinder_radius": np.asarray(parameters.cylinder_radius),
    }


def save_diagnostics(data: dict[str, np.ndarray], output_path: Path) -> None:
    """Persist recorded diagnostics to an ``.npz`` archive under ``data/``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-N", "--num-rods", type=int, default=455, show_default=True)
@click.option("-T", "--final-time", type=float, default=50.0, show_default=True)
@click.option("--gpu", is_flag=True, help="Run on the CUDA backend (default CPU).")
@click.option(
    "--smoke", is_flag=True, help="Downscaled fast run exercising all features."
)
@click.option("--render", is_flag=True, help="Render plots and mp4 after the run.")
def main(
    num_rods: int, final_time: float, gpu: bool, smoke: bool, render: bool
) -> None:
    """Build, integrate, and save the JAX nest packing case.

    Simulation configuration lives in :class:`NestParameters` at the top of this
    script; edit it (or its defaults) to change scale or physics. The CLI only
    selects run mode.
    """
    parameters = (
        NestParameters.smoke()
        if smoke
        else NestParameters(num_rods=num_rods, final_time=final_time)
    )
    device = "cuda" if gpu else "cpu"

    data = run_simulation(parameters, device=device)
    save_diagnostics(data, OUTPUT_PATH)
    print(f"Saved diagnostics to {OUTPUT_PATH}")

    if render:
        from post_processing import render_outputs

        render_outputs(data, output_dir=_EXAMPLE_DIR / "render")


if __name__ == "__main__":
    main()
