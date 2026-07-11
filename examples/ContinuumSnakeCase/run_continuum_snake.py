"""Reduced continuum snake GPU validation case.

A JAX-backed reduced prototype of the continuum snake case. It runs the same
reduced problem twice, once on the host with PyElastica and once with the JAX
stepper on CPU or GPU, then reports the final-state differences. It is a
framework-validation example rather than a replacement for the full continuum
snake benchmark.

Run with ``uv run --no-sync python run_continuum_snake_gpu.py`` (add ``--gpu``
for GPU, ``--smoke`` for a short rollout, ``--render`` for a centerline figure).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from environment import build_reference_simulation, build_simulation
from post_processing import (
    animate_snake_gait,
    plot_final_centerlines,
    print_summary,
    save_comparison,
    summarize_results,
)

import elastica as ea
import elastica_jax as eaj

SMOKE_FINAL_TIME = 0.002


@dataclass(frozen=True, kw_only=True)
class SnakeParameters:
    """Physical and numerical parameters for the reduced snake case.

    The final entry of ``b_coeff`` is the actuation wavelength; the remaining
    entries are the spline control points of :class:`elastica.MuscleTorques`.

    ``n_elem``, ``time_step``, and ``final_time`` are CLI-driven and therefore
    required here; their defaults live with the ``click`` options in ``main``.
    """

    n_elem: int
    time_step: float
    final_time: float
    base_length: float = 0.35
    density: float = 1000.0
    youngs_modulus: float = 1.0e6
    poisson_ratio: float = 0.5
    period: float = 2.0
    gravitational_acc: float = -9.80665
    slip_velocity_tol: float = 1.0e-8
    froude: float = 0.1
    contact_k: float = 1.0
    contact_nu: float = 1.0e-6
    damping_constant: float = 2.0e-3
    b_coeff: tuple[float, ...] = (
        3.4e-3,
        3.3e-3,
        4.2e-3,
        2.6e-3,
        3.6e-3,
        3.5e-3,
        1.0,
    )

    @property
    def base_radius(self) -> float:
        return self.base_length * 0.011

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (self.poisson_ratio + 1.0)

    @property
    def wave_number(self) -> float:
        return 2.0 * np.pi / float(self.b_coeff[-1])

    @property
    def b_coeff_array(self) -> np.ndarray:
        return self.b_coeff

    @property
    def plane_origin(self) -> np.ndarray:
        return np.array([0.0, -self.base_radius, 0.0], dtype=np.float64)

    @property
    def plane_normal(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    @property
    def kinetic_mu_array(self) -> np.ndarray:
        mu = self.base_length / (
            self.period * self.period * abs(self.gravitational_acc) * self.froude
        )
        return np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)

    @property
    def static_mu_array(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)


def run_reference(parameters: SnakeParameters) -> tuple[dict[str, np.ndarray], float]:
    """Run the PyElastica reference rollout and return its final state."""
    simulator, rod = build_reference_simulation(parameters)
    stepper = ea.PositionVerlet()
    time_value = np.float64(0.0)
    dt = np.float64(parameters.time_step)
    total_steps = int(parameters.final_time / parameters.time_step)

    start = time.perf_counter()
    for _ in range(total_steps):
        time_value = stepper.step(simulator, time_value, dt)
    elapsed = time.perf_counter() - start

    state = {
        "position_collection": rod.position_collection.copy(),
        "director_collection": rod.director_collection.copy(),
        "velocity_collection": rod.velocity_collection.copy(),
        "omega_collection": rod.omega_collection.copy(),
        "internal_forces": rod.internal_forces.copy(),
        "internal_torques": rod.internal_torques.copy(),
        "sigma": rod.sigma.copy(),
        "kappa": rod.kappa.copy(),
    }
    return state, elapsed


def run_jax(
    parameters: SnakeParameters,
    *,
    device: jax.Device,
    fps: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Run the JAX rollout, capturing the centerline trajectory for rendering.

    A warm-up integration is executed first so the timed chunk-integrate loop
    measures steady-state device throughput rather than JIT compilation. Frames
    are captured at ``fps`` by integrating in fixed-time chunks.

    Returns
    -------
    tuple[dict, dict, float]
        The final host-side state, the trajectory (``time`` and ``position``
        arrays), and the elapsed wall-clock time of the timed rollout.
    """
    stepper = eaj.PositionVerletJAX()
    simulator, block = build_simulation(parameters, device=device)
    initial_state = dict(block.jax_get_state())

    stepper.integrate(
        simulator,
        time=np.float64(0.0),
        final_time=np.float64(parameters.final_time),
        dt=np.float64(parameters.time_step),
    )
    jax.block_until_ready(block.position_collection_device)
    block.jax_set_state(dict(initial_state))

    steps_per_frame = max(1, round((1.0 / fps) / parameters.time_step))
    frame_dt = steps_per_frame * parameters.time_step
    n_frames = max(1, round(parameters.final_time / frame_dt))

    times: list[float] = []
    positions: list[np.ndarray] = []

    def capture(current_time: float) -> None:
        jax.block_until_ready(block.position_collection_device)
        times.append(current_time)
        positions.append(np.asarray(block.position_collection_device).copy())

    start = time.perf_counter()
    current_time = 0.0
    capture(current_time)
    for _ in range(n_frames):
        current_time = float(
            stepper.integrate(
                simulator,
                time=np.float64(current_time),
                final_time=np.float64(current_time + frame_dt),
                dt=np.float64(parameters.time_step),
            )
        )
        capture(current_time)
    elapsed = time.perf_counter() - start

    state = jax.tree_util.tree_map(np.asarray, block.jax_get_state())
    trajectory = {
        "time": np.asarray(times),
        "position": np.asarray(positions),
    }
    return state, trajectory, elapsed


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(("cpu", "gpu"), case_sensitive=False),
    default="cpu",
    show_default=True,
    help="JAX backend for the rollout.",
)
@click.option(
    "--smoke",
    is_flag=True,
    help="Run a short rollout for a quick end-to-end check.",
)
@click.option("--n-elem", type=int, default=50, show_default=True)
@click.option("--final-time", type=float, default=30.0, show_default=True)
@click.option("--time-step", type=float, default=1.0e-4, show_default=True)
@click.option("--fps", type=float, default=30.0, show_default=True)
@click.option(
    "--render",
    is_flag=True,
    help="Save the comparison figure, gait animation, and data archive.",
)
def main(
    backend: str,
    smoke: bool,
    n_elem: int,
    final_time: float,
    time_step: float,
    fps: float,
    render: bool,
) -> None:
    """Run the reduced snake case on CPU and JAX and report state differences."""
    if smoke:
        final_time = 0.002
    parameters = SnakeParameters(
        n_elem=n_elem, final_time=final_time, time_step=time_step
    )

    device = eaj.resolve_backend_devices(backend)[0]
    total_steps = int(parameters.final_time / parameters.time_step)
    print(f"JAX device: {device} (platform={device.platform})")
    print(f"Reduced snake rollout steps: {total_steps}")

    cpu_state, cpu_elapsed = run_reference(parameters)
    print(f"CPU reference elapsed: {cpu_elapsed:.4f} s")

    jax_state, trajectory, jax_elapsed = run_jax(parameters, device=device, fps=fps)
    print(f"JAX rollout elapsed: {jax_elapsed:.4f} s")

    diffs = summarize_results(cpu_state, jax_state)
    print_summary(diffs)

    if render:
        save_comparison(
            _EXAMPLE_DIR / "snake_comparison.npz",
            cpu_state=cpu_state,
            jax_state=jax_state,
            diffs=diffs,
            trajectory=trajectory,
        )
        plot_final_centerlines(
            cpu_state["position_collection"],
            jax_state["position_collection"],
            filename=_EXAMPLE_DIR / "final_centerlines.png",
        )
        animate_snake_gait(
            trajectory,
            video_path=_EXAMPLE_DIR / "render" / "snake_gait.mp4",
            fps=int(round(fps)),
        )


if __name__ == "__main__":
    main()
