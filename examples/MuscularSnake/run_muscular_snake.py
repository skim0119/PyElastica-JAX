"""Run the continuum muscular snake case and render its locomotion.

A single Cosserat-rod body is actuated by eight muscle rods (four antagonist
pairs) bound side-by-side and driven by a head-to-tail traveling wave. Sliding
on an anisotropic-friction ground plane, the lateral undulation is rectified
into net forward locomotion (Zhang et al., Nat. Commun. 2019).

Configuration lives in :class:`MuscularSnakeParameters` below; ``--smoke``
shortens the horizon while exercising the full operator stack. Frames are
captured from the packed JAX block at the requested fps and saved to ``data/``
as ``.npz``; with ``--render``, a velocity plot and top-down animation are
written to ``render/``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

DATA_DIR = _EXAMPLE_DIR / "data"
RENDER_DIR = _EXAMPLE_DIR / "render"

from block_utils import (  # noqa: E402
    body_kinematics,
    extract_simulation_rod_positions,
    extract_simulation_rod_velocities,
)
from environment import build_simulation  # noqa: E402
from post_processing import render_outputs, save_simulation_data  # noqa: E402

import elastica_jax as eaj  # noqa: E402


@dataclass(frozen=True, kw_only=True)
class MuscularSnakeParameters:
    """Physical and numerical parameters for the muscular snake case (SI units).

    ``final_time`` is supplied by the CLI (or :meth:`smoke`). The muscle
    connection layout is hardcoded for a 100-element body, so ``n_elem_body``
    is fixed rather than a scaling knob.
    """

    final_time: float

    n_elem_body: int = 100
    base_length_body: float = 1.0
    base_radius_body: float = 0.025
    density_body: float = 1000.0
    youngs_modulus: float = 1.0e7
    poisson_ratio: float = 0.5
    nu: float = 4e-3

    density_muscle: float = 2000.0
    youngs_modulus_muscle: float = 1.0e4
    n_muscle_fibers: int = 8
    n_elem_muscle_group_one: int = 13 * 3
    n_elem_muscle_group_two: int = 33
    base_length_group_one: float = 0.39
    base_length_group_two: float = 0.33
    muscle_radius_tendon: float = 0.003
    muscle_radius_belly: float = 0.006
    muscle_start_connection_index: tuple[int, ...] = (4, 4, 33, 33, 23, 23, 61, 61)
    muscle_force_amplitudes: np.ndarray = field(
        default_factory=lambda: np.array(
            [22.96, 22.96, 20.95, 20.95, 9.51, 9.51, 13.7, 13.7]
        )[::-1]
        / 2.0
    )

    time_step: float = 5e-6
    gravitational_acc: float = -9.81
    period: float = 1.0
    froude: float = 0.1

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / 2.0 * (self.poisson_ratio + 1.0)

    @property
    def shear_modulus_muscle(self) -> float:
        return self.youngs_modulus_muscle / 2.0 * (self.poisson_ratio + 1.0)

    @property
    def nu_body(self) -> float:
        return self.nu / self.density_body / (np.pi * self.base_radius_body**2)

    @property
    def nu_muscle(self) -> float:
        return self.nu / self.density_muscle / (np.pi * self.muscle_radius_tendon**2)

    @property
    def direction(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    @property
    def normal(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    @property
    def friction_mu(self) -> float:
        return self.base_length_body / (
            self.period * self.period * abs(self.gravitational_acc) * self.froude
        )

    @classmethod
    def smoke(cls) -> MuscularSnakeParameters:
        """Return a downscaled copy for a fast smoke test of the full stack."""
        return cls(final_time=0.05)


def _capture_frame_history(
    *,
    times: list[float],
    positions: list[list[np.ndarray]],
    radii: list[np.ndarray],
    center_of_mass: list[np.ndarray],
    avg_velocity: list[np.ndarray],
) -> list[dict[str, list]]:
    """Convert chunked frame buffers into the callback-history layout."""
    n_rods = len(positions[0])
    callback_history: list[dict[str, list]] = [
        {
            "time": list(times),
            "position": [],
            "radius": [radii[rod_idx]] * len(times),
            "center_of_mass": [],
            "avg_velocity": [],
        }
        for rod_idx in range(n_rods)
    ]
    for frame_idx in range(len(times)):
        for rod_idx in range(n_rods):
            callback_history[rod_idx]["position"].append(positions[frame_idx][rod_idx])
        callback_history[0]["center_of_mass"].append(center_of_mass[frame_idx])
        callback_history[0]["avg_velocity"].append(avg_velocity[frame_idx])
    return callback_history


def run_simulation(
    parameters: MuscularSnakeParameters,
    *,
    device: str,
    fps: float,
    output_dir: Path,
    verbose: bool = True,
) -> tuple[list[dict[str, list]], Path]:
    """Integrate the JAX case and stream diagnostics into an ``.npz`` archive."""
    simulator, body_block, muscle_block, rod_list = build_simulation(
        parameters, device=device
    )
    n_rods = len(rod_list)
    body_mass = rod_list[0].mass.copy()
    rod_radii = [rod.radius.copy() for rod in rod_list]

    frame_dt_target = 1.0 / fps
    steps_per_frame = max(1, round(frame_dt_target / parameters.time_step))
    frame_dt = steps_per_frame * parameters.time_step
    n_frames = int(np.ceil(parameters.final_time / frame_dt)) + 1

    times: list[float] = []
    positions: list[list[np.ndarray]] = []
    center_of_mass: list[np.ndarray] = []
    avg_velocity: list[np.ndarray] = []

    stepper = eaj.PositionVerletJAX()
    current_time = 0.0
    start = time.perf_counter()

    if verbose:
        print(f"Backend: {device}")
        print(f"Rods: {n_rods} (1 body + {n_rods - 1} muscles)")
        print(f"Time step: {parameters.time_step:.3e} s, frame_dt: {frame_dt:.3e} s")
        print(f"Final time: {parameters.final_time:.4f} s, frames: {n_frames}")

    for frame_idx in tqdm(
        range(n_frames), desc="Muscular snake rollout", disable=not verbose
    ):
        jax.block_until_ready(body_block)
        jax.block_until_ready(muscle_block)
        rod_positions = extract_simulation_rod_positions(body_block, muscle_block)
        rod_velocities = extract_simulation_rod_velocities(body_block, muscle_block)
        assert np.isfinite(rod_positions[0]).all(), (
            f"Simulation diverged at frame {frame_idx} (t={current_time:.4e} s)."
        )
        com, vel = body_kinematics(rod_positions[0], rod_velocities[0], body_mass)
        times.append(current_time)
        positions.append(rod_positions)
        center_of_mass.append(com)
        avg_velocity.append(vel)

        if frame_idx < n_frames - 1:
            chunk_final_time = min(current_time + frame_dt, parameters.final_time)
            stepper.integrate(
                simulator,
                time=np.float64(current_time),
                final_time=np.float64(chunk_final_time),
                dt=np.float64(parameters.time_step),
            )
            current_time = chunk_final_time

    elapsed = time.perf_counter() - start
    callback_history = _capture_frame_history(
        times=times,
        positions=positions,
        radii=rod_radii,
        center_of_mass=center_of_mass,
        avg_velocity=avg_velocity,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "muscular_snake_data.npz"
    save_simulation_data(
        data_path,
        callback_history=callback_history,
        fps=fps,
        period=parameters.period,
    )
    if verbose:
        print(f"Completed in {elapsed:.2f} s; saved {len(times)} frames to {data_path}")
    return callback_history, data_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-T", "--final-time", type=float, default=16.0, show_default=True)
@click.option(
    "--fps", type=float, default=30.0, show_default=True, help="Frame capture rate."
)
@click.option("--gpu", is_flag=True, help="Run on the CUDA backend (default CPU).")
@click.option(
    "--smoke", is_flag=True, help="Downscaled fast run exercising all features."
)
@click.option("--render", is_flag=True, help="Render velocity plot and mp4 after run.")
def main(final_time: float, fps: float, gpu: bool, smoke: bool, render: bool) -> None:
    """Build, integrate, and save the muscular snake rollout to ``data/``.

    Simulation configuration lives in :class:`MuscularSnakeParameters` at the
    top of this script; the CLI only selects run mode and I/O cadence.
    """
    parameters = (
        MuscularSnakeParameters.smoke()
        if smoke
        else MuscularSnakeParameters(final_time=final_time)
    )
    device = "cuda" if gpu else "cpu"

    callback_history, _data_path = run_simulation(
        parameters, device=device, fps=fps, output_dir=DATA_DIR
    )

    if render:
        render_outputs(
            RENDER_DIR,
            callback_history=callback_history,
            fps=fps,
            period=parameters.period,
        )


if __name__ == "__main__":
    main()
