"""Operators and helpers for the JAX butterfly free-rod example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import elastica as ea
import elastica_jax as eaj
from elastica.utils import MaxDimension


@dataclass(frozen=True)
class ButterflyParameters:
    """Physical and numerical parameters for the butterfly case."""

    n_elem: int = 4
    total_length: float = 3.0
    base_radius: float = 0.25
    density: float = 5000.0
    youngs_modulus: float = 1.0e4
    poisson_ratio: float = 0.5
    angle_of_inclination: float = np.deg2rad(45.0)
    final_time: float = 40.0

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (self.poisson_ratio + 1.0)

    @property
    def time_step(self) -> float:
        n_elem = self.n_elem + self.n_elem % 2
        return 0.01 * (self.total_length / n_elem)


def build_butterfly_positions(
    *,
    n_elem: int,
    total_length: float,
    angle_of_inclination: float,
) -> np.ndarray:
    """Return initial node positions for the V-shaped butterfly rod."""
    half_n_elem = n_elem // 2
    origin = np.zeros((3, 1))
    horizontal_direction = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)
    vertical_direction = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)

    positions = np.empty((MaxDimension.value(), n_elem + 1))
    dl = total_length / n_elem
    first_half = np.arange(half_n_elem + 1.0).reshape(1, -1)
    positions[..., : half_n_elem + 1] = origin + dl * first_half * (
        np.cos(angle_of_inclination) * horizontal_direction
        + np.sin(angle_of_inclination) * vertical_direction
    )
    positions[..., half_n_elem:] = positions[
        ..., half_n_elem : half_n_elem + 1
    ] + dl * first_half * (
        np.cos(angle_of_inclination) * horizontal_direction
        - np.sin(angle_of_inclination) * vertical_direction
    )
    return positions


def build_rod(parameters: ButterflyParameters) -> ea.CosseratRod:
    """Create the butterfly rod with a V-shaped initial configuration."""
    n_elem = parameters.n_elem + parameters.n_elem % 2
    positions = build_butterfly_positions(
        n_elem=n_elem,
        total_length=parameters.total_length,
        angle_of_inclination=parameters.angle_of_inclination,
    )
    return ea.CosseratRod.straight_rod(
        n_elem,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=parameters.total_length,
        base_radius=parameters.base_radius,
        density=parameters.density,
        youngs_modulus=parameters.youngs_modulus,
        shear_modulus=parameters.shear_modulus,
        position=positions,
    )


class ButterflySimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    """Simulator collection for the free butterfly rod."""


def build_simulator(
    parameters: ButterflyParameters,
    *,
    backend: str,
) -> tuple[ButterflySimulator, eaj._CosseratRodMemoryBlock, ea.CosseratRod]:
    """Build and finalize the butterfly simulator."""
    simulator = ButterflySimulator()
    rod_block_cls = eaj.configure_rod_block(device=backend)
    simulator.enable_block_supports(ea.CosseratRod, rod_block_cls)
    rod = build_rod(parameters)
    simulator.append(rod)
    simulator.finalize()
    block = tuple(simulator.final_systems())[0]
    return simulator, block, rod


def extract_positions(block: eaj._CosseratRodMemoryBlock) -> np.ndarray:
    """Return node positions as ``(3, n_nodes)`` on the host."""
    return np.asarray(block.device_state["position_collection"]).copy()


def total_energy(rod: ea.CosseratRod) -> float:
    """Return the total mechanical energy of the rod."""
    return float(
        rod.compute_translational_energy()
        + rod.compute_rotational_energy()
        + rod.compute_shear_energy()
        + rod.compute_bending_energy()
    )


def save_simulation_data(
    output_path: Path,
    *,
    times: list[float],
    positions: list[np.ndarray],
    energies: list[float],
    step_skip: int,
) -> None:
    """Persist recorded butterfly diagnostics to an ``.npz`` archive."""
    np.savez(
        output_path,
        time=np.array(times),
        position=np.array(positions, dtype=object),
        total_energy=np.array(energies),
        step_skip=np.array(step_skip),
    )


def plot_energy_evolution(
    output_path: Path,
    *,
    times: np.ndarray,
    energies: np.ndarray,
) -> None:
    """Plot total mechanical energy over time and save to file."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, energies, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total energy (J)")
    ax.set_title("Butterfly rod energy evolution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_animation(
    output_path: Path,
    *,
    times: np.ndarray,
    positions: np.ndarray,
    fps: int = 30,
) -> None:
    """Render a 2D side-view animation of the butterfly rod motion."""
    positions_over_time = np.asarray(positions, dtype=object)
    num_frames = len(positions_over_time)

    try:
        writer = manimation.writers["ffmpeg"](fps=fps)
    except (KeyError, RuntimeError):
        writer = manimation.writers["pillow"](fps=fps)
        output_path = output_path.with_suffix(".gif")

    x_coords = np.concatenate([frame[0] for frame in positions_over_time])
    z_coords = np.concatenate([frame[2] for frame in positions_over_time])
    x_margin = 0.1 * max(np.ptp(x_coords), 1.0e-6)
    z_margin = 0.1 * max(np.ptp(z_coords), 1.0e-6)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(x_coords.min() - x_margin, x_coords.max() + x_margin)
    ax.set_ylim(z_coords.min() - z_margin, z_coords.max() + z_margin)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    (line,) = ax.plot([], [], linewidth=2)

    with writer.saving(fig, str(output_path), dpi=150):
        for frame_idx in tqdm(range(num_frames), desc="Rendering butterfly"):
            frame = positions_over_time[frame_idx]
            line.set_data(frame[0], frame[2])
            ax.set_title(f"Butterfly rod (t = {times[frame_idx]:.3f} s)")
            writer.grab_frame()

    plt.close(fig)
