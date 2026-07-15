"""Operators and helpers for the JAX axial stretching example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import elastica as ea
import elastica_jax as eaj


@dataclass(frozen=True)
class AxialStretchingParameters:
    """Physical and numerical parameters for the axial stretching case."""

    n_elem: int = 19
    base_length: float = 1.0
    base_radius: float = 0.025
    density: float = 1000.0
    youngs_modulus: float = 1.0e4
    poisson_ratio: float = 0.5
    end_force_x: float = 1.0
    damping_constant: float = 0.1
    time_step: float = 0.1 / 19.0
    final_time: float = 0.2

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (self.poisson_ratio + 1.0)


class AxialStretchingSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    """Simulator collection for the axial stretching case."""


def build_simulator(
    parameters: AxialStretchingParameters,
    *,
    backend: str,
) -> tuple[AxialStretchingSimulator, eaj._CosseratRodMemoryBlock]:
    """Build and finalize the axial stretching simulator."""
    simulator = AxialStretchingSimulator()
    rod_block_cls = eaj.configure_rod_block(device=backend)
    simulator.enable_block_supports(ea.CosseratRod, rod_block_cls)

    rod = ea.CosseratRod.straight_rod(
        parameters.n_elem,
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        parameters.base_length,
        parameters.base_radius,
        parameters.density,
        youngs_modulus=parameters.youngs_modulus,
        shear_modulus=parameters.shear_modulus,
    )
    simulator.append(rod)

    end_force = np.array([parameters.end_force_x, 0.0, 0.0], dtype=np.float64)
    simulator.operate_block(rod_block_cls).using(eaj.OneEndFixedJax)
    simulator.operate_block(rod_block_cls).using(
        eaj.EndpointForcesJax,
        0.0 * end_force,
        end_force,
        ramp_up_time=1.0e-2,
    )
    simulator.operate_block(rod_block_cls).using(
        eaj.AnalyticalLinearDamperJax,
        time_step=np.float64(parameters.time_step),
        damping_constant=parameters.damping_constant,
    )
    simulator.finalize()
    block = tuple(simulator.final_systems())[0]
    return simulator, block


def extract_centerline(
    block: eaj._CosseratRodMemoryBlock,
) -> tuple[np.ndarray, np.ndarray]:
    """Return axial coordinate and x-displacement along the rod centerline."""
    positions = np.asarray(block.device_state["position_collection"])
    x = positions[0]
    rest_length = np.sum(np.linalg.norm(np.diff(positions, axis=1), axis=0))
    s = np.linspace(0.0, rest_length, positions.shape[1])
    return s, x


def save_simulation_data(
    output_path: Path,
    *,
    times: list[float],
    centerline_s: list[np.ndarray],
    centerline_x: list[np.ndarray],
    step_skip: int,
) -> None:
    """Persist recorded axial stretching diagnostics to an ``.npz`` archive."""
    np.savez(
        output_path,
        time=np.array(times),
        centerline_s=np.array(centerline_s, dtype=object),
        centerline_x=np.array(centerline_x, dtype=object),
        step_skip=np.array(step_skip),
    )


def plot_centerline_evolution(
    output_path: Path,
    *,
    times: np.ndarray,
    centerline_s: np.ndarray,
    centerline_x: np.ndarray,
) -> None:
    """Plot axial displacement along the rod at selected times."""
    fig, ax = plt.subplots(figsize=(10, 6))
    frame_indices = np.linspace(0, len(times) - 1, num=min(6, len(times)), dtype=int)
    for frame_idx in frame_indices:
        ax.plot(
            centerline_s[frame_idx],
            centerline_x[frame_idx],
            label=f"t = {times[frame_idx]:.3f} s",
        )
    ax.set_xlabel("Arc length s [m]")
    ax.set_ylabel("x position [m]")
    ax.set_title("Axial stretching centerline evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_centerline(
    output_path: Path,
    *,
    centerline_s: np.ndarray,
    centerline_x: np.ndarray,
) -> None:
    """Plot the final stretched rod centerline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(centerline_s[-1], centerline_x[-1], linewidth=2)
    ax.set_xlabel("Arc length s [m]")
    ax.set_ylabel("x position [m]")
    ax.set_title("Axial stretching final centerline")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
