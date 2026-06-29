"""Operators and helpers for the JAX Timoshenko beam example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

import elastica as ea
import elastica_jax as eaj


@dataclass(frozen=True)
class TimoshenkoParameters:
    """Physical and numerical parameters for the Timoshenko beam case."""

    n_elem: int = 100
    base_length: float = 3.0
    base_radius: float = 0.25
    density: float = 5000.0
    youngs_modulus: float = 1.0e6
    poisson_ratio: float = 99.0
    end_force_x: float = -15.0
    final_time: float = 10.0

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (self.poisson_ratio + 1.0)

    @property
    def base_area(self) -> float:
        return np.pi * self.base_radius**2

    @property
    def damping_constant(self) -> float:
        return 0.1 / 7.0 / self.density / self.base_area

    @property
    def time_step(self) -> float:
        return 0.07 * (self.base_length / self.n_elem)

    @property
    def end_force(self) -> np.ndarray:
        return np.array([self.end_force_x, 0.0, 0.0], dtype=np.float64)


class TimoshenkoSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    """Simulator collection for the Timoshenko beam case."""


def build_simulator(
    parameters: TimoshenkoParameters,
    *,
    backend: str,
    dtype: np.dtype,
) -> tuple[TimoshenkoSimulator, eaj._CosseratRodMemoryBlock, ea.CosseratRod]:
    """Build and finalize the Timoshenko beam simulator."""
    simulator = TimoshenkoSimulator()
    rod_block_cls = eaj.configure_rod_block(device=backend, device_dtype=dtype)
    simulator.enable_block_supports(ea.CosseratRod, rod_block_cls)

    rod = ea.CosseratRod.straight_rod(
        parameters.n_elem,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        parameters.base_length,
        parameters.base_radius,
        parameters.density,
        youngs_modulus=parameters.youngs_modulus,
        shear_modulus=parameters.shear_modulus,
    )
    simulator.append(rod)

    end_force = parameters.end_force
    simulator.operate_block(rod_block_cls).using(eaj.OneEndFixedJax)
    simulator.operate_block(rod_block_cls).using(
        eaj.EndpointForcesJax,
        0.0 * end_force,
        end_force,
        ramp_up_time=parameters.final_time / 2.0,
    )
    simulator.operate_block(rod_block_cls).using(
        eaj.AnalyticalLinearDamperJax,
        time_step=np.float64(parameters.time_step),
        damping_constant=parameters.damping_constant,
    )
    simulator.finalize()
    block = tuple(simulator.final_systems())[0]
    return simulator, block, rod


def extract_centerline(
    block: eaj._CosseratRodMemoryBlock,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arc length and transverse deflection along the beam centerline."""
    positions = np.asarray(block.position_collection_device)
    s = positions[2]
    deflection = positions[0]
    return s, deflection


def analytical_shearable_deflection(
    rod: ea.CosseratRod,
    end_force: np.ndarray,
    *,
    n_elem: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Return analytical Timoshenko beam deflection for comparison."""
    base_length = float(np.sum(rod.rest_lengths))
    s = np.linspace(0.0, base_length, n_elem)
    acting_force = float(np.abs(end_force[np.nonzero(end_force)][0]))

    linear_prefactor = -acting_force / rod.shear_matrix[0, 0, 0]
    quadratic_prefactor = (
        -acting_force * base_length / 2.0 / rod.bend_matrix[0, 0, 0]
    )
    cubic_prefactor = acting_force / 6.0 / rod.bend_matrix[0, 0, 0]
    deflection = s * (
        linear_prefactor + s * (quadratic_prefactor + s * cubic_prefactor)
    )
    return s, deflection


analytical_shearable = analytical_shearable_deflection


def save_simulation_data(
    output_path: Path,
    *,
    times: list[float],
    centerline_s: list[np.ndarray],
    centerline_deflection: list[np.ndarray],
    step_skip: int,
) -> None:
    """Persist recorded Timoshenko diagnostics to an ``.npz`` archive."""
    np.savez(
        output_path,
        time=np.array(times),
        centerline_s=np.array(centerline_s, dtype=object),
        centerline_deflection=np.array(centerline_deflection, dtype=object),
        step_skip=np.array(step_skip),
    )


def plot_final_deflection(
    output_path: Path,
    *,
    rod: ea.CosseratRod,
    end_force: np.ndarray,
    centerline_s: np.ndarray,
    centerline_deflection: np.ndarray,
) -> None:
    """Plot simulated and analytical Timoshenko deflection curves."""
    analytical_s, analytical_deflection = analytical_shearable_deflection(
        rod,
        end_force,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    ax.plot(analytical_s, analytical_deflection, "k--", label="Timoshenko analytical")
    ax.plot(
        centerline_s[-1],
        centerline_deflection[-1],
        color=to_rgb("xkcd:bluish"),
        linewidth=2,
        label=f"Simulation (n = {rod.n_elems})",
    )
    ax.set_xlabel("Arc length s [m]")
    ax.set_ylabel("Transverse deflection [m]")
    ax.set_title("Timoshenko beam deflection")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
