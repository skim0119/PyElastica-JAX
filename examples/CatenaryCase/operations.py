"""Operators and helpers for the JAX catenary example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from tqdm import tqdm

import elastica as ea
import elastica_jax as eaj


@dataclass(frozen=True)
class CatenaryParameters:
    """Physical and numerical parameters for the catenary case."""

    n_elem: int = 500
    base_length: float = 1.0
    base_radius: float = 0.01
    mass: float = 0.2
    youngs_modulus: float = 1.0e4
    poisson_ratio: float = 0.5
    gravity: float = 9.80665
    damping_constant: float = 0.3
    time_step: float = 1.0e-4
    final_time: float = 30.0

    @property
    def density(self) -> float:
        base_area = np.pi * self.base_radius**2
        volume = base_area * self.base_length
        return self.mass / volume

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (self.poisson_ratio + 1.0)


class GravityForcesJax(eaj.NoOpsJax):
    """Apply uniform gravitational body forces to a rod."""

    def __init__(self, *, acc_gravity: np.ndarray, **kwargs: object) -> None:
        self.acc_gravity = acc_gravity

    def jax_operate_synchronize(
        self,
        rod_view: eaj.JAXRodView,
        time: eaj.JAXTime,
    ) -> eaj.JAXRodView:
        del time
        rod_view.external_forces = (
            rod_view.external_forces
            + self.acc_gravity[:, None] * rod_view.mass[None, :]
        )
        return rod_view


class FixedEndsConstraintJax(eaj.NoOpsJax):
    """Fix selected node positions and director orientations."""

    def __init__(
        self,
        *,
        constrained_position_idx: tuple[int, ...] = (0, -1),
        constrained_director_idx: tuple[int, ...] = (0, -1),
        _system: eaj.RodSystemLike,
    ) -> None:
        n_nodes = _system.position_collection.shape[1]
        n_elems = _system.director_collection.shape[2]
        self.fixed_positions = {
            idx if idx >= 0 else n_nodes + idx: _system.position_collection[
                ..., idx
            ].copy()
            for idx in constrained_position_idx
        }
        self.fixed_directors = {
            idx if idx >= 0 else n_elems + idx: _system.director_collection[
                ..., idx
            ].copy()
            for idx in constrained_director_idx
        }

    def jax_operate_constrain_values(
        self,
        rod_view: eaj.JAXRodView,
        time: eaj.JAXTime,
    ) -> eaj.JAXRodView:
        del time
        for node_idx, fixed_position in self.fixed_positions.items():
            rod_view.position_collection = rod_view.position_collection.at[
                :, node_idx
            ].set(fixed_position)
        for elem_idx, fixed_director in self.fixed_directors.items():
            rod_view.director_collection = rod_view.director_collection.at[
                :, :, elem_idx
            ].set(fixed_director)
        return rod_view

    def jax_operate_constrain_rates(
        self,
        rod_view: eaj.JAXRodView,
        time: eaj.JAXTime,
    ) -> eaj.JAXRodView:
        del time
        for node_idx in self.fixed_positions:
            rod_view.velocity_collection = rod_view.velocity_collection.at[
                :, node_idx
            ].set(0.0)
        for elem_idx in self.fixed_directors:
            rod_view.omega_collection = rod_view.omega_collection.at[:, elem_idx].set(
                0.0
            )
        return rod_view


class CatenarySimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    """Simulator collection for the catenary case."""


def build_simulator(
    parameters: CatenaryParameters,
    *,
    backend: str,
) -> tuple[CatenarySimulator, eaj._CosseratRodMemoryBlock]:
    """Build and finalize the catenary simulator."""
    simulator = CatenarySimulator()
    rod_block = eaj.configure_rod_block(device=backend)
    simulator.enable_block_supports(ea.CosseratRod, rod_block)

    start = np.zeros(3)
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    rod = ea.CosseratRod.straight_rod(
        parameters.n_elem,
        start,
        direction,
        normal,
        parameters.base_length,
        parameters.base_radius,
        parameters.density,
        youngs_modulus=parameters.youngs_modulus,
        shear_modulus=parameters.shear_modulus,
    )
    simulator.append(rod)

    gravity_vector = -parameters.gravity * normal
    simulator.operate_block(rod_block).using(
        GravityForcesJax,
        acc_gravity=gravity_vector,
    )
    simulator.operate_block(rod_block).using(
        eaj.AnalyticalLinearDamperJax,
        damping_constant=parameters.damping_constant,
        time_step=parameters.time_step,
    )
    simulator.operate_block(rod_block).using(
        FixedEndsConstraintJax,
        constrained_position_idx=(0, -1),
        constrained_director_idx=(0, -1),
    )
    simulator.finalize()
    return simulator, rod_block


def extract_positions(block: eaj._CosseratRodMemoryBlock) -> np.ndarray:
    """Return node positions as ``(3, n_nodes)`` on the host."""
    return np.asarray(block.device_state["position_collection"]).copy()


def save_simulation_data(
    output_path: Path,
    *,
    times: list[float],
    positions: list[np.ndarray],
    step_skip: int,
    base_length: float,
) -> None:
    """Persist recorded catenary diagnostics to an ``.npz`` archive."""
    np.savez(
        output_path,
        time=np.array(times),
        position=np.array(positions, dtype=object),
        step_skip=np.array(step_skip),
        base_length=np.array(base_length),
    )


def plot_final_shape(
    output_path: Path,
    *,
    positions: np.ndarray,
    base_length: float,
) -> None:
    """Plot the final catenary shape against the analytical solution."""
    final_position = positions[-1]
    lowest_point = np.min(final_position[2])
    x_catenary = np.linspace(0.0, base_length, 100)

    def f_non_elastic_catenary(x: float) -> float:
        return x * (1.0 - np.cosh(1.0 / (2.0 * x))) - lowest_point

    a = float(optimize.fsolve(f_non_elastic_catenary, x0=1.0)[0])
    y_catenary = a * np.cosh((x_catenary - 0.5) / a) - a * np.cosh(1.0 / (2.0 * a))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(final_position[0], final_position[2], label="Simulation", linewidth=2)
    ax.plot(
        x_catenary,
        y_catenary,
        label="Catenary (analytical)",
        linewidth=2,
        linestyle="dashed",
    )
    ax.set_xlim(0.0, base_length)
    ax.set_ylim(lowest_point, 0.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Catenary final shape")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_animation(
    output_path: Path,
    *,
    times: np.ndarray,
    positions: np.ndarray,
    base_length: float,
    fps: int = 20,
) -> None:
    """Render a 2D animation of the settling catenary."""
    writer = manimation.writers["ffmpeg"](fps=fps)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0.0, base_length)
    ax.set_ylim(-0.5 * base_length, 0.5 * base_length)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    (line,) = ax.plot([], [], linewidth=2)

    with writer.saving(fig, str(output_path), dpi=150):
        for frame_idx in tqdm(range(len(positions)), desc="Rendering catenary"):
            frame = positions[frame_idx]
            line.set_data(frame[0], frame[2])
            ax.set_title(f"Catenary (t = {times[frame_idx]:.3f} s)")
            writer.grab_frame()

    plt.close(fig)
