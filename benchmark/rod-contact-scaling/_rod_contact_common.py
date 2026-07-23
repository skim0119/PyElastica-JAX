"""Shared setup for rod-rod contact weak-scaling benchmarks."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402

jax.config.update("jax_enable_x64", True)

BenchmarkTiming: TypeAlias = tuple[float, float]

ROD_LENGTH = 0.35
ROD_RADIUS = 0.011 * ROD_LENGTH
N_ELEMENTS = 10
DENSITY = 1000.0
YOUNGS_MODULUS = 1.0e5
DT = 2.0e-4
CONTACT_STIFFNESS = 1.0e3
CONTACT_DAMPING = 1.0e-3
ATTRACT_STRENGTH = 100.0
DAMPING_RATE = 1.0e-4
RING_RADIUS_RATIO = 1.5
STEPS_BETWEEN_DETECTION = 0


@dataclass(frozen=True)
class ContactScalingConfig:
    """Fixed physics for the rod-rod contact scaling benchmark."""

    n_rods: int
    n_elements: int = N_ELEMENTS
    rod_length: float = ROD_LENGTH
    rod_radius: float = ROD_RADIUS
    time_step: float = DT
    steps_between_detection: int = STEPS_BETWEEN_DETECTION

    @property
    def ring_radius(self) -> float:
        return RING_RADIUS_RATIO * self.rod_length

    @property
    def attract_center(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.5 * self.rod_length], dtype=np.float64)


class ContactScalingSimulator(eaj.Simulator):
    """System collection for the rod-rod contact scaling benchmark."""


class ContactScalingPairwiseSimulator(eaj.Simulator):
    """JAX scaling benchmark using per-rod-pair ``RodRodContactJax`` operators."""


class ContactScalingPyElasticaSimulator(
    ea.BaseSystemCollection, ea.Forcing, ea.Damping, ea.Contact
):
    """PyElastica baseline simulator for the rod-rod contact benchmark."""


class CenterAttractForcesJax(eaj.NoBlockOpJax):
    """Apply a horizontal-plane spring that pulls nodes toward a fixed center.

    Parameters
    ----------
    attract_strength : float
        Linear stiffness multiplying ``(center - position)`` in the xy plane.
    center : numpy.ndarray
        Attraction target, shape ``(3,)``. The z component is ignored.
    """

    def __init__(
        self,
        *,
        attract_strength: float,
        center: np.ndarray,
        _system=None,
    ) -> None:
        assert _system is not None, "CenterAttractForcesJax requires a finalized block."
        self.attract_strength = attract_strength
        self.center = center

    def jax_block_operate_synchronize(self, state, time):
        positions = state["position_collection"]
        delta = self.center[:, None] - positions
        plane_mask = jnp.array([1.0, 1.0, 0.0], dtype=positions.dtype)[:, None]
        delta = delta * plane_mask
        attract_forces = self.attract_strength * delta * state["mass"][None, :]
        return {
            **state,
            "external_forces": state["external_forces"] + attract_forces,
        }


class CenterAttractForcesPy(ea.NoForces):
    """Apply a horizontal-plane spring that pulls nodes toward a fixed center."""

    def __init__(self, *, attract_strength: float, center: np.ndarray) -> None:
        super().__init__()
        self.attract_strength = attract_strength
        self.center = center

    def apply_forces(self, system, time: np.float64 = 0.0) -> None:
        delta = self.center[:, None] - system.position_collection
        delta[2, :] = 0.0
        system.external_forces += self.attract_strength * delta * system.mass[None, :]


def _orthonormal_normal(direction: np.ndarray) -> np.ndarray:
    seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(seed, direction)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = seed - np.dot(seed, direction) * direction
    return normal / np.linalg.norm(normal)


def build_simulation(
    config: ContactScalingConfig,
    *,
    device: jax.Device,
    broad_phase: str = "spatial_hash",
) -> tuple[ContactScalingSimulator, eaj._CosseratRodMemoryBlock]:
    """Build a packed rod block with center attraction and rod-rod contact only."""
    simulator = ContactScalingSimulator()
    rod_block = eaj.configure_rod_block(device=device)
    simulator.enable_block_supports(ea.CosseratRod, rod_block)

    ring_radius = config.ring_radius
    for rod_idx in range(config.n_rods):
        theta = 2.0 * np.pi * rod_idx / config.n_rods
        start = np.array(
            [
                ring_radius * np.cos(theta),
                ring_radius * np.sin(theta),
                0.5 * config.rod_length,
            ],
            dtype=np.float64,
        )
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rod = ea.CosseratRod.straight_rod(
            config.n_elements,
            start,
            direction,
            _orthonormal_normal(direction),
            config.rod_length,
            config.rod_radius,
            DENSITY,
            youngs_modulus=YOUNGS_MODULUS,
        )
        simulator.append(rod)

    simulator.operate_block(rod_block).using(
        CenterAttractForcesJax,
        attract_strength=ATTRACT_STRENGTH,
        center=config.attract_center,
    )
    simulator.operate_block(rod_block).using(
        eaj.CapsuleContactOp,
        n_elements_per_rod=config.n_elements,
        contact_stiffness=CONTACT_STIFFNESS,
        contact_damping=CONTACT_DAMPING,
        steps_between_detection=config.steps_between_detection,
        time_step=config.time_step,
        broad_phase=broad_phase,
    )
    simulator.operate_block(rod_block).using(
        eaj.AnalyticalLinearDamperJax,
        damping_constant=DAMPING_RATE,
        time_step=config.time_step,
    )
    simulator.finalize()

    metadata = eaj.build_block_capsule_metadata(
        rod_block,
        n_elements_per_rod=config.n_elements,
        broad_phase=broad_phase,
    )
    eaj.install_capsule_contact_state(
        rod_block,
        metadata,
        device=device,
        dtype=rod_block.device_dtype,
    )
    return simulator, rod_block


def build_simulation_pairwise(
    config: ContactScalingConfig,
    *,
    device: jax.Device,
) -> tuple[
    ContactScalingPairwiseSimulator, eaj._CosseratRodMemoryBlock, list[ea.CosseratRod]
]:
    """Build packed rods with PyElastica-style per-pair rod-rod contact."""
    simulator = ContactScalingPairwiseSimulator()
    rod_block = eaj.configure_rod_block(device=device)
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    rods: list[ea.CosseratRod] = []

    ring_radius = config.ring_radius
    for rod_idx in range(config.n_rods):
        theta = 2.0 * np.pi * rod_idx / config.n_rods
        start = np.array(
            [
                ring_radius * np.cos(theta),
                ring_radius * np.sin(theta),
                0.5 * config.rod_length,
            ],
            dtype=np.float64,
        )
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rod = ea.CosseratRod.straight_rod(
            config.n_elements,
            start,
            direction,
            _orthonormal_normal(direction),
            config.rod_length,
            config.rod_radius,
            DENSITY,
            youngs_modulus=YOUNGS_MODULUS,
        )
        simulator.append(rod)
        rods.append(rod)

    simulator.operate_block(rod_block).using(
        CenterAttractForcesJax,
        attract_strength=ATTRACT_STRENGTH,
        center=config.attract_center,
    )
    simulator.operate_block(rod_block).using(
        eaj.AnalyticalLinearDamperJax,
        damping_constant=DAMPING_RATE,
        time_step=config.time_step,
    )

    for left_idx in range(len(rods)):
        for right_idx in range(left_idx + 1, len(rods)):
            simulator.pairwise_interaction(rods[left_idx], rods[right_idx]).using(
                eaj.RodRodContactJax,
                k=CONTACT_STIFFNESS,
                nu=CONTACT_DAMPING,
            )

    simulator.finalize()
    return simulator, rod_block, rods


def build_simulation_pyelastica(
    config: ContactScalingConfig,
) -> tuple[ContactScalingPyElasticaSimulator, list[ea.CosseratRod]]:
    """Build the matching PyElastica rod-rod contact benchmark."""
    simulator = ContactScalingPyElasticaSimulator()
    rods: list[ea.CosseratRod] = []

    ring_radius = config.ring_radius
    for rod_idx in range(config.n_rods):
        theta = 2.0 * np.pi * rod_idx / config.n_rods
        start = np.array(
            [
                ring_radius * np.cos(theta),
                ring_radius * np.sin(theta),
                0.5 * config.rod_length,
            ],
            dtype=np.float64,
        )
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rod = ea.CosseratRod.straight_rod(
            config.n_elements,
            start,
            direction,
            _orthonormal_normal(direction),
            config.rod_length,
            config.rod_radius,
            DENSITY,
            youngs_modulus=YOUNGS_MODULUS,
        )
        simulator.append(rod)
        simulator.add_forcing_to(rod).using(
            CenterAttractForcesPy,
            attract_strength=ATTRACT_STRENGTH,
            center=config.attract_center,
        )
        simulator.dampen(rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=DAMPING_RATE,
            time_step=config.time_step,
        )
        rods.append(rod)

    for left_idx in range(len(rods)):
        for right_idx in range(left_idx + 1, len(rods)):
            simulator.detect_contact_between(rods[left_idx], rods[right_idx]).using(
                ea.RodRodContact,
                k=CONTACT_STIFFNESS,
                nu=CONTACT_DAMPING,
            )

    simulator.finalize()
    return simulator, rods


def _block_until_ready(rod_block: eaj._CosseratRodMemoryBlock) -> None:
    for leaf in jax.tree_util.tree_leaves(rod_block.jax_get_state()):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def run_rollout(
    *,
    backend: str,
    n_rods: int,
    steps: int,
    warmup_runs: int,
    n_elements: int = N_ELEMENTS,
    steps_between_detection: int = STEPS_BETWEEN_DETECTION,
    broad_phase: str = "spatial_hash",
) -> BenchmarkTiming:
    """Build the simulator and time a fixed-length Position Verlet rollout."""
    assert steps > 0, "steps must be positive."
    assert warmup_runs >= 0, "warmup_runs must be nonnegative."

    config = ContactScalingConfig(
        n_rods=n_rods,
        n_elements=n_elements,
        steps_between_detection=steps_between_detection,
    )
    device = eaj.resolve_backend_devices(backend)[0]
    dt_value = np.float64(config.time_step)

    with jax.default_device(device):
        instantiate_start = time.perf_counter()
        simulator, rod_block = build_simulation(
            config, device=device, broad_phase=broad_phase
        )
        instantiate_seconds = time.perf_counter() - instantiate_start

        stepper = eaj.PositionVerletJAX()
        time_value = np.float64(0.0)
        for _ in range(warmup_runs):
            time_value = stepper.integrate(
                simulator,
                time=time_value,
                final_time=time_value + steps * dt_value,
                dt=dt_value,
            )
            _block_until_ready(rod_block)

        rollout_start = time.perf_counter()
        time_value = stepper.integrate(
            simulator,
            time=time_value,
            final_time=time_value + steps * dt_value,
            dt=dt_value,
        )
        _block_until_ready(rod_block)
        rollout_seconds = time.perf_counter() - rollout_start

    return instantiate_seconds, rollout_seconds


def run_rollout_pairwise(
    *,
    backend: str,
    n_rods: int,
    steps: int,
    warmup_runs: int,
    n_elements: int = N_ELEMENTS,
) -> BenchmarkTiming:
    """Build the pairwise JAX simulator and time a fixed-length rollout."""
    assert steps > 0, "steps must be positive."
    assert warmup_runs >= 0, "warmup_runs must be nonnegative."

    config = ContactScalingConfig(
        n_rods=n_rods,
        n_elements=n_elements,
    )
    device = eaj.resolve_backend_devices(backend)[0]
    dt_value = np.float64(config.time_step)

    with jax.default_device(device):
        instantiate_start = time.perf_counter()
        simulator, rod_block, _ = build_simulation_pairwise(config, device=device)
        instantiate_seconds = time.perf_counter() - instantiate_start

        stepper = eaj.PositionVerletJAX()
        time_value = np.float64(0.0)
        for _ in range(warmup_runs):
            time_value = stepper.integrate(
                simulator,
                time=time_value,
                final_time=time_value + steps * dt_value,
                dt=dt_value,
            )
            _block_until_ready(rod_block)

        rollout_start = time.perf_counter()
        time_value = stepper.integrate(
            simulator,
            time=time_value,
            final_time=time_value + steps * dt_value,
            dt=dt_value,
        )
        _block_until_ready(rod_block)
        rollout_seconds = time.perf_counter() - rollout_start

    return instantiate_seconds, rollout_seconds


def run_rollout_pyelastica(
    *,
    n_rods: int,
    steps: int,
    warmup_runs: int,
    n_elements: int = N_ELEMENTS,
) -> BenchmarkTiming:
    """Build the PyElastica simulator and time a fixed-length rollout."""
    assert steps > 0, "steps must be positive."
    assert warmup_runs >= 0, "warmup_runs must be nonnegative."

    config = ContactScalingConfig(
        n_rods=n_rods,
        n_elements=n_elements,
    )
    dt_value = np.float64(config.time_step)

    instantiate_start = time.perf_counter()
    simulator, _ = build_simulation_pyelastica(config)
    instantiate_seconds = time.perf_counter() - instantiate_start

    stepper = ea.PositionVerlet()
    time_value = np.float64(0.0)
    for _ in range(warmup_runs):
        for _ in range(steps):
            time_value = stepper.step(simulator, time_value, dt_value)

    rollout_start = time.perf_counter()
    for _ in range(steps):
        time_value = stepper.step(simulator, time_value, dt_value)
    rollout_seconds = time.perf_counter() - rollout_start
    return instantiate_seconds, rollout_seconds
