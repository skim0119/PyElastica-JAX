"""Simulator assembly for the JAX nest packing case.

Builds a packed Cosserat-rod block that reproduces the C++ ``Nest`` case in the
native mm-g-s unit system: random rods settle under gravity inside a cylinder,
interacting through Hertzian rod-rod contact and a Hertzian ground/wall
substrate. Dissipation uses the analytical linear damper (the one intentional
departure from the C++ force-based viscous damping).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import elastica as ea
import elastica_jax as eaj
from operators import SubstrateInteractionJax

if TYPE_CHECKING:
    from run_nest import NestParameters

_SHEAR_CORRECTION = 4.0 / 3.0


def generate_random_rods(
    *,
    num_rods: int,
    rod_length: float,
    rod_radius: float,
    cylinder_radius: float,
    height_range: float,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate random rod origins, directions, and normals inside a cylinder.

    Mirrors the initialization loop of the C++ ``Nest::run`` case: rod origins
    are sampled in a disk of the cylinder radius over a height band, orientations
    are sampled uniformly, and any rod whose far end would leave the cylinder is
    tilted toward vertical until it fits.

    Parameters
    ----------
    num_rods : int
        Number of rods to place.
    rod_length : float
        Rod length (same unit system as the rest of the case).
    rod_radius : float
        Rod cross-section radius.
    cylinder_radius : float
        Radius of the containing cylinder.
    height_range : float
        Vertical band over which rod origins are sampled.
    random_seed : int
        Seed for the NumPy random generator.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
        Per-rod ``(origin, unit_direction, unit_normal)`` tuples.
    """
    rng = np.random.default_rng(random_seed)

    rods: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(num_rods):
        radial = rng.uniform(0.0, cylinder_radius)
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        height = rng.uniform(rod_radius, height_range + rod_radius)
        origin = np.array([radial * np.cos(azimuth), radial * np.sin(azimuth), height])

        alpha = rng.uniform(0.0, 2.0 * np.pi)
        beta = rng.uniform(0.0, np.pi)
        direction = _direction_from_angles(alpha, beta)

        endpoint = origin + rod_length * direction
        while endpoint[0] ** 2 + endpoint[1] ** 2 > cylinder_radius**2:
            beta += 0.05 if beta < np.pi / 2 else -0.05
            direction = _direction_from_angles(alpha, beta)
            endpoint = origin + rod_length * direction

        rods.append((origin, direction, _orthonormal_normal(direction)))

    return rods


def _direction_from_angles(alpha: float, beta: float) -> np.ndarray:
    """Return a unit direction from azimuth ``alpha`` and elevation ``beta``."""
    direction = np.array(
        [np.cos(beta) * np.cos(alpha), np.cos(beta) * np.sin(alpha), np.sin(beta)]
    )
    return direction / np.linalg.norm(direction)


def _orthonormal_normal(direction: np.ndarray) -> np.ndarray:
    """Return a unit vector orthogonal to ``direction`` (matching the C++ choice)."""
    if abs(direction[2]) > 1e-10:
        normal = np.array([1.0, 1.0, (-direction[0] - direction[1]) / direction[2]])
    else:
        normal = np.array([0.0, 0.0, 1.0])
    return normal / np.linalg.norm(normal)


def build_simulation(
    parameters: NestParameters,
    *,
    device: str,
) -> tuple[eaj.Simulator, eaj._CosseratRodMemoryBlock]:
    """Build and finalize the nest simulator.

    Parameters
    ----------
    parameters : NestParameters
        Physical and numerical configuration for the case.
    device : str
        Backend passed to ``configure_rod_block`` (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    tuple[eaj.Simulator, elastica_jax._CosseratRodMemoryBlock]
        The finalized simulator and its packed rod block.
    """
    simulator = eaj.Simulator()
    rod_block = eaj.configure_rod_block(device=device)
    simulator.enable_block_supports(ea.CosseratRod, rod_block)

    _append_rods(simulator, parameters)
    _register_operators(simulator, rod_block, parameters)

    # `configure_rod_block` returns the block instance that PyElastica packs in
    # place during `finalize()`, so it is the finalized block (README, "How to
    # collect data?"); no lookup through `final_systems()` is needed.
    simulator.finalize()
    metadata = eaj.build_block_capsule_metadata(
        rod_block, n_elements_per_rod=parameters.n_elements
    )
    eaj.install_capsule_contact_state(
        rod_block,
        metadata,
        device=rod_block.device,
        dtype=rod_block.device_dtype,
    )
    return simulator, rod_block


def _append_rods(simulator: eaj.Simulator, parameters: NestParameters) -> None:
    """Create straight rods with C++-matched shear stiffness and append them."""
    shear_stiffness = (
        _SHEAR_CORRECTION * parameters.shear_modulus * np.pi * parameters.rod_radius**2
    )
    rods = generate_random_rods(
        num_rods=parameters.num_rods,
        rod_length=parameters.rod_length,
        rod_radius=parameters.rod_radius,
        cylinder_radius=parameters.cylinder_radius,
        height_range=parameters.height_range,
        random_seed=parameters.random_seed,
    )
    for origin, direction, normal in rods:
        rod = ea.CosseratRod.straight_rod(
            parameters.n_elements,
            origin,
            direction,
            normal,
            parameters.rod_length,
            parameters.rod_radius,
            parameters.density,
            youngs_modulus=parameters.youngs_modulus,
            shear_modulus=parameters.shear_modulus,
        )
        rod.shear_matrix[0, 0, :] = shear_stiffness
        rod.shear_matrix[1, 1, :] = shear_stiffness
        simulator.append(rod)


def _register_operators(
    simulator: eaj.Simulator,
    rod_block: eaj._CosseratRodMemoryBlock,
    parameters: NestParameters,
) -> None:
    """Register gravity, substrate, rod-rod contact, and damping operators.

    Registration order defines the synchronize stage order, matching the C++
    force assembly: gravity, then boundary (ground/wall) interaction, then
    rod-rod collision.
    """
    gravity_vector = np.array([0.0, 0.0, -parameters.gravity])
    simulator.operate_block(rod_block).using(
        eaj.GravityForcesJax, acc_gravity=gravity_vector
    )
    simulator.operate_block(rod_block).using(
        SubstrateInteractionJax,
        cylinder_radius=parameters.cylinder_radius,
        rod_radius=parameters.rod_radius,
        stiffness=parameters.plane_stiffness,
        damping=parameters.plane_damping,
        kinetic_friction_coefficient=parameters.plane_friction_kinetic,
        static_velocity_threshold=parameters.plane_static_velocity,
        friction_start_time=parameters.friction_start_time,
    )
    simulator.operate_block(rod_block).using(
        eaj.CapsuleContactOp,
        n_elements_per_rod=parameters.n_elements,
        contact_stiffness=parameters.contact_stiffness,
        contact_damping=parameters.contact_damping,
        contact_stiffness_initial=parameters.contact_stiffness_soft,
        contact_damping_initial=parameters.contact_damping_soft,
        stiffness_ramp_time=parameters.stiffness_ramp_time,
        hertzian=True,
        friction_coefficient=parameters.contact_friction,
        static_velocity_threshold=parameters.contact_static_velocity,
        friction_start_time=parameters.friction_start_time,
        steps_between_detection=parameters.steps_between_detection,
        time_step=parameters.time_step,
    )
    simulator.operate_block(rod_block).using(
        eaj.AnalyticalLinearDamperJax,
        damping_constant=parameters.damping_constant,
        time_step=parameters.time_step,
    )
