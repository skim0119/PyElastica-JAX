"""Simulator assembly for the JAX active-matter snake-pit case.

Builds a Cosserat-rod block of randomly packed active snakes enclosed by a
floor and four side walls. Rods are driven by a traveling-wave internal torque
(:class:`~operators.ActiveMatterForcingJax`), interact through capsule-capsule
rod-rod contact and capsule-half-space wall contact, and are dissipated by an
analytical linear damper. See ``CASE_DESCRIPTION.md`` for the reference physics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import elastica as ea
import elastica_jax as eaj
import jax
from operators import ActiveMatterForcingJax, spline_actuation_amplitude

if TYPE_CHECKING:
    from run_snake_pit import SnakePitParameters

GRAVITY_AXIS = np.array([0.0, 0.0, 1.0])


def build_simulation(
    parameters: SnakePitParameters,
    *,
    device: jax.Device,
    seed: int,
) -> tuple[eaj.Simulator, eaj._CosseratRodMemoryBlock]:
    """Build and finalize the snake-pit simulator on the given device.

    Parameters
    ----------
    parameters : SnakePitParameters
        Physical and numerical configuration for the case.
    device : jax.Device
        Device that owns the packed JAX rod block.
    seed : int
        Random seed for rod placement inside the packing cylinder.

    Returns
    -------
    tuple[eaj.Simulator, elastica_jax._CosseratRodMemoryBlock]
        The finalized simulator and its rod block.
    """
    wall_origins, wall_normals = parameters.pit_walls()

    simulator = eaj.Simulator()
    rod_block_cls = eaj.configure_rod_block(device=device)
    simulator.enable_block_supports(ea.CosseratRod, rod_block_cls)

    for start, direction in instantiate_rods_in_cylinder(parameters, seed):
        normal = _orthonormal_normal(direction)
        rod = ea.CosseratRod.straight_rod(
            parameters.n_elements,
            start,
            direction,
            normal,
            parameters.length,
            parameters.radius,
            parameters.density,
            youngs_modulus=parameters.youngs_modulus,
            # Shear modulus is intentionally left at the default: it is unused
            # by this case and its value is slightly controversial.
        )
        simulator.append(rod)

    spline_amplitude = spline_actuation_amplitude(parameters.n_elements)
    simulator.operate_block(rod_block_cls).using(
        ActiveMatterForcingJax,
        parameters=parameters,
        n_snakes=parameters.n_snakes,
        n_elements=parameters.n_elements,
        gravity_axis=GRAVITY_AXIS,
        spline_amplitude=spline_amplitude,
        ramp="sinusoidal",
    )
    simulator.operate_block(rod_block_cls).using(
        eaj.CapsuleContactOp,
        n_elements_per_rod=parameters.n_elements,
        contact_stiffness=parameters.contact_stiffness,
        contact_damping=parameters.contact_damping,
        steps_between_detection=parameters.steps_between_detection,
        time_step=parameters.time_step,
    )
    simulator.operate_block(rod_block_cls).using(
        eaj.WallContactOp,
        n_elements_per_rod=parameters.n_elements,
        wall_origins=wall_origins,
        wall_normals=wall_normals,
        contact_stiffness=parameters.contact_stiffness,
        contact_damping=parameters.contact_damping,
    )
    simulator.operate_block(rod_block_cls).using(
        eaj.AnalyticalLinearDamperJax,
        damping_constant=parameters.damping_rate,
        time_step=parameters.time_step,
    )
    # `configure_rod_block*` returns the block instance that PyElastica packs in
    # place during `finalize()`, so it is the finalized block (README, "How to
    # collect data?"); no lookup through `final_systems()` is needed.
    simulator.finalize()

    metadata = eaj.build_block_capsule_metadata(
        rod_block_cls, n_elements_per_rod=parameters.n_elements
    )
    eaj.install_capsule_contact_state(
        rod_block_cls,
        metadata,
        device=device,
        dtype=rod_block_cls.device_dtype,
    )
    return simulator, rod_block_cls


def instantiate_rods_in_cylinder(
    parameters: SnakePitParameters,
    seed: int,
    *,
    max_attempts: int = 10_000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Place rods with random pose inside a vertical packing cylinder.

    Each rod is accepted only when both endpoints lie inside the cylinder.
    """
    rng = np.random.default_rng(seed)
    n_rods = parameters.n_snakes
    cylinder_radius = parameters.packing_initial_radial_span_ratio * parameters.length
    height = parameters.packing_initial_vertical_span_ratio * parameters.length
    min_z = parameters.radius
    rod_length = parameters.length

    rho = cylinder_radius * np.sqrt(rng.random((n_rods, max_attempts)))
    phi = 2.0 * np.pi * rng.random((n_rods, max_attempts))
    start_z = min_z + rng.random((n_rods, max_attempts)) * (height - min_z)
    start_x = rho * np.cos(phi)
    start_y = rho * np.sin(phi)

    alpha = 2.0 * np.pi * rng.random((n_rods, max_attempts))
    beta = 0.2 * np.pi * rng.random((n_rods, max_attempts))
    cos_beta = np.cos(beta)
    dir_x = np.cos(alpha) * cos_beta
    dir_y = np.sin(alpha) * cos_beta
    dir_z = np.sin(beta)

    end_x = start_x + rod_length * dir_x
    end_y = start_y + rod_length * dir_y
    end_z = start_z + rod_length * dir_z

    valid = (
        (np.hypot(end_x, end_y) <= cylinder_radius)
        & (min_z <= end_z)
        & (end_z <= height)
    )
    assert np.all(valid.any(axis=1)), (
        "Could not place a snake inside the random cylinder within "
        f"{max_attempts} attempts."
    )

    attempt_idx = valid.argmax(axis=1)
    rod_idx = np.arange(n_rods)
    starts = np.stack(
        (
            start_x[rod_idx, attempt_idx],
            start_y[rod_idx, attempt_idx],
            start_z[rod_idx, attempt_idx],
        ),
        axis=1,
    )
    directions = np.stack(
        (
            dir_x[rod_idx, attempt_idx],
            dir_y[rod_idx, attempt_idx],
            dir_z[rod_idx, attempt_idx],
        ),
        axis=1,
    )
    return [(starts[i], directions[i]) for i in range(n_rods)]


def _orthonormal_normal(direction: np.ndarray) -> np.ndarray:
    """Return a unit vector orthogonal to ``direction`` for the rod frame."""
    normal_seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal_seed, direction)) > 0.9:
        normal_seed = np.array([0.0, 1.0, 0.0])
    normal = normal_seed - np.dot(normal_seed, direction) * direction
    return normal / np.linalg.norm(normal)
