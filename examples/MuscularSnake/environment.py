"""Simulator assembly for the muscular snake case.

Builds a continuum musculoskeletal snake: a single Cosserat rod body actuated by
eight muscle rods (four antagonist pairs) bound side-by-side through surface
joints and driven by a head-to-tail traveling wave. The body slides on an
anisotropic-friction ground plane, converting lateral undulation into net
forward locomotion (Zhang et al., Nat. Commun. 2019).

The body and muscles live in separate memory blocks on the same device. Block
operators apply body-only or muscle-only physics without per-rod slicing, while
``pairwise_interaction`` couples the blocks through surface joints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402
from block_utils import distinct_body_and_muscle_rod_types  # noqa: E402
from operators import (  # noqa: E402
    MuscularSnakeMuscleForcesBlockJax,
    MuscularSnakePlaneContactJax,
    SurfaceJointSideBySideJax,
    get_connection_vector_straight_straight_rod,
)

if TYPE_CHECKING:
    from run_muscular_snake import MuscularSnakeParameters

_TENDON_HARDENING_FACTOR = 50000
_MUSCLE_BELLY_START_ELEM = 4 * 3
_MUSCLE_BELLY_END_ELEM = 9 * 3


class MuscularSnakeSimulator(
    ea.BaseSystemCollection,
    eaj.JAXOps,
    eaj.JAXOpsBlock,
    eaj.JAXInteraction,
):
    """System collection mixin for the JAX muscular snake case."""


def build_simulation(
    parameters: MuscularSnakeParameters,
    *,
    device: str,
) -> tuple[
    MuscularSnakeSimulator,
    eaj._CosseratRodMemoryBlock,
    eaj._CosseratRodMemoryBlock,
    list[ea.CosseratRod],
]:
    """Build and finalize the JAX muscular snake simulator.

    Parameters
    ----------
    parameters : MuscularSnakeParameters
        Physical and numerical configuration for the case.
    device : str
        Backend passed to ``configure_rod_block`` (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    tuple
        The finalized simulator, body block, muscle block, and reference rods.
    """
    body_type, muscle_type = distinct_body_and_muscle_rod_types()
    simulator = MuscularSnakeSimulator()
    body_block = eaj.configure_rod_block(device=device)
    muscle_block = eaj.configure_rod_block(device=device)
    simulator.enable_block_supports(body_type, body_block)
    simulator.enable_block_supports(muscle_type, muscle_block)

    snake_body = _build_snake_body(body_type, parameters)
    body_elem_length = float(snake_body.rest_lengths[0])
    muscle_rods, muscle_end_connection_index = _build_muscle_rods(
        muscle_type,
        parameters,
        body_elem_length=body_elem_length,
    )
    rod_list = [snake_body, *muscle_rods]
    simulator.append(snake_body)
    for muscle_rod in muscle_rods:
        simulator.append(muscle_rod)

    _register_block_operators(
        simulator,
        body_block,
        muscle_block,
        muscle_rods,
        parameters,
    )
    _register_muscle_connections(
        simulator,
        snake_body,
        muscle_rods,
        muscle_end_connection_index,
        body_elem_length=body_elem_length,
        parameters=parameters,
    )

    simulator.finalize()
    finalized_blocks = tuple(simulator.final_systems())
    assert len(finalized_blocks) == 2, "Expected body and muscle memory blocks."
    return simulator, finalized_blocks[0], finalized_blocks[1], rod_list


def _build_snake_body(
    body_type: type[ea.CosseratRod],
    parameters: MuscularSnakeParameters,
) -> ea.CosseratRod:
    """Return the straight Cosserat rod representing the snake body."""
    start = np.array([0.0, 0.0, parameters.base_radius_body])
    return body_type.straight_rod(
        parameters.n_elem_body,
        start,
        parameters.direction,
        parameters.normal,
        parameters.base_length_body,
        parameters.base_radius_body,
        parameters.density_body,
        youngs_modulus=parameters.youngs_modulus,
        shear_modulus=parameters.shear_modulus,
    )


def _build_muscle_rods(
    muscle_type: type[ea.CosseratRod],
    parameters: MuscularSnakeParameters,
    *,
    body_elem_length: float,
) -> tuple[list[ea.CosseratRod], list[int]]:
    """Return the eight muscle rods and their body end-connection indices."""
    muscle_rod_list: list[ea.CosseratRod] = []
    muscle_end_connection_index: list[int] = []
    half = parameters.n_muscle_fibers // 2
    group_specs = (
        (half, parameters.n_elem_muscle_group_one, parameters.base_length_group_one),
        (
            parameters.n_muscle_fibers - half,
            parameters.n_elem_muscle_group_two,
            parameters.base_length_group_two,
        ),
    )

    muscle_idx = 0
    for group_count, n_elem_muscle, base_length_muscle in group_specs:
        muscle_radius = np.full(n_elem_muscle, parameters.muscle_radius_tendon)
        muscle_radius[_MUSCLE_BELLY_START_ELEM:_MUSCLE_BELLY_END_ELEM] = (
            parameters.muscle_radius_belly
        )
        for _ in range(group_count):
            body_index = parameters.muscle_start_connection_index[muscle_idx]
            side_sign = -1 if muscle_idx % 2 == 0 else 1
            start_muscle = np.array(
                [
                    body_index * body_elem_length,
                    side_sign * (parameters.base_radius_body + 0.003),
                    parameters.base_radius_body,
                ]
            )
            muscle_rod = muscle_type.straight_rod(
                n_elem_muscle,
                start_muscle,
                parameters.direction,
                parameters.normal,
                base_length_muscle,
                muscle_radius,
                parameters.density_muscle,
                youngs_modulus=parameters.youngs_modulus_muscle,
                shear_modulus=parameters.shear_modulus_muscle,
            )
            _harden_tendon_regions(muscle_rod)
            muscle_rod_list.append(muscle_rod)
            muscle_end_connection_index.append(body_index + n_elem_muscle)
            muscle_idx += 1

    return muscle_rod_list, muscle_end_connection_index


def _harden_tendon_regions(muscle_rod: ea.CosseratRod) -> None:
    """Stiffen the shear response of the muscle rod's tendon end regions."""
    muscle_rod.shear_matrix[..., : 4 * 3] *= _TENDON_HARDENING_FACTOR
    muscle_rod.shear_matrix[..., 9 * 3 :] *= _TENDON_HARDENING_FACTOR


def _register_block_operators(
    simulator: MuscularSnakeSimulator,
    body_block: eaj._CosseratRodMemoryBlock,
    muscle_block: eaj._CosseratRodMemoryBlock,
    muscle_rods: list[ea.CosseratRod],
    parameters: MuscularSnakeParameters,
) -> None:
    """Register body and muscle block operators."""
    kinetic_mu_array = (
        np.array([1.0, 1.5, 2.0], dtype=np.float64) * parameters.friction_mu
    )
    static_mu_array = 2.0 * kinetic_mu_array
    muscle_time_delays = np.asarray(
        [
            parameters.muscle_start_connection_index[::-1][muscle_idx] / 101.76
            for muscle_idx in range(len(muscle_rods))
        ],
        dtype=np.float64,
    )
    muscle_sides = np.asarray(
        [1 if muscle_idx % 2 == 0 else -1 for muscle_idx in range(len(muscle_rods))],
        dtype=np.int64,
    )

    simulator.operate_block(body_block).using(
        eaj.GravityAnalyticalDamperJax,
        acc_gravity=np.array([0.0, 0.0, parameters.gravitational_acc]),
        time_step=parameters.time_step,
        damping_constant=parameters.nu_body,
    )
    simulator.operate_block(body_block).using(
        MuscularSnakePlaneContactJax,
        plane_origin=np.array([0.0, 0.0, 0.0]),
        plane_normal=parameters.normal,
        slip_velocity_tol=1e-8,
        contact_k=1e1,
        contact_nu=40.0,
        kinetic_mu_array=kinetic_mu_array,
        static_mu_array=static_mu_array,
    )
    simulator.operate_block(muscle_block).using(
        MuscularSnakeMuscleForcesBlockJax,
        wave_number=2.0 * np.pi / parameters.period,
        arm_length=parameters.base_radius_body + 0.003,
        muscle_amplitudes=parameters.muscle_force_amplitudes,
        muscle_time_delays=muscle_time_delays,
        muscle_sides=muscle_sides,
    )
    for muscle_rod in muscle_rods:
        simulator.operate(muscle_rod).using(
            eaj.AnalyticalLinearDamperJax,
            time_step=np.float64(parameters.time_step),
            damping_constant=parameters.nu_muscle,
        )


def _register_muscle_connections(
    simulator: MuscularSnakeSimulator,
    snake_body: ea.CosseratRod,
    muscle_rods: list[ea.CosseratRod],
    muscle_end_connection_index: list[int],
    *,
    body_elem_length: float,
    parameters: MuscularSnakeParameters,
) -> None:
    """Bind each muscle rod side-by-side to the snake body with surface joints."""
    for muscle_idx, muscle_rod in enumerate(muscle_rods):
        (
            rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame,
            offset_btw_rods,
        ) = get_connection_vector_straight_straight_rod(
            snake_body,
            muscle_rod,
            (
                parameters.muscle_start_connection_index[muscle_idx],
                muscle_end_connection_index[muscle_idx],
            ),
            (0, muscle_rod.n_elems),
        )
        element_indices = np.arange(muscle_rod.n_elems)
        stiffness_scale = np.ones(muscle_rod.n_elems) * 2.0
        stiffness_scale[12:27] = 0.01 * 5.0
        repulsive_scale = np.ones(muscle_rod.n_elems) * 20.0
        body_offset = parameters.muscle_start_connection_index[muscle_idx]
        body_element_index = element_indices + body_offset
        muscle_element_index = element_indices
        radius_body = snake_body.radius[body_element_index]
        radius_muscle = muscle_rod.radius[muscle_element_index]
        connection_stiffness = (
            radius_body
            * radius_muscle
            / (radius_body + radius_muscle)
            * body_elem_length
            * parameters.youngs_modulus
            / (radius_body + radius_muscle)
        )
        simulator.pairwise_interaction(snake_body, muscle_rod).using(
            SurfaceJointSideBySideJax,
            k=connection_stiffness * stiffness_scale,
            nu=1e-4,
            k_repulsive=connection_stiffness * repulsive_scale,
            rod_one_direction_vec_in_material_frame=(
                rod_one_direction_vec_in_material_frame[..., element_indices]
            ),
            rod_two_direction_vec_in_material_frame=(
                rod_two_direction_vec_in_material_frame[..., element_indices]
            ),
            offset_btw_rods=offset_btw_rods[element_indices],
            body_element_index=body_element_index,
            muscle_element_index=muscle_element_index,
        )
