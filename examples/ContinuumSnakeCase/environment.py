"""Simulator construction for the reduced continuum snake GPU validation case.

This module holds the simulator mixin definitions and the ``build_simulation`` /
``build_reference_simulation`` factories shared by the run script. The JAX
simulation and the host-side PyElastica reference are built from the same
:class:`SnakeParameters` so their final states can be compared.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import elastica as ea
import elastica_jax as eaj

import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from operators import SnakeMuscleTorquesJax, SnakePlaneContactJax

if TYPE_CHECKING:
    from run_continuum_snake_gpu import SnakeParameters


class SnakeJAXSimulator(eaj.Simulator):
    """Device-side simulator collection for the reduced snake case."""


class SnakeReferenceSimulator(
    ea.BaseSystemCollection, ea.Forcing, ea.Damping, ea.Contact
):
    """Host-side PyElastica reference simulator used for validation."""


def build_rod(parameters: SnakeParameters) -> ea.CosseratRod:
    """Create the straight Cosserat rod shared by both simulators."""
    return ea.CosseratRod.straight_rod(
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


def build_simulation(
    parameters: SnakeParameters,
    *,
    device: jax.Device | str,
) -> tuple[SnakeJAXSimulator, eaj._CosseratRodMemoryBlock]:
    """Build and finalize the JAX snake simulator.

    Parameters
    ----------
    parameters
        Physical and numerical parameters for the case.
    device
        JAX device (or device string) the rod block is allocated on.

    Returns
    -------
    tuple[SnakeJAXSimulator, elastica_jax._CosseratRodMemoryBlock]
        The finalized simulator and its single rod memory block.
    """
    rod_block = eaj.configure_rod_block(device=device or "cpu")

    simulator = SnakeJAXSimulator()
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    rod = build_rod(parameters)
    simulator.append(rod)

    simulator.operate(rod).using(
        SnakeMuscleTorquesJax,
        b_coeff=parameters.b_coeff_array,
        period=parameters.period,
        base_length=parameters.base_length,
        gravitational_acc=parameters.gravitational_acc,
    )
    simulator.operate(rod).using(
        SnakePlaneContactJax,
        plane_origin=parameters.plane_origin,
        plane_normal=parameters.plane_normal,
        slip_velocity_tol=parameters.slip_velocity_tol,
        k=parameters.contact_k,
        nu=parameters.contact_nu,
        static_mu_array=parameters.static_mu_array,
        kinetic_mu_array=parameters.kinetic_mu_array,
    )
    simulator.operate(rod).using(
        eaj.AnalyticalLinearDamperJax,
        time_step=np.float64(parameters.time_step),
        damping_constant=parameters.damping_constant,
    )
    simulator.finalize()
    return simulator, rod_block


def build_reference_simulation(
    parameters: SnakeParameters,
) -> tuple[SnakeReferenceSimulator, ea.CosseratRod]:
    """Build and finalize the host-side PyElastica reference simulator."""
    simulator = SnakeReferenceSimulator()
    rod = build_rod(parameters)
    simulator.append(rod)

    simulator.add_forcing_to(rod).using(
        ea.GravityForces,
        acc_gravity=np.array([0.0, parameters.gravitational_acc, 0.0]),
    )
    simulator.add_forcing_to(rod).using(
        ea.MuscleTorques,
        base_length=parameters.base_length,
        b_coeff=parameters.b_coeff_array[:-1],
        period=parameters.period,
        wave_number=parameters.wave_number,
        phase_shift=0.0,
        rest_lengths=rod.rest_lengths,
        ramp_up_time=parameters.period,
        direction=parameters.plane_normal,
        with_spline=True,
    )
    ground_plane = ea.Plane(
        plane_origin=parameters.plane_origin,
        plane_normal=parameters.plane_normal,
    )
    simulator.append(ground_plane)
    simulator.detect_contact_between(rod, ground_plane).using(
        ea.RodPlaneContactWithAnisotropicFriction,
        k=parameters.contact_k,
        nu=parameters.contact_nu,
        slip_velocity_tol=parameters.slip_velocity_tol,
        static_mu_array=parameters.static_mu_array,
        kinetic_mu_array=parameters.kinetic_mu_array,
    )
    simulator.dampen(rod).using(
        ea.AnalyticalLinearDamper,
        damping_constant=parameters.damping_constant,
        time_step=parameters.time_step,
    )
    simulator.finalize()
    return simulator, rod
