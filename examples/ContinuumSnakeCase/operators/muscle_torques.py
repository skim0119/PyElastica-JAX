"""Gravity and continuum-snake muscle-torque actuation as a JAX block operator."""

from __future__ import annotations

import numpy as np

import elastica as ea
import elastica_jax as eaj
from elastica_jax._linalg import _jax_batch_matvec as _batch_matvec

import jax
import jax.numpy as jnp


class SnakeMuscleTorquesJax(eaj.NoOpsJax):
    """Apply gravity and travelling-wave muscle torques to a single JAX rod."""

    def __init__(
        self,
        *,
        b_coeff: np.ndarray,
        period: float,
        base_length: float,
        gravitational_acc: float,
        _system,
    ) -> None:
        torque_template = ea.MuscleTorques(
            base_length=base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / float(b_coeff[-1]),
            phase_shift=0.0,
            direction=np.array([0.0, 1.0, 0.0]),
            rest_lengths=_system.rest_lengths,
            ramp_up_time=period,
            with_spline=True,
        )
        self.gravity = np.asarray([0.0, gravitational_acc, 0.0], dtype=np.float64)
        self.muscle_direction = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        self.muscle_s = np.asarray(torque_template.s, dtype=np.float64)
        self.muscle_spline = np.asarray(torque_template.my_spline, dtype=np.float64)
        self.muscle_angular_frequency = np.float64(2.0 * np.pi / period)
        self.muscle_wave_number = np.float64(2.0 * np.pi / float(b_coeff[-1]))
        self.muscle_phase_shift = np.float64(0.0)
        self.muscle_ramp_up_time = np.float64(period)

    def jax_operate_synchronize(self, rod_view, time):
        external_forces, external_torques = _apply_gravity_and_muscle_torques(
            time_value=time,
            director_collection=rod_view.director_collection,
            mass=rod_view.mass,
            gravity=self.gravity,
            muscle_direction=self.muscle_direction,
            muscle_s=self.muscle_s,
            muscle_spline=self.muscle_spline,
            muscle_angular_frequency=self.muscle_angular_frequency,
            muscle_wave_number=self.muscle_wave_number,
            muscle_phase_shift=self.muscle_phase_shift,
            muscle_ramp_up_time=self.muscle_ramp_up_time,
        )
        rod_view.external_forces = external_forces
        rod_view.external_torques = external_torques
        return rod_view


def _apply_gravity_and_muscle_torques(
    *,
    time_value: jax.Array,
    director_collection: jax.Array,
    mass: jax.Array,
    gravity: jax.Array,
    muscle_direction: jax.Array,
    muscle_s: jax.Array,
    muscle_spline: jax.Array,
    muscle_angular_frequency: jax.Array,
    muscle_wave_number: jax.Array,
    muscle_phase_shift: jax.Array,
    muscle_ramp_up_time: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return nodal external forces and torques from gravity and muscle actuation.

    The muscle torque magnitude follows the travelling-wave spline used by
    :class:`elastica.MuscleTorques`, ramped linearly up to ``muscle_ramp_up_time``.

    Parameters
    ----------
    time_value
        Current simulation time.
    director_collection
        Element director collection of shape ``(3, 3, n_elem)``.
    mass
        Nodal mass array of shape ``(n_nodes,)``.
    gravity
        Gravitational acceleration vector of shape ``(3,)``.
    muscle_direction
        Actuation direction in the material frame, shape ``(3,)``.
    muscle_s
        Normalized arc-length samples of the spline, shape ``(n_elem,)``.
    muscle_spline
        Spline torque amplitudes, shape ``(n_elem,)``.
    muscle_angular_frequency, muscle_wave_number, muscle_phase_shift
        Travelling-wave parameters.
    muscle_ramp_up_time
        Linear ramp-up duration for the actuation.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Nodal external forces ``(3, n_nodes)`` and torques ``(3, n_elem)``.
    """
    external_forces = gravity[:, None] * mass[None, :]
    external_torques = jnp.zeros(
        (3, director_collection.shape[2]), dtype=director_collection.dtype
    )

    factor = jnp.minimum(1.0, time_value / muscle_ramp_up_time)
    torque_mag = (
        factor
        * muscle_spline
        * jnp.sin(
            muscle_angular_frequency * time_value
            - muscle_wave_number * muscle_s
            + muscle_phase_shift
        )
    )
    torque = muscle_direction[:, None] * torque_mag[::-1][None, :]
    torque_world = _batch_matvec(director_collection, torque)

    external_torques = external_torques.at[:, 1:].add(torque_world[:, 1:])
    external_torques = external_torques.at[:, :-1].add(
        -_batch_matvec(director_collection[:, :, :-1], torque[:, 1:])
    )
    return external_forces, external_torques
