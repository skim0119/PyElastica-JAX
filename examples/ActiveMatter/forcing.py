"""Shared forcing helpers for active-matter JAX examples."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

import elastica_jax as eaj


def spline_actuation_amplitude(n_elements: int) -> np.ndarray:
    spline_x = np.array(
        [0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85, 0.93]
    )
    spline_y = np.array(
        [
            0.003394,
            0.003474,
            0.003604,
            0.003697,
            0.003673,
            0.003525,
            0.003330,
            0.003170,
            0.003126,
            0.003219,
            0.003371,
            0.003489,
        ]
    )
    centers = (np.arange(n_elements) + 0.5) / n_elements
    return CubicSpline(spline_x, spline_y)(centers)


def apply_gravity_and_actuation(
    *,
    state,
    parameters,
    element_indices: np.ndarray,
    n_snakes: int,
    n_elements: int,
    gravity_axis: np.ndarray,
    spline_amplitude: np.ndarray,
    time,
    ramp: str,
):
    """Apply nodal gravity and traveling-wave torque actuation to block state."""
    dtype = state["position_collection"].dtype
    gravity = parameters.gravitational_acc * jnp.asarray(gravity_axis, dtype=dtype)
    external_forces = state["mass"][None, :] * gravity[:, None]
    external_torques = jnp.zeros_like(state["external_torques"])

    elem = jnp.asarray(element_indices)
    directors = jnp.moveaxis(state["director_collection"][:, :, elem.reshape(-1)], 2, 0)
    directors = directors.reshape(n_snakes, n_elements, 3, 3)

    s = jnp.arange(n_elements, dtype=dtype) + 0.5
    s /= n_elements
    wave = jnp.sin(
        2.0 * jnp.pi * time / parameters.time_period
        - 2.0 * jnp.pi * s / parameters.wave_length
    )
    if ramp == "sinusoidal":
        start = parameters.activation_start_time_nd * parameters.time_period
        phase = jnp.clip((time - start) / parameters.time_period, 0.0, 1.0)
        ramp_factor = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
        torque_magnitude = (
            0.5 * ramp_factor * jnp.asarray(spline_amplitude, dtype=dtype) * wave
        )
    elif ramp == "linear":
        ramp_factor = jnp.minimum(1.0, time / parameters.time_period)
        torque_magnitude = (
            ramp_factor * jnp.asarray(spline_amplitude, dtype=dtype) * wave
        )
    else:
        raise ValueError(f"Unsupported actuation ramp {ramp!r}.")

    torque_world = (
        torque_magnitude[None, None, :]
        * jnp.asarray(gravity_axis, dtype=dtype)[None, :, None]
    )
    torque_world = jnp.broadcast_to(torque_world, (n_snakes, 3, n_elements))
    torque_field = jnp.einsum("neij,nje->nei", directors, torque_world)
    torque_couple = jnp.zeros_like(torque_field)
    torque_couple = torque_couple.at[:, 1:, :].add(torque_field[:, 1:, :])
    torque_couple = torque_couple.at[:, :-1, :].add(-torque_field[:, 1:, :])
    external_torques = external_torques.at[:, element_indices].add(
        jnp.moveaxis(torque_couple, -1, 0)
    )
    return external_forces, external_torques


class ActiveMatterForcingJax(eaj.NoBlockOpJax):
    """Gravity and traveling-wave actuation without contact."""

    def __init__(
        self,
        *,
        parameters,
        n_snakes: int,
        n_elements: int,
        gravity_axis: np.ndarray,
        spline_amplitude: np.ndarray,
        ramp: str,
        _system=None,
    ) -> None:
        assert _system is not None, "ActiveMatterForcingJax requires a finalized block."
        block = _system
        offsets = np.arange(n_elements, dtype=np.int32)
        self.element_indices = (
            block.start_idx_in_rod_elems[:, None].astype(np.int32) + offsets[None, :]
        )
        self.parameters = parameters
        self.n_snakes = n_snakes
        self.n_elements = n_elements
        self.gravity_axis = gravity_axis
        self.spline_amplitude = spline_amplitude
        self.ramp = ramp

    def jax_block_operate_synchronize(self, state, time):
        external_forces, external_torques = apply_gravity_and_actuation(
            state=state,
            parameters=self.parameters,
            element_indices=self.element_indices,
            n_snakes=self.n_snakes,
            n_elements=self.n_elements,
            gravity_axis=self.gravity_axis,
            spline_amplitude=self.spline_amplitude,
            time=time,
            ramp=self.ramp,
        )
        updated = dict(state)
        updated["external_forces"] = external_forces
        updated["external_torques"] = external_torques
        return updated
