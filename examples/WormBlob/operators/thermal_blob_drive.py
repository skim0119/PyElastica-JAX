"""Thermal-gradient active forcing for the worm-blob case."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline

import elastica_jax as eaj

from block_utils import gather_vector_batch, uniform_index_matrix
from role_map import role_weights_jax


class WarmBlobDriveJax(eaj.NoBlockOpJax):
    r"""Role-differentiated active drive set by a thermal ``x`` gradient.

    Each rod samples its centroid ``x`` and is assigned smooth ``(pull, wiggle,
    binder)`` role weights (see :func:`role_map.role_weights_jax`). Two channels
    are superposed:

    - **Pull** (cold side): a steady body force of magnitude
      ``pull_force_scale`` toward the cold ``-x`` direction, scaled by the
      puller weight and distributed over rod nodes by mass fraction.
    - **Snake actuation** (hot side): a traveling-wave internal torque couple
      along ``torque_axis``, matching the ActiveMatter snake-pit form

      .. math::

          m(s, t) = 0.5 \, r(t) \, A(s) \, \sin\!\left(
              \frac{2\pi t}{T} - \frac{2\pi s}{\lambda}
          \right)

      where ``A(s)`` is the cubic-spline envelope, ``T`` the actuation
      ``time_period``, ``lambda`` the ``wave_length``, and ``r(t)`` a
      sinusoidal temporal ramp. The torque field is rotated into each element
      material frame and applied as an equal-and-opposite couple across
      neighboring elements, then scaled by the per-rod wiggler weight and
      ``actuation_amplitude_scale``.

    Parameters
    ----------
    cold_x : float
        World ``x`` of the coldest edge (puller side).
    hot_x : float
        World ``x`` of the hottest edge (wiggler side).
    pull_force_scale : float
        Total pull force magnitude per rod at full puller weight.
    time_period : float
        Actuation period ``T`` in seconds.
    wave_length : float
        Traveling-wave length ``lambda`` in normalized arc coordinate.
    activation_start_time_nd : float
        Non-dimensional delay before the sinusoidal ramp begins, in units of
        ``time_period``.
    actuation_amplitude_scale : float
        Multiplier on the spline envelope (and thus peak torque).
    torque_axis : numpy.ndarray
        Unit world axis (shape ``(3,)``) for the actuation torque.
    n_rods : int
        Number of rods in the packed block.
    n_elements : int
        Elements per rod.
    spline_amplitude : numpy.ndarray
        Per-element actuation envelope, shape ``(n_elements,)``.
    """

    def __init__(
        self,
        *,
        cold_x: float,
        hot_x: float,
        pull_force_scale: float,
        time_period: float,
        wave_length: float,
        activation_start_time_nd: float,
        actuation_amplitude_scale: float,
        torque_axis: np.ndarray,
        n_rods: int,
        n_elements: int,
        spline_amplitude: np.ndarray,
        _system,
    ) -> None:
        assert _system is not None, "WarmBlobDriveJax requires a finalized block."
        self.node_indices = jnp.asarray(
            uniform_index_matrix(
                _system.start_idx_in_rod_nodes,
                _system.end_idx_in_rod_nodes,
            )
        )
        offsets = np.arange(n_elements, dtype=np.int32)
        self.element_indices = (
            _system.start_idx_in_rod_elems[:, None].astype(np.int32)
            + offsets[None, :]
        )
        self.pull_force_scale = pull_force_scale
        self.time_period = time_period
        self.wave_length = wave_length
        self.activation_start_time_nd = activation_start_time_nd
        self.actuation_amplitude_scale = actuation_amplitude_scale
        self.cold_x = cold_x
        self.hot_x = hot_x
        self.torque_axis = torque_axis
        self.n_rods = n_rods
        self.n_elements = n_elements
        self.spline_amplitude = spline_amplitude
        self.pull_direction = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    def jax_block_operate_synchronize(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        positions = gather_vector_batch(state["position_collection"], self.node_indices)
        rod_center_x = jnp.mean(positions, axis=2)[:, 0]
        pull_weight, wiggle_weight, _binder_weight = role_weights_jax(
            rod_center_x,
            cold_x=float(self.cold_x),
            hot_x=float(self.hot_x),
        )

        node_mass = jnp.take(state["mass"], self.node_indices, axis=-1)
        mass_fraction = node_mass / jnp.sum(node_mass, axis=1, keepdims=True)
        pull_force = (
            self.pull_force_scale
            * pull_weight[:, None, None]
            * self.pull_direction[None, :, None]
            * mass_fraction[:, None, :]
        )
        external_forces = (
            state["external_forces"]
            .at[:, self.node_indices]
            .add(jnp.moveaxis(pull_force, 0, 1))
        )

        external_torques = _snake_actuation_torques(
            state=state,
            element_indices=self.element_indices,
            n_rods=self.n_rods,
            n_elements=self.n_elements,
            torque_axis=self.torque_axis,
            spline_amplitude=self.spline_amplitude,
            time_period=self.time_period,
            wave_length=self.wave_length,
            activation_start_time_nd=self.activation_start_time_nd,
            actuation_amplitude_scale=self.actuation_amplitude_scale,
            wiggle_weight=wiggle_weight,
            time=time,
        )
        return {
            **state,
            "external_forces": external_forces,
            "external_torques": external_torques,
        }


def spline_actuation_amplitude(n_elements: int) -> np.ndarray:
    """Return the cubic-spline actuation envelope sampled at element centers.

    Uses the same control points as the ActiveMatter snake-pit case.
    """
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


def _snake_actuation_torques(
    *,
    state: dict[str, jax.Array],
    element_indices: np.ndarray,
    n_rods: int,
    n_elements: int,
    torque_axis: np.ndarray,
    spline_amplitude: np.ndarray,
    time_period: float,
    wave_length: float,
    activation_start_time_nd: float,
    actuation_amplitude_scale: float,
    wiggle_weight: jax.Array,
    time: np.float64,
) -> jax.Array:
    """Apply traveling-wave torque couples, scaled by per-rod wiggle weight."""
    external_torques = state["external_torques"]

    directors = jnp.moveaxis(
        state["director_collection"][:, :, element_indices.reshape(-1)], 2, 0
    )
    directors = directors.reshape(n_rods, n_elements, 3, 3)

    s = (jnp.arange(n_elements) + 0.5) / n_elements
    wave = jnp.sin(2.0 * jnp.pi * time / time_period - 2.0 * jnp.pi * s / wave_length)
    start = activation_start_time_nd * time_period
    phase = jnp.clip((time - start) / time_period, 0.0, 1.0)
    ramp_factor = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
    torque_magnitude = (
        0.5
        * ramp_factor
        * actuation_amplitude_scale
        * spline_amplitude
        * wave
    )

    torque_world = torque_magnitude[None, None, :] * torque_axis[None, :, None]
    torque_world = jnp.broadcast_to(torque_world, (n_rods, 3, n_elements))
    torque_world = torque_world * wiggle_weight[:, None, None]
    torque_field = jnp.einsum("neij,nje->nei", directors, torque_world)

    torque_couple = jnp.zeros_like(torque_field)
    torque_couple = torque_couple.at[:, 1:, :].add(torque_field[:, 1:, :])
    torque_couple = torque_couple.at[:, :-1, :].add(-torque_field[:, 1:, :])
    return external_torques.at[:, element_indices].add(
        jnp.moveaxis(torque_couple, -1, 0)
    )
