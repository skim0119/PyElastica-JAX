"""Sinusoidal muscle actuation block operator for the muscular snake case."""

from __future__ import annotations

import numpy as np

import elastica_jax as eaj
from elastica_jax._calculus import difference_kernel

import jax
import jax.numpy as jnp

_MUSCLE_BELLY_START_ELEM = 4 * 3
_MUSCLE_BELLY_END_ELEM = 9 * 3


class MuscularSnakeMuscleForcesBlockJax(eaj.NoBlockOpJax):
    r"""Traveling-wave muscle actuation on every rod in the muscle memory block.

    Each muscle rod contracts along its tangent with a force whose magnitude
    follows a sinusoidal traveling wave, phase-delayed head-to-tail (Zhang et
    al., Nat. Commun. 2019).
    """

    def __init__(
        self,
        *,
        wave_number: float,
        arm_length: float,
        muscle_amplitudes: np.ndarray,
        muscle_time_delays: np.ndarray,
        muscle_sides: np.ndarray,
        _system=None,
    ) -> None:
        assert _system is not None, (
            "MuscularSnakeMuscleForcesBlockJax requires a finalized block."
        )
        block = _system
        self.wave_number = np.float64(wave_number)
        self.muscle_specs: list[dict[str, object]] = []
        for muscle_idx in range(block.n_rods):
            node_start = int(block.start_idx_in_rod_nodes[muscle_idx])
            actuation_start = node_start + _MUSCLE_BELLY_START_ELEM
            actuation_end = node_start + _MUSCLE_BELLY_END_ELEM
            elem_start = int(block.start_idx_in_rod_elems[muscle_idx])
            self.muscle_specs.append(
                {
                    "elem_slice": slice(
                        elem_start + _MUSCLE_BELLY_START_ELEM,
                        elem_start + _MUSCLE_BELLY_END_ELEM,
                    ),
                    "node_slice": slice(actuation_start, actuation_end + 1),
                    "amplitude": np.float64(
                        float(muscle_amplitudes[muscle_idx]) / arm_length
                    ),
                    "time_delay": np.float64(muscle_time_delays[muscle_idx]),
                    "side_of_body": np.int64(muscle_sides[muscle_idx]),
                }
            )

    def jax_block_operate_synchronize(self, state, time):
        external_forces = state["external_forces"]
        tangents = state["tangents"]
        for spec in self.muscle_specs:
            external_forces = _apply_muscle_forces(
                amplitude=spec["amplitude"],
                wave_number=self.wave_number,
                side_of_body=spec["side_of_body"],
                time_value=time,
                time_delay=spec["time_delay"],
                elem_slice=spec["elem_slice"],
                node_slice=spec["node_slice"],
                tangents=tangents,
                external_forces=external_forces,
            )
        return {**state, "external_forces": external_forces}


def _apply_muscle_forces(
    *,
    amplitude: jax.Array,
    wave_number: jax.Array,
    side_of_body: jax.Array,
    time_value: jax.Array,
    time_delay: jax.Array,
    elem_slice: slice,
    node_slice: slice,
    tangents: jax.Array,
    external_forces: jax.Array,
) -> jax.Array:
    real_time = time_value - time_delay
    ramp = jnp.where(real_time <= 0.0, 0.0, jnp.minimum(1.0, real_time))
    muscle_forces = (
        ramp
        * amplitude
        * (side_of_body * 0.5 * jnp.sin(wave_number * real_time) + 0.5)
        * tangents[:, elem_slice]
    )
    nodal_forces = difference_kernel(muscle_forces)
    return external_forces.at[:, node_slice].add(nodal_forces)
