"""GPU-oriented Position Verlet timestepper."""

from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from ..protocol import (
    JAXBlock,
    JAXPyTree,
    JAXSystems,
)


class PositionVerletJAX:
    """
    Dedicated Position Verlet integrator for device-backed systems.

    Notes
    -----
    Integration uses one collection-level rollout over all finalized blocks.
    Device-kernel adaptation belongs to the block implementation; the
    timestepper only invokes the block and collection interfaces.
    """

    def __init__(self) -> None:
        self._compiled_rollout_cache: dict[tuple[Any, ...], Any] = {}

    def integrate(
        self,
        system_collection: JAXSystems,
        *,  # Force user to explicitly define the below items
        time: float,
        final_time: float,
        dt: float,
    ) -> float:
        """
        Integrate Position Verlet steps from ``time`` to ``final_time`` with step ``dt``.

        Parameters
        ----------
        system_collection
            Finalized simulator exposing JAX block and stage transforms.
        time
            Current simulation time.
        final_time
            Target simulation time. Must differ from ``time`` by an integer
            multiple of ``dt``.
        dt
            Fixed Position Verlet step size.

        Returns
        -------
        float
            Final simulation time reached by the rollout.
        """
        assert dt > 0.0, "dt must be positive."
        assert final_time >= time, "final_time must be greater than or equal to time."

        simulation_time = np.float64(time)
        target_time = np.float64(final_time)
        simulation_dt = np.float64(dt)
        duration = float(target_time - simulation_time)
        rounded_steps = np.round(duration / float(simulation_dt))
        assert np.isclose(duration, float(simulation_dt) * rounded_steps), (
            "final_time - time must be an integer multiple of dt."
        )
        n_steps = int(rounded_steps)

        systems = tuple(system_collection.final_systems())
        assert systems, "At least one JAX block is required for integration."
        states = tuple(system.jax_get_state() for system in systems)

        compiled_rollout = self._get_compiled_rollout(
            system_collection=system_collection,
            systems=systems,
            n_steps=n_steps,
        )
        time_jax, dt_jax, half_dt_jax = self._make_step_scalars(
            systems[0],
            simulation_time=simulation_time,
            simulation_dt=simulation_dt,
        )
        final_time_jax, final_states = compiled_rollout(
            time_jax,
            states,
            dt_jax,
            half_dt_jax,
        )

        for system, state in zip(systems, final_states):
            system.jax_set_state(state)

        return float(final_time_jax)

    @staticmethod
    def _body_fn(
        _: int,
        carry: tuple[jax.Array, tuple[JAXPyTree, ...]],
        *,
        systems: tuple[JAXBlock, ...],
        system_collection: JAXSystems,
        dt_jax: jax.Array,
        half_dt_jax: jax.Array,
    ) -> tuple[jax.Array, tuple[JAXPyTree, ...]]:
        step_time, step_states = carry

        step_states = tuple(
            system.jax_kinematic_step(state, step_time, half_dt_jax)
            for system, state in zip(systems, step_states)
        )
        step_time = step_time + half_dt_jax

        step_states = system_collection.jax_constrain_values(step_states, step_time)

        step_states = tuple(
            system.jax_compute_internal_forces_and_torques(state, step_time)
            for system, state in zip(systems, step_states)
        )

        step_states = system_collection.jax_synchronize(step_states, step_time)

        step_states = tuple(
            system.jax_dynamic_step(state, step_time, dt_jax)
            for system, state in zip(systems, step_states)
        )

        step_states = system_collection.jax_constrain_rates(step_states, step_time)

        step_states = tuple(
            system.jax_kinematic_step(state, step_time, half_dt_jax)
            for system, state in zip(systems, step_states)
        )
        step_time = step_time + half_dt_jax

        step_states = system_collection.jax_constrain_values(step_states, step_time)

        step_states = tuple(
            system.jax_zero_external_loads(state, step_time)
            for system, state in zip(systems, step_states)
        )

        return step_time, step_states

    def _get_compiled_rollout(
        self,
        system_collection: JAXSystems,
        systems: tuple[JAXBlock, ...],
        n_steps: int,
    ) -> Callable[..., tuple[jax.Array, tuple[JAXPyTree, ...]]]:
        cache_key = (
            id(system_collection),
            n_steps,
        )
        if cache_key in self._compiled_rollout_cache:
            return self._compiled_rollout_cache[cache_key]

        def body_fn(  # type: ignore[no-untyped-def]
            step_idx: int,
            carry,
            dt_jax: jax.Array,
            half_dt_jax: jax.Array,
        ):
            return self._body_fn(
                step_idx,
                carry,
                systems=systems,
                system_collection=system_collection,
                dt_jax=dt_jax,
                half_dt_jax=half_dt_jax,
            )

        @jax.jit
        def rollout(
            time_arg: jax.Array,
            states: tuple[JAXPyTree, ...],
            dt_jax: jax.Array,
            half_dt_jax: jax.Array,
        ) -> tuple[jax.Array, tuple[JAXPyTree, ...]]:
            step_body = lambda idx, carry: body_fn(idx, carry, dt_jax, half_dt_jax)
            return jax.lax.fori_loop(0, n_steps, step_body, (time_arg, states))

        self._compiled_rollout_cache[cache_key] = rollout
        return rollout

    @staticmethod
    def _make_step_scalars(
        system: JAXBlock,
        *,
        simulation_time: np.float64,
        simulation_dt: np.float64,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        time_jax = system.device_put(float(simulation_time))
        dt_jax = system.device_put(float(simulation_dt))
        half_dt_jax = system.device_put(float(0.5 * simulation_dt))
        return time_jax, dt_jax, half_dt_jax
