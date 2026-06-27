"""GPU-oriented Position Verlet timestepper."""

pass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..memory_block.sharded_cosserat_rod_jax import (
    _ShardedCosseratRodBlock,
    is_sharded_block_state,
)
from ..protocol import JAXBlock, JAXSystems, JAXPyTree


class PositionVerletJAX:
    """Dedicated Position Verlet integrator for device-backed systems."""

    def __init__(self) -> None:
        self._compiled_rollout_cache: dict[tuple[Any, ...], Any] = {}

    @staticmethod
    def _body_fn(
        _,
        carry,
        *,
        systems: tuple[JAXBlock, ...],
        system_collection: JAXSystems,
        dt_jax: jax.Array,
        half_dt_jax: jax.Array,
    ):
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

    @staticmethod
    def _rollout_uses_sharded_state(states: tuple[JAXPyTree, ...]) -> bool:
        return any(
            isinstance(state, dict) and is_sharded_block_state(state)
            for state in states
        )

    def _get_compiled_rollout(
        self,
        system_collection: JAXSystems,
        systems: tuple[JAXBlock, ...],
        n_steps: int,
        simulation_time: np.float64,
        simulation_dt: np.float64,
        reference_dtype: np.dtype,
        *,
        states_only_args: bool,
    ):
        cache_key = (
            id(system_collection),
            n_steps,
            float(simulation_time),
            float(simulation_dt),
            reference_dtype,
            states_only_args,
        )
        if cache_key in self._compiled_rollout_cache:
            return self._compiled_rollout_cache[cache_key]

        time_jax = jnp.asarray(simulation_time, dtype=reference_dtype)
        dt_jax = jnp.asarray(simulation_dt, dtype=reference_dtype)
        half_dt_jax = jnp.asarray(0.5 * simulation_dt, dtype=reference_dtype)

        def body_fn(step_idx: int, carry):  # type: ignore[no-untyped-def]
            return self._body_fn(
                step_idx,
                carry,
                systems=systems,
                system_collection=system_collection,
                dt_jax=dt_jax,
                half_dt_jax=half_dt_jax,
            )

        if states_only_args:

            def rollout(
                states: tuple[JAXPyTree, ...],
            ) -> tuple[jax.Array, tuple[JAXPyTree, ...]]:
                carry_time = time_jax
                carry_states = states
                for _ in range(n_steps):
                    carry_time, carry_states = self._body_fn(
                        0,
                        (carry_time, carry_states),
                        systems=systems,
                        system_collection=system_collection,
                        dt_jax=dt_jax,
                        half_dt_jax=half_dt_jax,
                    )
                return carry_time, carry_states

            compiled_rollout = rollout
            self._compiled_rollout_cache[cache_key] = compiled_rollout
            return compiled_rollout

        def rollout(
            time_arg: jax.Array,
            states: tuple[JAXPyTree, ...],
            dt_arg: jax.Array,
            half_dt_arg: jax.Array,
        ) -> tuple[jax.Array, tuple[JAXPyTree, ...]]:
            del dt_arg, half_dt_arg
            return jax.lax.fori_loop(0, n_steps, body_fn, (time_arg, states))

        compiled_rollout = jax.jit(rollout)
        self._compiled_rollout_cache[cache_key] = compiled_rollout
        return compiled_rollout

    @staticmethod
    def _reference_device_from_states(
        systems: tuple[JAXBlock, ...],
        states: tuple[JAXPyTree, ...],
    ) -> jax.Device:
        first_system = systems[0]
        first_state = states[0]
        if isinstance(first_state, dict) and is_sharded_block_state(first_state):
            for leaf in jax.tree_util.tree_leaves(first_state["shards"][0]):
                if hasattr(leaf, "devices"):
                    return next(iter(leaf.devices()))
                if hasattr(leaf, "device"):
                    return leaf.device

        if isinstance(first_system, _ShardedCosseratRodBlock):
            if first_system.mesh.is_sharded:
                return first_system.mesh.devices[0]
            return first_system._primary_block.position_collection_device.device

        if hasattr(first_system, "position_collection_device"):
            return first_system.position_collection_device.device

        for leaf in jax.tree_util.tree_leaves(states):
            if hasattr(leaf, "devices"):
                return next(iter(leaf.devices()))
            if hasattr(leaf, "device"):
                return leaf.device

        return jax.devices()[0]

    @staticmethod
    def _reference_dtype_from_states(
        states: tuple[JAXPyTree, ...],
    ) -> np.dtype:
        first_state = states[0]
        if isinstance(first_state, dict) and is_sharded_block_state(first_state):
            for leaf in jax.tree_util.tree_leaves(first_state["shards"][0]):
                if hasattr(leaf, "dtype"):
                    return np.dtype(leaf.dtype)
        for leaf in jax.tree_util.tree_leaves(states):
            if hasattr(leaf, "dtype"):
                return np.dtype(leaf.dtype)
        return np.dtype(np.float64)

    def integrate(
        self,
        SystemCollection: JAXSystems,
        time: float,
        final_time: float,
        dt: float,
    ) -> float:
        """
        Integrate Position Verlet steps from ``time`` to ``final_time`` with step ``dt``.
        """
        assert dt > 0.0, "dt must be positive."
        assert final_time >= time, "final_time must be greater than or equal to time."

        simulation_time = np.float64(time)
        target_time = np.float64(final_time)
        simulation_dt = np.float64(dt)
        duration = float(target_time - simulation_time)
        n_steps = int(np.round(duration / float(simulation_dt)))
        assert np.isclose(
            simulation_time + n_steps * simulation_dt, target_time
        ), "final_time - time must be an integer multiple of dt."

        systems = tuple(SystemCollection.final_systems())
        states = tuple(system.jax_get_state() for system in systems)
        reference_dtype = self._reference_dtype_from_states(states)
        states_only_args = self._rollout_uses_sharded_state(states)
        compiled_rollout = self._get_compiled_rollout(
            system_collection=SystemCollection,
            systems=systems,
            n_steps=n_steps,
            simulation_time=simulation_time,
            simulation_dt=simulation_dt,
            reference_dtype=reference_dtype,
            states_only_args=states_only_args,
        )
        if states_only_args:
            final_time_jax, final_states = compiled_rollout(states)
        else:
            reference_device = self._reference_device_from_states(systems, states)
            dt_jax = jax.device_put(
                np.asarray(simulation_dt, dtype=reference_dtype),
                device=reference_device,
            )
            half_dt_jax = jax.device_put(
                np.asarray(0.5 * simulation_dt, dtype=reference_dtype),
                device=reference_device,
            )
            time_jax = jax.device_put(
                np.asarray(simulation_time, dtype=reference_dtype),
                device=reference_device,
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
