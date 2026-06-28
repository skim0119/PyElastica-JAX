"""GPU-oriented Position Verlet timestepper."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..memory_block.sharded_cosserat_rod_jax import (
    SHARDED_STATE_KEY,
    _ShardedCosseratRodBlock,
)
from ..protocol import (
    JAXBlock,
    JAXBlockExecution,
    JAXBlockStages,
    JAXPyTree,
    JAXSystems,
)


class PositionVerletJAX:
    """Dedicated Position Verlet integrator for device-backed systems."""

    def __init__(self) -> None:
        self._compiled_rollout_cache: dict[tuple[Any, ...], Any] = {}

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

    @staticmethod
    def _apply_stage(
        state: JAXPyTree,
        time: jax.Array,
        operators: tuple[Any, ...],
    ) -> JAXPyTree:
        for operator in operators:
            state = operator(state, time)
        return state

    @classmethod
    def _block_body_fn(
        cls,
        _: int,
        carry: tuple[jax.Array, JAXPyTree],
        *,
        system: JAXBlock,
        stages: JAXBlockStages,
        dt_jax: jax.Array,
        half_dt_jax: jax.Array,
    ) -> tuple[jax.Array, JAXPyTree]:
        step_time, state = carry
        state = system.jax_kinematic_step(state, step_time, half_dt_jax)
        step_time = step_time + half_dt_jax
        state = cls._apply_stage(state, step_time, stages.constrain_values)
        state = system.jax_compute_internal_forces_and_torques(state, step_time)
        state = cls._apply_stage(state, step_time, stages.synchronize)
        state = system.jax_dynamic_step(state, step_time, dt_jax)
        state = cls._apply_stage(state, step_time, stages.constrain_rates)
        state = system.jax_kinematic_step(state, step_time, half_dt_jax)
        step_time = step_time + half_dt_jax
        state = cls._apply_stage(state, step_time, stages.constrain_values)
        state = system.jax_zero_external_loads(state, step_time)
        return step_time, state

    @staticmethod
    def _devices_in_state(state: JAXPyTree) -> set[Any]:
        devices: set[Any] = set()
        if isinstance(state, dict) and state.get(SHARDED_STATE_KEY, False):
            for shard_state in state["shards"]:
                for leaf in jax.tree_util.tree_leaves(shard_state):
                    if hasattr(leaf, "devices"):
                        devices.update(leaf.devices())
                    elif hasattr(leaf, "device"):
                        devices.add(leaf.device)
            return devices

        for leaf in jax.tree_util.tree_leaves(state):
            if hasattr(leaf, "devices"):
                devices.update(leaf.devices())
            elif hasattr(leaf, "device"):
                devices.add(leaf.device)
        return devices

    @classmethod
    def _state_spans_multiple_devices(cls, states: tuple[JAXPyTree, ...]) -> bool:
        rollout_devices: set[Any] = set()
        for state in states:
            rollout_devices.update(cls._devices_in_state(state))
        return len(rollout_devices) > 1

    def _get_compiled_rollout(
        self,
        system_collection: JAXSystems,
        systems: tuple[JAXBlock, ...],
        n_steps: int,
        simulation_dt: np.float64,
        reference_dtype: np.dtype,
    ) -> Callable[..., tuple[jax.Array, tuple[JAXPyTree, ...]]]:
        cache_key = (
            id(system_collection),
            n_steps,
            float(simulation_dt),
            reference_dtype,
        )
        if cache_key in self._compiled_rollout_cache:
            return self._compiled_rollout_cache[cache_key]

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

        def rollout(
            time_arg: jax.Array,
            states: tuple[JAXPyTree, ...],
        ) -> tuple[jax.Array, tuple[JAXPyTree, ...]]:
            return jax.lax.fori_loop(0, n_steps, body_fn, (time_arg, states))

        compiled_rollout = jax.jit(rollout)
        self._compiled_rollout_cache[cache_key] = compiled_rollout
        return compiled_rollout

    def _get_compiled_block_rollout(
        self,
        system: JAXBlock,
        stages: JAXBlockStages,
        n_steps: int,
        simulation_dt: np.float64,
        reference_dtype: np.dtype,
    ) -> Callable[[jax.Array, JAXPyTree], tuple[jax.Array, JAXPyTree]]:
        stage_key = tuple(
            id(operator)
            for operators in (
                stages.constrain_values,
                stages.synchronize,
                stages.constrain_rates,
            )
            for operator in operators
        )
        cache_key = (
            "block",
            id(system),
            stage_key,
            n_steps,
            float(simulation_dt),
            reference_dtype,
        )
        if cache_key in self._compiled_rollout_cache:
            return self._compiled_rollout_cache[cache_key]

        dt_jax = jnp.asarray(simulation_dt, dtype=reference_dtype)
        half_dt_jax = jnp.asarray(0.5 * simulation_dt, dtype=reference_dtype)

        def body_fn(step_idx: int, carry):  # type: ignore[no-untyped-def]
            return self._block_body_fn(
                step_idx,
                carry,
                system=system,
                stages=stages,
                dt_jax=dt_jax,
                half_dt_jax=half_dt_jax,
            )

        def rollout(
            time_arg: jax.Array,
            state: JAXPyTree,
        ) -> tuple[jax.Array, JAXPyTree]:
            return jax.lax.fori_loop(0, n_steps, body_fn, (time_arg, state))

        compiled_rollout = jax.jit(rollout)
        self._compiled_rollout_cache[cache_key] = compiled_rollout
        return compiled_rollout

    def _integrate_independent_blocks(
        self,
        *,
        systems: tuple[JAXBlock, ...],
        states: tuple[JAXPyTree, ...],
        executions: tuple[JAXBlockExecution, ...],
        n_steps: int,
        simulation_time: np.float64,
        simulation_dt: np.float64,
    ) -> tuple[jax.Array, tuple[JAXPyTree, ...]]:
        assert len(executions) == len(systems), (
            "Independent block execution metadata must match finalized systems."
        )
        final_times: list[jax.Array] = []
        final_states: list[JAXPyTree] = []

        for system, state, execution in zip(systems, states, executions, strict=True):
            if execution.shard_stages is not None:
                assert isinstance(system, _ShardedCosseratRodBlock), (
                    "Shard execution stages require a sharded block."
                )
                shard_states = state["shards"]
                assert len(execution.shard_stages) == len(shard_states), (
                    "Shard execution metadata must match block state shards."
                )
                updated_shards = []
                for shard_system, shard_state, shard_stages in zip(
                    system._shard_blocks,
                    shard_states,
                    execution.shard_stages,
                    strict=True,
                ):
                    shard_time, updated_shard = self._run_compiled_block(
                        system=shard_system,
                        state=shard_state,
                        stages=shard_stages,
                        n_steps=n_steps,
                        simulation_time=simulation_time,
                        simulation_dt=simulation_dt,
                    )
                    final_times.append(shard_time)
                    updated_shards.append(updated_shard)
                final_states.append(
                    {SHARDED_STATE_KEY: True, "shards": tuple(updated_shards)}
                )
                continue

            block_time, updated_state = self._run_compiled_block(
                system=system,
                state=state,
                stages=execution.stages,
                n_steps=n_steps,
                simulation_time=simulation_time,
                simulation_dt=simulation_dt,
            )
            final_times.append(block_time)
            final_states.append(updated_state)

        assert final_times, "At least one JAX block is required for integration."
        return final_times[0], tuple(final_states)

    def _run_compiled_block(
        self,
        *,
        system: JAXBlock,
        state: JAXPyTree,
        stages: JAXBlockStages,
        n_steps: int,
        simulation_time: np.float64,
        simulation_dt: np.float64,
    ) -> tuple[jax.Array, JAXPyTree]:
        reference_dtype = self._reference_dtype_from_states((state,))
        reference_device = self._reference_device_from_states((system,), (state,))
        time_jax = jax.device_put(
            np.asarray(simulation_time, dtype=reference_dtype),
            device=reference_device,
        )
        compiled_rollout = self._get_compiled_block_rollout(
            system=system,
            stages=stages,
            n_steps=n_steps,
            simulation_dt=simulation_dt,
            reference_dtype=reference_dtype,
        )
        return compiled_rollout(time_jax, state)

    @staticmethod
    def _reference_device_from_states(
        systems: tuple[JAXBlock, ...],
        states: tuple[JAXPyTree, ...],
    ) -> Any:
        first_system = systems[0]
        first_state = states[0]
        if isinstance(first_state, dict) and first_state.get(SHARDED_STATE_KEY, False):
            for leaf in jax.tree_util.tree_leaves(first_state["shards"][0]):
                if hasattr(leaf, "devices"):
                    return next(iter(leaf.devices()))
                if hasattr(leaf, "device"):
                    return leaf.device

        if isinstance(first_system, _ShardedCosseratRodBlock):
            return first_system._devices[0]

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
        if isinstance(first_state, dict) and first_state.get(SHARDED_STATE_KEY, False):
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

        Parameters
        ----------
        SystemCollection
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
        n_steps = int(np.round(duration / float(simulation_dt)))
        assert np.isclose(simulation_time + n_steps * simulation_dt, target_time), (
            "final_time - time must be an integer multiple of dt."
        )

        systems = tuple(SystemCollection.final_systems())
        assert systems, "SystemCollection must contain at least one JAX block."
        states = tuple(system.jax_get_state() for system in systems)
        reference_dtype = self._reference_dtype_from_states(states)
        spans_multiple_devices = self._state_spans_multiple_devices(states)
        independent_executions = None
        if spans_multiple_devices and hasattr(
            SystemCollection, "jax_independent_block_executions"
        ):
            independent_executions = SystemCollection.jax_independent_block_executions()
        if independent_executions is not None:
            final_time_jax, final_states = self._integrate_independent_blocks(
                systems=systems,
                states=states,
                executions=independent_executions,
                n_steps=n_steps,
                simulation_time=simulation_time,
                simulation_dt=simulation_dt,
            )
            for system, state in zip(systems, final_states, strict=True):
                system.jax_set_state(state)
            return float(final_time_jax)

        assert not spans_multiple_devices, (
            "Multi-device rollout requires block-local execution metadata. "
            "Cross-block coupled operations are not supported by the jitted "
            "PositionVerletJAX rollout."
        )

        compiled_rollout = self._get_compiled_rollout(
            system_collection=SystemCollection,
            systems=systems,
            n_steps=n_steps,
            simulation_dt=simulation_dt,
            reference_dtype=reference_dtype,
        )
        reference_device = self._reference_device_from_states(systems, states)
        time_jax = jax.device_put(
            np.asarray(simulation_time, dtype=reference_dtype),
            device=reference_device,
        )
        final_time_jax, final_states = compiled_rollout(time_jax, states)

        for system, state in zip(systems, final_states):
            system.jax_set_state(state)

        return float(final_time_jax)
