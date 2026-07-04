from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, Type

from elastica_jax.block_operation import NoBlockOpJax
from elastica_jax.operations import NoOpsJax
from elastica_jax.memory_block.memory_block_rod_jax import (
    _CosseratRodMemoryBlock,
    _ELEMENT_ATTRS,
    _NODE_ATTRS,
    _SYNCABLE_ATTRS,
    _VORONOI_ATTRS,
)
from elastica_jax.memory_block.mpi_cosserat_rod_jax import (
    _MpiCosseratRodBlock,
)
from elastica_jax.memory_block.sharded_cosserat_rod_jax import (
    SHARDED_STATE_KEY,
    _ShardedCosseratRodBlock,
)
from elastica_jax.protocol import JAXBlockExecution, JAXBlockStages
import jax
import jax.numpy as jnp
import numpy as np

from elastica.modules.protocol import ModuleProtocol, SystemCollectionProtocol
from .jax_ops import JAXBasicMixins


# Stage hooks registered during finalize(). Each row is:
# (stage, block-wide method, block-native per-rod method, single-rod method).
# The single-rod column reuses ``NoOpsJax`` / ``jax_operate_*`` kernels from
# ``operate()`` by instantiating one operator per rod and batching through the
# same gather/scatter path as ``jax_per_rod_operate_*``.
_BLOCK_STAGE_METHODS = (
    (
        "constrain_values",
        "jax_block_operate_constrain_values",
        "jax_per_rod_operate_constrain_values",
        "jax_operate_constrain_values",
    ),
    (
        "synchronize",
        "jax_block_operate_synchronize",
        "jax_per_rod_operate_synchronize",
        "jax_operate_synchronize",
    ),
    (
        "constrain_rates",
        "jax_block_operate_constrain_rates",
        "jax_per_rod_operate_constrain_rates",
        "jax_operate_constrain_rates",
    ),
)


class JAXBlockOpTarget(Protocol):
    """
    Configured rod block registered with ``enable_block_supports``.

    Pass the same instance to ``operate_block`` before ``finalize()``.
    After ``finalize()``, that object is the built block in ``final_systems()``.
    """

    def __call__(
        self,
        systems: list[Any],
        system_idx_list: list[Any],
    ) -> Any: ...


class _PerRodStateView:
    def __init__(
        self, state: dict[str, Any], *, updates: dict[str, Any] | None = None
    ) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_updates", {} if updates is None else dict(updates))

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        return self._updates.get(attr, self._state[attr])

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_"):
            object.__setattr__(self, attr, value)
            return
        self._updates[attr] = value

    def commit(self) -> dict[str, Any]:
        updated = dict(self._state)
        updated.update(self._updates)
        return updated


class JAXOpsBlock(JAXBasicMixins, SystemCollectionProtocol):
    """
    Register pure JAX block operations and expose JAX stage transforms.

    User code normally interacts with this mixin through:

    ```python
    rod_block = eaj.configure_rod_block()
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    simulator.operate_block(rod_block).using(MyOp, ...)
    ```

    where `MyOp` derives from `eaj.NoBlockOpJax`. Pass the same block
    instance registered with ``enable_block_supports``.

    `JAXOpsBlock` supports three execution styles:

    - block-native operators implementing ``jax_block_operate_*``
    - block per-rod operators implementing ``jax_per_rod_operate_*``
    - single-rod operators (``NoOpsJax`` / ``jax_operate_*``) reused on blocks
      via ``operate_block``; one operator instance is created per rod so
      constructors can capture rod-local state from ``_system``

    Block per-rod and single-rod paths share the same gather/scatter batching
    during ``finalize()``. Block-native operators run directly on the packed
    block state.
    """

    _jax_block_ops_list: list[_JAXBlockOp]

    def __init__(self) -> None:
        self._jax_block_ops_list = []
        self._jax_local_block_stages: dict[int, dict[str, list[Any]]] = {}
        self._jax_local_shard_stages: dict[int, tuple[dict[str, list[Any]], ...]] = {}
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_block_ops)

    def operate_block(self, target: JAXBlockOpTarget | Type[Any]) -> ModuleProtocol:
        jax_op = _JAXBlockOp(target)
        self._jax_block_ops_list.append(jax_op)
        return jax_op

    @classmethod
    def _wrap_jax_block_operator(
        cls,
        *,
        block_state_idx: int,
        block_system: _CosseratRodMemoryBlock | _ShardedCosseratRodBlock,
        operator: Any,
    ) -> Callable[..., tuple[Any, ...]]:
        def apply(*, states, time):  # type: ignore[no-untyped-def]
            block_state = states[block_state_idx]
            if block_state.get(SHARDED_STATE_KEY, False):
                assert isinstance(block_system, _ShardedCosseratRodBlock)
                merged = block_system.merge_shard_states(block_state)
                updated_state = block_system.scatter_merged_state(
                    operator(merged, time),
                    block_state,
                )
            else:
                updated_state = operator(block_state, time)
            updated_states = list(states)
            updated_states[block_state_idx] = updated_state
            return tuple(updated_states)

        return apply

    @staticmethod
    def _wrap_local_jax_block_operator(
        operator: Any,
    ) -> Callable[[dict[str, Any], Any], dict[str, Any]]:
        def apply(state: dict[str, Any], time: Any) -> dict[str, Any]:
            return operator(state, time)

        return apply

    @staticmethod
    def _uniform_index_matrix(
        start_idx: np.ndarray,
        end_idx: np.ndarray,
    ) -> np.ndarray:
        widths = end_idx - start_idx
        assert np.all(widths == widths[0]), (
            "Per-rod JAX block operators require uniform discretization across rods."
        )
        offsets = np.arange(int(widths[0]), dtype=np.int32)
        return start_idx[:, None].astype(np.int32) + offsets[None, :]

    @classmethod
    def _per_rod_indices(
        cls,
        block_system: _CosseratRodMemoryBlock | _ShardedCosseratRodBlock,
    ) -> dict[str, jax.Array]:
        return {
            "node": jnp.asarray(
                cls._uniform_index_matrix(
                    block_system.start_idx_in_rod_nodes,
                    block_system.end_idx_in_rod_nodes,
                )
            ),
            "element": jnp.asarray(
                cls._uniform_index_matrix(
                    block_system.start_idx_in_rod_elems,
                    block_system.end_idx_in_rod_elems,
                )
            ),
            "voronoi": jnp.asarray(
                cls._uniform_index_matrix(
                    block_system.start_idx_in_rod_voronoi,
                    block_system.end_idx_in_rod_voronoi,
                )
            ),
        }

    @staticmethod
    def _gather_attr(array: jax.Array, indices: jax.Array) -> jax.Array:
        if array.ndim == 1:
            return jnp.take(array, indices, axis=-1)
        if array.ndim == 2:
            return jnp.moveaxis(jnp.take(array, indices, axis=-1), 1, 0)
        if array.ndim == 3:
            return jnp.moveaxis(jnp.take(array, indices, axis=-1), 2, 0)
        raise ValueError(f"Unsupported array rank {array.ndim} for per-rod batching.")

    @staticmethod
    def _scatter_attr(
        array: jax.Array, indices: jax.Array, values: jax.Array
    ) -> jax.Array:
        if array.ndim == 1:
            return array.at[indices].set(values)
        if array.ndim == 2:
            return array.at[:, indices].set(jnp.moveaxis(values, 0, 1))
        if array.ndim == 3:
            return array.at[:, :, indices].set(jnp.moveaxis(values, 0, 2))
        raise ValueError(f"Unsupported array rank {array.ndim} for per-rod batching.")

    @classmethod
    def _apply_per_rod_operator_to_block_state(
        cls,
        *,
        block_system: _CosseratRodMemoryBlock | _ShardedCosseratRodBlock,
        block_state: dict[str, Any],
        operators: Any | tuple[Any, ...],
        time: Any,
    ) -> dict[str, Any]:
        attr_domains = (
            {attr: "node" for attr in _NODE_ATTRS}
            | {attr: "element" for attr in _ELEMENT_ATTRS}
            | {attr: "voronoi" for attr in _VORONOI_ATTRS}
        )
        attrs = tuple(_SYNCABLE_ATTRS)
        operator_list = (operators,) if not isinstance(operators, tuple) else operators

        def commit_view(
            updated_view: Any, rod_view: _PerRodStateView
        ) -> dict[str, Any]:
            if isinstance(updated_view, _PerRodStateView):
                return updated_view.commit()
            if hasattr(updated_view, "commit"):
                return updated_view.commit()
            return rod_view.commit()

        def apply_to_working_state(
            working_state: dict[str, Any],
            indices: dict[str, jax.Array],
            *,
            per_rod_operators: tuple[Any, ...],
        ) -> dict[str, Any]:
            local_state = {
                attr: cls._gather_attr(working_state[attr], indices[attr_domains[attr]])
                for attr in attrs
            }
            updated_rods: list[dict[str, Any]] = []
            for rod_index, operator in enumerate(per_rod_operators):
                single_state = {attr: local_state[attr][rod_index] for attr in attrs}
                rod_view = _PerRodStateView(single_state)
                updated_view = operator(rod_view, time)
                updated_rods.append(commit_view(updated_view, rod_view))
            updated_local_state = {
                attr: jnp.stack([rod[attr] for rod in updated_rods], axis=0)
                for attr in attrs
            }
            updated_block_state = dict(working_state)
            for attr in attrs:
                updated_block_state[attr] = cls._scatter_attr(
                    working_state[attr],
                    indices[attr_domains[attr]],
                    updated_local_state[attr],
                )
            return updated_block_state

        if block_state.get(SHARDED_STATE_KEY, False):
            assert isinstance(block_system, _ShardedCosseratRodBlock)
            updated_shards = []
            rod_offset = 0
            for inner_block, shard_state in zip(
                block_system._shard_blocks, block_state["shards"]
            ):
                shard_operators = operator_list[
                    rod_offset : rod_offset + inner_block.n_rods
                ]
                shard_indices = cls._per_rod_indices(inner_block)
                updated_shards.append(
                    apply_to_working_state(
                        shard_state,
                        shard_indices,
                        per_rod_operators=shard_operators,
                    )
                )
                rod_offset += inner_block.n_rods
            return {SHARDED_STATE_KEY: True, "shards": tuple(updated_shards)}

        indices = cls._per_rod_indices(block_system)
        return apply_to_working_state(
            block_state,
            indices,
            per_rod_operators=operator_list,
        )

    @classmethod
    def _wrap_jax_per_rod_operator(
        cls,
        *,
        block_state_idx: int,
        block_system: _CosseratRodMemoryBlock | _ShardedCosseratRodBlock,
        operators: Any | tuple[Any, ...],
    ) -> Callable[..., tuple[Any, ...]]:
        def apply(*, states, time):  # type: ignore[no-untyped-def]
            block_state = states[block_state_idx]
            updated_block_state = cls._apply_per_rod_operator_to_block_state(
                block_system=block_system,
                block_state=block_state,
                operators=operators,
                time=time,
            )
            updated_states = list(states)
            updated_states[block_state_idx] = updated_block_state
            return tuple(updated_states)

        return apply

    @classmethod
    def _wrap_local_jax_per_rod_operator(
        cls,
        *,
        block_system: _CosseratRodMemoryBlock,
        operators: Any | tuple[Any, ...],
    ) -> Callable[[dict[str, Any], Any], dict[str, Any]]:
        def apply(state: dict[str, Any], time: Any) -> dict[str, Any]:
            return cls._apply_per_rod_operator_to_block_state(
                block_system=block_system,
                block_state=state,
                operators=operators,
                time=time,
            )

        return apply

    def _finalize_jax_block_ops(self) -> None:
        final_systems = tuple(self.final_systems())

        for jax_op in self._jax_block_ops_list:
            block_state_idx, block_system = self._find_target_block(
                final_systems,
                jax_op.target(),
            )
            staged_wrappers = []
            instantiate_with_block = False
            instantiate_per_rod_single_ops = False
            for (
                stage,
                block_method_name,
                per_rod_method_name,
                single_rod_method_name,
            ) in _BLOCK_STAGE_METHODS:
                op_cls = jax_op.operator_cls()
                has_block = hasattr(op_cls, block_method_name) and getattr(
                    op_cls, block_method_name
                ) is not getattr(NoBlockOpJax, block_method_name)
                has_per_rod = hasattr(op_cls, per_rod_method_name) and getattr(
                    op_cls, per_rod_method_name
                ) is not getattr(NoBlockOpJax, per_rod_method_name)
                has_single_rod = hasattr(op_cls, single_rod_method_name) and getattr(
                    op_cls, single_rod_method_name
                ) is not getattr(NoOpsJax, single_rod_method_name)

                assert not (has_block and (has_per_rod or has_single_rod)), (
                    f"{op_cls} mixes block and per-rod JAX block operator "
                    f"implementations for stage {stage!r}. Choose one style per stage."
                )

                if has_block:
                    instantiate_with_block = True
                    continue

                if has_per_rod:
                    instantiate_with_block = True
                    continue

                if has_single_rod:
                    instantiate_per_rod_single_ops = True
                    continue

            assert not (
                instantiate_with_block and instantiate_per_rod_single_ops
            ), (
                f"{jax_op.operator_cls()} mixes block-style and single-rod JAX "
                "block operator methods. Use one constructor contract."
            )

            instantiate_target: Any = block_system
            shard_op_instances: tuple[Any, ...] = ()
            shard_per_rod_operators: tuple[tuple[Any, ...], ...] = ()
            per_rod_operators: tuple[Any, ...] | None = None
            if instantiate_per_rod_single_ops:
                if isinstance(block_system, _ShardedCosseratRodBlock):
                    shard_per_rod_operators = tuple(
                        tuple(jax_op.instantiate(rod) for rod in shard._systems)
                        for shard in block_system._shard_blocks
                    )
                    per_rod_operators = tuple(
                        operator
                        for shard_operators in shard_per_rod_operators
                        for operator in shard_operators
                    )
                else:
                    per_rod_operators = tuple(
                        jax_op.instantiate(rod) for rod in block_system._systems
                    )
                instantiate_target = block_system._systems[0]
            op_instance = jax_op.instantiate(instantiate_target)
            if (
                isinstance(block_system, _ShardedCosseratRodBlock)
                and instantiate_with_block
            ):
                shard_op_instances = tuple(
                    jax_op.instantiate(shard) for shard in block_system._shard_blocks
                )

            local_stages = self._jax_local_block_stages.setdefault(
                block_state_idx,
                {stage: [] for stage, *_ in _BLOCK_STAGE_METHODS},
            )
            shard_stages: tuple[dict[str, list[Any]], ...] = ()
            if isinstance(block_system, _ShardedCosseratRodBlock):
                shard_stages = self._jax_local_shard_stages.setdefault(
                    block_state_idx,
                    tuple(
                        {stage: [] for stage, *_ in _BLOCK_STAGE_METHODS}
                        for _ in block_system._shard_blocks
                    ),
                )

            for (
                stage,
                block_method_name,
                per_rod_method_name,
                single_rod_method_name,
            ) in _BLOCK_STAGE_METHODS:
                has_block = hasattr(type(op_instance), block_method_name) and getattr(
                    type(op_instance), block_method_name
                ) is not getattr(NoBlockOpJax, block_method_name)
                has_per_rod = hasattr(
                    type(op_instance), per_rod_method_name
                ) and getattr(type(op_instance), per_rod_method_name) is not getattr(
                    NoBlockOpJax, per_rod_method_name
                )
                has_single_rod = hasattr(
                    type(op_instance), single_rod_method_name
                ) and getattr(type(op_instance), single_rod_method_name) is not getattr(
                    NoOpsJax, single_rod_method_name
                )

                if has_block:
                    operator = getattr(op_instance, block_method_name)
                    wrapped = self._wrap_jax_block_operator(
                        block_state_idx=block_state_idx,
                        block_system=block_system,
                        operator=operator,
                    )
                    staged_wrappers.append((stage, wrapped))
                    if isinstance(block_system, _ShardedCosseratRodBlock):
                        for shard_stage_map, shard_instance in zip(
                            shard_stages, shard_op_instances, strict=True
                        ):
                            shard_stage_map[stage].append(
                                self._wrap_local_jax_block_operator(
                                    getattr(shard_instance, block_method_name)
                                )
                            )
                    else:
                        local_stages[stage].append(
                            self._wrap_local_jax_block_operator(operator)
                        )
                    continue

                if has_per_rod or has_single_rod:
                    method_name = (
                        per_rod_method_name if has_per_rod else single_rod_method_name
                    )
                    operators: Any | tuple[Any, ...]
                    if has_single_rod and per_rod_operators is not None:
                        operators = tuple(
                            getattr(op, method_name) for op in per_rod_operators
                        )
                    else:
                        operators = getattr(op_instance, method_name)
                    wrapped = self._wrap_jax_per_rod_operator(
                        block_state_idx=block_state_idx,
                        block_system=block_system,
                        operators=operators,
                    )
                    staged_wrappers.append((stage, wrapped))
                    if isinstance(block_system, _ShardedCosseratRodBlock):
                        for shard_index, (shard_stage_map, shard_block) in enumerate(
                            zip(
                                shard_stages,
                                block_system._shard_blocks,
                                strict=True,
                            )
                        ):
                            if has_single_rod:
                                shard_operators = tuple(
                                    getattr(operator, method_name)
                                    for operator in shard_per_rod_operators[shard_index]
                                )
                            else:
                                shard_operators = getattr(
                                    shard_op_instances[shard_index], method_name
                                )
                            shard_stage_map[stage].append(
                                self._wrap_local_jax_per_rod_operator(
                                    block_system=shard_block,
                                    operators=shard_operators,
                                )
                            )
                    else:
                        local_stages[stage].append(
                            self._wrap_local_jax_per_rod_operator(
                                block_system=block_system,
                                operators=operators,
                            )
                        )

            assert staged_wrappers, (
                f"{type(op_instance)} does not define any JAX block stage methods. "
                "Implement at least one block-stage method, "
                "`jax_per_rod_operate_*`, or single-rod `jax_operate_*`."
            )

            for stage, wrapped in staged_wrappers:
                stage_group = self._stage_group(stage)
                stage_group.append_id(jax_op)
                stage_group.add_operators(jax_op, [wrapped])

        self._jax_block_ops_list = []
        del self._jax_block_ops_list

    def jax_independent_block_executions(
        self,
    ) -> tuple[JAXBlockExecution, ...] | None:
        """
        Return block-local execution stages when no cross-block operators exist.

        Returns
        -------
        tuple[JAXBlockExecution, ...] | None
            One execution description per finalized block. ``None`` indicates that
            at least one registered stage operator couples block states.
        """
        stage_groups = {
            "constrain_values": self._feature_group_constrain_values,
            "synchronize": self._feature_group_synchronize,
            "constrain_rates": self._feature_group_constrain_rates,
        }
        registered_counts = {
            stage: sum(
                len(stage_map[stage])
                for stage_map in self._jax_local_block_stages.values()
            )
            + sum(
                len(shard_stage_maps[0][stage])
                for shard_stage_maps in self._jax_local_shard_stages.values()
            )
            for stage in stage_groups
        }
        actual_counts = {
            stage: sum(1 for _ in group) for stage, group in stage_groups.items()
        }
        actual_counts["constrain_rates"] += sum(1 for _ in self._feature_group_damping)
        if registered_counts != actual_counts:
            return None

        executions = []
        for block_state_idx, system in enumerate(self.final_systems()):
            stage_map = self._jax_local_block_stages.get(
                block_state_idx,
                {stage: [] for stage in stage_groups},
            )
            stages = self._make_block_stages(stage_map)
            shard_stage_maps = self._jax_local_shard_stages.get(block_state_idx)
            if shard_stage_maps is None and isinstance(
                system, _ShardedCosseratRodBlock
            ):
                shard_stage_maps = tuple(
                    {stage: [] for stage in stage_groups} for _ in system._shard_blocks
                )
            executions.append(
                JAXBlockExecution(
                    stages=stages,
                    shard_stages=(
                        tuple(
                            self._make_block_stages(item) for item in shard_stage_maps
                        )
                        if shard_stage_maps is not None
                        else None
                    ),
                )
            )
        return tuple(executions)

    @staticmethod
    def _make_block_stages(stage_map: dict[str, list[Any]]) -> JAXBlockStages:
        return JAXBlockStages(
            constrain_values=tuple(stage_map["constrain_values"]),
            synchronize=tuple(stage_map["synchronize"]),
            constrain_rates=tuple(stage_map["constrain_rates"]),
        )

    def _stage_group(self, stage: str):  # type: ignore[no-untyped-def]
        assert stage in (
            "constrain_values",
            "synchronize",
            "constrain_rates",
        ), f"Unsupported JAX block operator stage {stage!r}."
        if stage == "constrain_values":
            return self._feature_group_constrain_values
        if stage == "synchronize":
            return self._feature_group_synchronize
        return self._feature_group_constrain_rates

    @staticmethod
    def _find_target_block(
        final_systems: tuple[Any, ...],
        target: JAXBlockOpTarget | Type[Any],
    ) -> tuple[int, _CosseratRodMemoryBlock | _ShardedCosseratRodBlock | _MpiCosseratRodBlock]:
        if not isinstance(target, type):
            for block_state_idx, system in enumerate(final_systems):
                if system is target:
                    return block_state_idx, system
            raise RuntimeError(
                "Requested JAX block operator target was not found in finalized "
                "block systems. Pass the same block instance registered with "
                "`enable_block_supports(...)`."
            )

        target_type = target
        for block_state_idx, system in enumerate(final_systems):
            if not isinstance(
                system,
                (_CosseratRodMemoryBlock, _ShardedCosseratRodBlock, _MpiCosseratRodBlock),
            ):
                continue
            if isinstance(system, target_type):
                return block_state_idx, system
            if isinstance(system, (_ShardedCosseratRodBlock, _MpiCosseratRodBlock)):
                if target_type in (
                    _CosseratRodMemoryBlock,
                    _ShardedCosseratRodBlock,
                    _MpiCosseratRodBlock,
                ):
                    return block_state_idx, system
            if any(isinstance(subsystem, target_type) for subsystem in system._systems):
                return block_state_idx, system
        raise RuntimeError(
            "Requested JAX block operator target was not found in finalized block systems."
        )


class _JAXBlockOp:
    def __init__(self, target: JAXBlockOpTarget | Type[Any]) -> None:
        self._target = target
        self._op_cls: Type[Any]
        self._args: Any
        self._kwargs: Any

    def using(
        self,
        cls: Type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert issubclass(cls, (NoBlockOpJax, NoOpsJax)), (
            f"{cls} is not a valid JAX block operator. It must derive from "
            "NoBlockOpJax or NoOpsJax (single-rod ops reused on blocks)."
        )
        self._op_cls = cls
        self._args = args
        self._kwargs = kwargs

    def target(self) -> JAXBlockOpTarget | Type[Any]:
        return self._target

    def operator_cls(self) -> Type[Any]:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX block operator provided. Did you forget to call "
                "`simulator.operate_block(...).using(...)`?"
            )
        return self._op_cls

    def id(self) -> Any:
        return self._target

    def instantiate(self, system: Any) -> Any:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX block operator provided. Did you forget to call "
                "`simulator.operate_block(...).using(...)`?"
            )
        return self._op_cls(*self._args, _system=system, **self._kwargs)
