from __future__ import annotations

from collections.abc import Callable
from typing import Any, Type

from elastica_jax.block_operation import NoBlockOpJax
from elastica_jax.operations import NoOpsJax
from elastica_jax.memory_block.memory_block_rod_jax import (
    _CosseratRodMemoryBlock,
    _ELEMENT_ATTRS,
    _NODE_ATTRS,
    _SYNCABLE_ATTRS,
    _VORONOI_ATTRS,
)
from elastica_jax.memory_block.memory_block_rod_vertical_jax import (
    _CosseratRodVerticalMemoryBlock,
)
from elastica_jax.memory_block.mpi_cosserat_rod_jax import (
    _MpiCosseratRodBlock,
)
from elastica_jax.memory_block.protocol import RodBlockProtocol
from elastica_jax.protocol import JAXBlockExecution, JAXBlockStages
import jax
import jax.numpy as jnp
import numpy as np

from elastica.modules.protocol import ModuleProtocol, SystemCollectionProtocol
from .jax_ops import JAXBasicMixins

_JAX_ROD_BLOCK_TYPES = (
    _CosseratRodMemoryBlock,
    _CosseratRodVerticalMemoryBlock,
    _MpiCosseratRodBlock,
)
JAXRodBlockSystem = (
    _CosseratRodMemoryBlock
    | _CosseratRodVerticalMemoryBlock
    | _MpiCosseratRodBlock
)

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
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_block_ops)

    def operate_block(self, target: RodBlockProtocol | Type[Any]) -> ModuleProtocol:
        jax_op = _JAXBlockOp(target)
        self._jax_block_ops_list.append(jax_op)
        return jax_op

    @classmethod
    def _vmap_block_operator_over_rods(
        cls,
        operator: Any,
        block_system: JAXRodBlockSystem,
        state: dict[str, Any],
        time: Any,
    ) -> dict[str, Any]:
        """Apply a rod-shaped block operator across every rod with ``vmap``."""
        vmapped = jax.vmap(operator, in_axes=(0, None))
        if cls._is_stacked_layout(block_system):
            return vmapped(state, time)

        indices = cls._per_rod_indices(block_system)
        attr_domains = (
            {attr: "node" for attr in _NODE_ATTRS}
            | {attr: "element" for attr in _ELEMENT_ATTRS}
            | {attr: "voronoi" for attr in _VORONOI_ATTRS}
        )
        attrs = tuple(_SYNCABLE_ATTRS)
        gathered = {
            attr: cls._gather_attr(state[attr], indices[attr_domains[attr]])
            for attr in attrs
        }
        updated_gathered = vmapped(gathered, time)
        updated = dict(state)
        for attr in attrs:
            updated[attr] = cls._scatter_attr(
                state[attr],
                indices[attr_domains[attr]],
                updated_gathered[attr],
            )
        return updated

    @classmethod
    def _wrap_jax_block_operator(
        cls,
        *,
        block_state_idx: int,
        block_system: JAXRodBlockSystem,
        operator: Any,
    ) -> Callable[..., tuple[Any, ...]]:
        def apply(*, states, time):  # type: ignore[no-untyped-def]
            block_state = states[block_state_idx]
            updated_state = cls._vmap_block_operator_over_rods(
                operator, block_system, block_state, time
            )
            updated_states = list(states)
            updated_states[block_state_idx] = updated_state
            return tuple(updated_states)

        return apply

    @classmethod
    def _wrap_local_jax_block_operator(
        cls,
        operator: Any,
        block_system: JAXRodBlockSystem,
    ) -> Callable[[dict[str, Any], Any], dict[str, Any]]:
        def apply(state: dict[str, Any], time: Any) -> dict[str, Any]:
            return cls._vmap_block_operator_over_rods(
                operator, block_system, state, time
            )

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
        block_system: (JAXRodBlockSystem),
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
    def _is_stacked_layout(block_system: object) -> bool:
        """
        Return whether ``block_system`` stores rods on a stacked last axis.

        MPI wrappers delegate to an inner horizontal or vertical block; detect
        that inner layout instead of only the wrapper type.
        """
        if isinstance(block_system, _CosseratRodVerticalMemoryBlock):
            return True
        if isinstance(block_system, _MpiCosseratRodBlock):
            return issubclass(
                block_system.inner_block_cls,
                _CosseratRodVerticalMemoryBlock,
            )
        return False

    @staticmethod
    def _gather_attr(
        array: jax.Array,
        indices: jax.Array,
        *,
        stacked: bool = False,
    ) -> jax.Array:
        if stacked:
            if array.ndim == 2:
                return jnp.take_along_axis(array, indices, axis=-1)
            if array.ndim == 3:
                return jnp.take_along_axis(array, indices[:, None, :], axis=-1)
            if array.ndim == 4:
                return jnp.take_along_axis(array, indices[:, None, None, :], axis=-1)
            raise ValueError(
                f"Unsupported stacked array rank {array.ndim} for per-rod batching."
            )
        if array.ndim == 1:
            return jnp.take(array, indices, axis=-1)
        if array.ndim == 2:
            return jnp.moveaxis(jnp.take(array, indices, axis=-1), 1, 0)
        if array.ndim == 3:
            return jnp.moveaxis(jnp.take(array, indices, axis=-1), 2, 0)
        raise ValueError(f"Unsupported array rank {array.ndim} for per-rod batching.")

    @staticmethod
    def _scatter_attr(
        array: jax.Array,
        indices: jax.Array,
        values: jax.Array,
        *,
        stacked: bool = False,
    ) -> jax.Array:
        if stacked:
            # Vertical blocks gather/scatter the full rod domain on axis -1.
            del array, indices
            return values
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
        block_system: JAXRodBlockSystem,
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
        stacked = cls._is_stacked_layout(block_system)

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
            stacked_layout: bool,
        ) -> dict[str, Any]:
            local_state = {
                attr: cls._gather_attr(
                    working_state[attr],
                    indices[attr_domains[attr]],
                    stacked=stacked_layout,
                )
                for attr in attrs
            }
            n_rods = int(next(iter(local_state.values())).shape[0])
            if len(per_rod_operators) == 1:
                # One jax_per_rod operator instance applies to every rod.
                shared_operator = per_rod_operators[0]
                per_rod_operators = tuple(shared_operator for _ in range(n_rods))
            assert len(per_rod_operators) == n_rods, (
                f"Expected {n_rods} per-rod operators, got {len(per_rod_operators)}."
            )
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
                    stacked=stacked_layout,
                )
            return updated_block_state

        indices = cls._per_rod_indices(block_system)
        return apply_to_working_state(
            block_state,
            indices,
            per_rod_operators=operator_list,
            stacked_layout=stacked,
        )

    @classmethod
    def _wrap_jax_per_rod_operator(
        cls,
        *,
        block_state_idx: int,
        block_system: (JAXRodBlockSystem),
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
        block_system: _CosseratRodMemoryBlock | _MpiCosseratRodBlock,
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

            assert not (instantiate_with_block and instantiate_per_rod_single_ops), (
                f"{jax_op.operator_cls()} mixes block-style and single-rod JAX "
                "block operator methods. Use one constructor contract."
            )

            instantiate_target: Any = block_system
            per_rod_operators: tuple[Any, ...] | None = None
            if instantiate_per_rod_single_ops:
                per_rod_operators = tuple(
                    jax_op.instantiate(rod) for rod in block_system._systems
                )
                instantiate_target = block_system._systems[0]
            op_instance = jax_op.instantiate(instantiate_target)

            local_stages = self._jax_local_block_stages.setdefault(
                block_state_idx,
                {stage: [] for stage, *_ in _BLOCK_STAGE_METHODS},
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
                    local_stages[stage].append(
                        self._wrap_local_jax_block_operator(operator, block_system)
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
            executions.append(JAXBlockExecution(stages=stages))
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
        target: RodBlockProtocol | Type[Any],
    ) -> tuple[int, JAXRodBlockSystem]:
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
            if not isinstance(system, _JAX_ROD_BLOCK_TYPES):
                continue
            if isinstance(system, target_type):
                return block_state_idx, system
            if isinstance(system, _MpiCosseratRodBlock):
                if target_type in _JAX_ROD_BLOCK_TYPES:
                    return block_state_idx, system
            if any(isinstance(subsystem, target_type) for subsystem in system._systems):
                return block_state_idx, system
        raise RuntimeError(
            "Requested JAX block operator target was not found in finalized block systems."
        )


class _JAXBlockOp:
    def __init__(self, target: RodBlockProtocol | Type[Any]) -> None:
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

    def target(self) -> RodBlockProtocol | Type[Any]:
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
