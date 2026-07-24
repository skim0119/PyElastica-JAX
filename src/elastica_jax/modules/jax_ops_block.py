from __future__ import annotations

from collections.abc import Callable
from typing import Any, Type

from elastica_jax.block_operation import NoBlockOpJax
from elastica_jax.operations import NoOpsJax
from elastica_jax.memory_block.memory_block_rod_jax import (
    _CosseratRodMemoryBlock,
)
from elastica_jax.memory_block.memory_block_rod_vertical_jax import (
    _CosseratRodVerticalMemoryBlock,
)
from elastica_jax.memory_block.mpi_cosserat_rod_jax import (
    _MpiCosseratRodBlock,
)
from elastica_jax.memory_block.protocol import RodBlockProtocol

from elastica.modules.protocol import ModuleProtocol, SystemCollectionProtocol
from .jax_ops import JAXBasicMixins

_JAX_ROD_BLOCK_TYPES = (
    _CosseratRodMemoryBlock,
    _CosseratRodVerticalMemoryBlock,
    _MpiCosseratRodBlock,
)
JAXRodBlockSystem = (
    _CosseratRodMemoryBlock | _CosseratRodVerticalMemoryBlock | _MpiCosseratRodBlock
)

# Stage hooks registered during finalize(). Each row is:
# (stage, block-wide method, block-native per-rod method, single-rod method).
# The single-rod column reuses ``NoOpsJax`` / ``jax_operate_*`` kernels from
# ``operate()`` by instantiating one operator per rod and batching through
# ``Block.map_rods`` (same path as ``jax_per_rod_operate_*``).
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
      (receive authoritative packed/stacked Block state as-is)
    - block per-rod operators implementing ``jax_per_rod_operate_*``
      (batched via ``Block.map_rods``)
    - single-rod operators (``NoOpsJax`` / ``jax_operate_*``) reused on blocks
      via ``operate_block``; one operator instance is created per rod so
      constructors can capture rod-local state from ``_system``
    """

    _jax_block_ops_list: list[_JAXBlockOp]

    def __init__(self) -> None:
        self._jax_block_ops_list = []
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_block_ops)

    def operate_block(self, target: RodBlockProtocol | Type[Any]) -> ModuleProtocol:
        jax_op = _JAXBlockOp(target)
        self._jax_block_ops_list.append(jax_op)
        return jax_op

    @classmethod
    def _wrap_jax_block_operator(
        cls,
        *,
        block_state_idx: int,
        operator: Any,
    ) -> Callable[..., tuple[Any, ...]]:
        def apply(*, states, time):  # type: ignore[no-untyped-def]
            block_state = states[block_state_idx]
            updated_state = operator(block_state, time)
            updated_states = list(states)
            updated_states[block_state_idx] = updated_state
            return tuple(updated_states)

        return apply

    @classmethod
    def _wrap_jax_per_rod_operator(
        cls,
        *,
        block_state_idx: int,
        block_system: RodBlockProtocol,
        operators: Any | tuple[Any, ...],
    ) -> Callable[..., tuple[Any, ...]]:
        def apply(*, states, time):  # type: ignore[no-untyped-def]
            block_state = states[block_state_idx]
            updated_block_state = block_system.map_rods(
                block_state,
                operators,
                time,
            )
            updated_states = list(states)
            updated_states[block_state_idx] = updated_block_state
            return tuple(updated_states)

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
                packed_rods = block_system.rods()
                per_rod_operators = tuple(
                    jax_op.instantiate(rod) for rod in packed_rods
                )
                instantiate_target = packed_rods[0]
            op_instance = jax_op.instantiate(instantiate_target)

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
                        operator=operator,
                    )
                    staged_wrappers.append((stage, wrapped))
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
            if any(isinstance(subsystem, target_type) for subsystem in system.rods()):
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
