from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Type, cast

import numpy as np

from elastica_jax.operations import NoOpsJax
from elastica_jax.memory_block.protocol import RodBlockProtocol, RodView
from elastica_jax.protocol import JAXPyTree, JAXScalar
from elastica.typing import SystemIdxType

from elastica.modules.protocol import ModuleProtocol, SystemCollectionProtocol

_STAGE_METHODS = (
    ("constrain_values", "jax_operate_constrain_values"),
    ("synchronize", "jax_operate_synchronize"),
    ("constrain_rates", "jax_operate_constrain_rates"),
)

type RodTarget = tuple[int, int, RodBlockProtocol]

JAXStageCallable = Callable[..., tuple[JAXPyTree, ...]]


class JAXBasicMixins:
    """Shared JAX stage transforms mixed into system-collection modules."""

    _feature_group_synchronize: Any
    _feature_group_constrain_values: Any
    _feature_group_constrain_rates: Any
    _feature_group_damping: Any

    def jax_synchronize(
        self,
        states: tuple[JAXPyTree, ...],
        time: JAXScalar,
    ) -> tuple[JAXPyTree, ...]:
        for func in self._feature_group_synchronize:
            states = func(states=states, time=time)
        return states

    def jax_constrain_values(
        self,
        states: tuple[JAXPyTree, ...],
        time: JAXScalar,
    ) -> tuple[JAXPyTree, ...]:
        for func in self._feature_group_constrain_values:
            states = func(states=states, time=time)
        return states

    def jax_constrain_rates(
        self,
        states: tuple[JAXPyTree, ...],
        time: JAXScalar,
    ) -> tuple[JAXPyTree, ...]:
        for func in chain(
            self._feature_group_constrain_rates,
            self._feature_group_damping,
        ):
            states = func(states=states, time=time)
        return states

    @staticmethod
    def _find_rod_target(
        final_systems: tuple[object, ...],
        sys_id: SystemIdxType,
    ) -> RodTarget:
        for block_idx, system in enumerate(final_systems):
            if not callable(getattr(system, "rod_view", None)):
                continue
            block = cast(RodBlockProtocol, system)
            matches = np.flatnonzero(block.system_idx_list == sys_id)
            if matches.size:
                return block_idx, int(matches[0]), block
        raise RuntimeError(
            "JAX operator target rod was not found in finalized JAX blocks. "
            "Enable JAX block support for the rod type before finalize()."
        )


class JAXOps(JAXBasicMixins, SystemCollectionProtocol):
    """
    Register pure JAX rod-local operators and expose JAX stage transforms.
    """

    _jax_ops_list: list[ModuleProtocol]

    def __init__(self) -> None:
        self._jax_ops_list = []
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_ops)

    def operate(self, system: Any) -> ModuleProtocol:
        sys_idx = self.get_system_index(system)
        jax_op: ModuleProtocol = _JAXOp(sys_idx)
        self._jax_ops_list.append(jax_op)
        return jax_op

    @classmethod
    def _wrap_jax_rod_operator(
        cls,
        *,
        target: RodTarget,
        operator: Any,
    ) -> JAXStageCallable:
        block_idx, rod_idx, block = target

        def apply(
            *,
            states: tuple[JAXPyTree, ...],
            time: JAXScalar,
        ) -> tuple[JAXPyTree, ...]:
            block_state = states[block_idx]
            rod_view = block.rod_view(block_state, rod_idx)
            updated_view = operator(rod_view, time)
            committed_state = (
                updated_view.commit()
                if isinstance(updated_view, RodView)
                else rod_view.commit()
            )
            updated_states = list(states)
            updated_states[block_idx] = committed_state
            return tuple(updated_states)

        return apply

    def _finalize_jax_ops(self) -> None:
        final_systems = tuple(self.final_systems())
        self._jax_ops_list.sort(key=lambda x: x.id())

        for jax_op in self._jax_ops_list:
            sys_id = jax_op.id()
            op_instance = jax_op.instantiate(self[sys_id])
            target = self._find_rod_target(final_systems, sys_id)
            staged_wrappers = []
            for stage, method_name in _STAGE_METHODS:
                method = getattr(op_instance, method_name, None)
                if method is None:
                    continue
                if getattr(type(op_instance), method_name) is getattr(
                    NoOpsJax, method_name
                ):
                    continue
                wrapped = self._wrap_jax_rod_operator(
                    target=target,
                    operator=method,
                )
                staged_wrappers.append((stage, wrapped))

            assert staged_wrappers, (
                f"{type(op_instance)} does not define any JAX stage methods. "
                "Implement at least one of "
                "`jax_operate_constrain_values`, "
                "`jax_operate_synchronize`, or "
                "`jax_operate_constrain_rates`."
            )

            for stage, wrapped in staged_wrappers:
                stage_group = self._stage_group(stage)
                stage_group.append_id(jax_op)
                stage_group.add_operators(jax_op, [wrapped])

        self._jax_ops_list = []
        del self._jax_ops_list

    def _stage_group(self, stage: str) -> Any:
        assert stage in (
            "constrain_values",
            "synchronize",
            "constrain_rates",
        ), f"Unsupported JAX operator stage {stage!r}."
        if stage == "constrain_values":
            return self._feature_group_constrain_values
        if stage == "synchronize":
            return self._feature_group_synchronize
        return self._feature_group_constrain_rates


class _JAXOp:
    def __init__(self, sys_idx: SystemIdxType) -> None:
        self._sys_idx = sys_idx
        self._op_cls: Type[Any]
        self._args: Any
        self._kwargs: Any

    def using(
        self,
        cls: Type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert issubclass(cls, NoOpsJax), (
            f"{cls} is not a valid JAX operator. It must derive from NoOpsJax."
        )
        self._op_cls = cls
        self._args = args
        self._kwargs = kwargs

    def id(self) -> SystemIdxType:
        return self._sys_idx

    def instantiate(self, system: Any) -> Any:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX operator provided. Did you forget to call "
                "`simulator.operate(system).using(...)`?"
            )
        return self._op_cls(*self._args, _system=system, **self._kwargs)
