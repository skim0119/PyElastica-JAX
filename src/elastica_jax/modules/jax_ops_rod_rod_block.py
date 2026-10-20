from __future__ import annotations

from typing import Any, Type

from elastica_jax.rod_rod_operation import NoRodRodBlockOpJax
from elastica_jax.memory_block.protocol import JAXBlockState, RodView
from elastica_jax.protocol import JAXPyTree, JAXScalar
from elastica.typing import SystemIdxType, SystemProtocol

from elastica.modules.protocol import SystemCollectionProtocol
from .jax_ops import JAXBasicMixins, JAXStageCallable, RodTarget


class JAXInteraction(JAXBasicMixins, SystemCollectionProtocol):
    """
    Register JAX rod-to-rod interaction operators.

    Pair interactions such as rod-rod contact are applied through rod-local views
    into one or two blocks, regardless of layout. Rods may live in the same block or in
    separate blocks on the same device; cross-device coupling is not supported.
    """

    _jax_rod2rod_ops_list: list[_JAXRodRodOp]

    def __init__(self) -> None:
        self._jax_rod2rod_ops_list = []
        super().__init__()
        self._feature_group_finalize.append(self._finalize_jax_rod2rod_ops)

    def pairwise_interaction(
        self,
        first_rod: SystemProtocol,
        second_rod: SystemProtocol,
    ) -> _JAXRodRodOp:
        first_sys_idx = self.get_system_index(first_rod)
        second_sys_idx = self.get_system_index(second_rod)
        jax_op = _JAXRodRodOp(first_sys_idx, second_sys_idx)
        self._jax_rod2rod_ops_list.append(jax_op)
        return jax_op

    @classmethod
    def _wrap_jax_rod2rod_operator(
        cls,
        *,
        first_target: RodTarget,
        second_target: RodTarget,
        operator: Any,
    ) -> JAXStageCallable:
        first_idx, first_rod_idx, first_block = first_target
        second_idx, second_rod_idx, second_block = second_target
        same_block = first_idx == second_idx

        def apply(
            *,
            states: tuple[JAXPyTree, ...],
            time: JAXScalar,
        ) -> tuple[JAXPyTree, ...]:
            if same_block:
                block_state = states[first_idx]
                shared_updates: JAXBlockState = {}
                rod_one_view = first_block.rod_view(
                    block_state, first_rod_idx, updates=shared_updates
                )
                rod_two_view = second_block.rod_view(
                    block_state, second_rod_idx, updates=shared_updates
                )
                updated_views = operator(rod_one_view, rod_two_view, time)

                if updated_views is None:
                    committed_state = rod_one_view.commit()
                else:
                    updated_rod_one_view, _updated_rod_two_view = updated_views
                    committed_state = (
                        updated_rod_one_view.commit()
                        if isinstance(updated_rod_one_view, RodView)
                        else rod_one_view.commit()
                    )

                updated_states = list(states)
                updated_states[first_idx] = committed_state
                return tuple(updated_states)

            rod_one_state = states[first_idx]
            rod_two_state = states[second_idx]
            rod_one_view = first_block.rod_view(rod_one_state, first_rod_idx)
            rod_two_view = second_block.rod_view(rod_two_state, second_rod_idx)
            updated_views = operator(rod_one_view, rod_two_view, time)

            if updated_views is None:
                committed_rod_one_state = rod_one_view.commit()
                committed_rod_two_state = rod_two_view.commit()
            else:
                updated_rod_one_view, updated_rod_two_view = updated_views
                committed_rod_one_state = (
                    updated_rod_one_view.commit()
                    if isinstance(updated_rod_one_view, RodView)
                    else rod_one_view.commit()
                )
                committed_rod_two_state = (
                    updated_rod_two_view.commit()
                    if isinstance(updated_rod_two_view, RodView)
                    else rod_two_view.commit()
                )

            updated_states = list(states)
            updated_states[first_idx] = committed_rod_one_state
            updated_states[second_idx] = committed_rod_two_state
            return tuple(updated_states)

        return apply

    def _finalize_jax_rod2rod_ops(self) -> None:
        final_systems = tuple(self.final_systems())

        for jax_op in self._jax_rod2rod_ops_list:
            first_target = self._find_rod_target(final_systems, jax_op.first_id())
            second_target = self._find_rod_target(final_systems, jax_op.second_id())
            op_instance = jax_op.instantiate(
                first_system=self[jax_op.first_id()],
                second_system=self[jax_op.second_id()],
            )
            method = getattr(op_instance, "jax_operation", None)
            assert method is not None, (
                f"{type(op_instance)} does not define `jax_operation`."
            )
            if getattr(type(op_instance), "jax_operation") is getattr(
                NoRodRodBlockOpJax, "jax_operation"
            ):
                raise AssertionError(
                    f"{type(op_instance)} must override `jax_operation`."
                )
            wrapped = self._wrap_jax_rod2rod_operator(
                first_target=first_target,
                second_target=second_target,
                operator=method,
            )
            self._feature_group_synchronize.append_id(jax_op)
            self._feature_group_synchronize.add_operators(jax_op, [wrapped])

        self._jax_rod2rod_ops_list = []
        del self._jax_rod2rod_ops_list


class _JAXRodRodOp:
    def __init__(
        self, first_sys_idx: SystemIdxType, second_sys_idx: SystemIdxType
    ) -> None:
        self._first_sys_idx = first_sys_idx
        self._second_sys_idx = second_sys_idx
        self._op_cls: Type[Any]
        self._args: Any
        self._kwargs: Any

    def using(
        self,
        cls: Type[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert issubclass(cls, NoRodRodBlockOpJax), (
            f"{cls} is not a valid JAX rod-to-rod operator. "
            "It must derive from NoRodRodBlockOpJax."
        )
        self._op_cls = cls
        self._args = args
        self._kwargs = kwargs

    def instantiate(
        self,
        *,
        first_system: Any,
        second_system: Any,
    ) -> Any:
        if not hasattr(self, "_op_cls"):
            raise RuntimeError(
                "No JAX rod-to-rod operator provided. Did you forget to call "
                "`simulator.pairwise_interaction(...).using(...)`?"
            )
        return self._op_cls(
            *self._args,
            _first_system=first_system,
            _second_system=second_system,
            **self._kwargs,
        )

    def id(self) -> tuple[SystemIdxType, SystemIdxType]:
        return (self._first_sys_idx, self._second_sys_idx)

    def first_id(self) -> SystemIdxType:
        return self._first_sys_idx

    def second_id(self) -> SystemIdxType:
        return self._second_sys_idx
