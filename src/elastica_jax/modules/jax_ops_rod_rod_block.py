from __future__ import annotations

from typing import Any, Type

import numpy as np

from elastica_jax.rod_rod_operation import NoRodRodBlockOpJax
from elastica_jax.memory_block.memory_block_rod_jax import (
    JAXRodView,
    JAXRodViewMetadata,
    _CosseratRodMemoryBlock,
)
from elastica_jax.protocol import JAXPyTree, JAXScalar
from elastica.typing import SystemIdxType, SystemProtocol

from elastica.modules.protocol import SystemCollectionProtocol
from .jax_ops import JAXBasicMixins, JAXStageCallable


class JAXInteraction(JAXBasicMixins, SystemCollectionProtocol):
    """
    Register JAX rod-to-rod interaction operators.

    Pair interactions such as rod-rod contact are applied through rod-local views
    into one or two packed blocks. Rods may live in the same block or in
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
        first_metadata: JAXRodViewMetadata,
        second_metadata: JAXRodViewMetadata,
        operator: Any,
    ) -> JAXStageCallable:
        same_block = first_metadata.block_state_idx == second_metadata.block_state_idx

        def apply(
            *,
            states: tuple[JAXPyTree, ...],
            time: JAXScalar,
        ) -> tuple[JAXPyTree, ...]:
            if same_block:
                block_state = states[first_metadata.block_state_idx]
                shared_updates: dict[str, Any] = {}
                rod_one_view = JAXRodView(
                    block_state, first_metadata, updates=shared_updates
                )
                rod_two_view = JAXRodView(
                    block_state, second_metadata, updates=shared_updates
                )
                updated_views = operator(rod_one_view, rod_two_view, time)

                if updated_views is None:
                    committed_state = rod_one_view.commit()
                else:
                    updated_rod_one_view, _updated_rod_two_view = updated_views
                    committed_state = (
                        updated_rod_one_view.commit()
                        if isinstance(updated_rod_one_view, JAXRodView)
                        else rod_one_view.commit()
                    )

                updated_states = list(states)
                updated_states[first_metadata.block_state_idx] = committed_state
                return tuple(updated_states)

            rod_one_state = states[first_metadata.block_state_idx]
            rod_two_state = states[second_metadata.block_state_idx]
            rod_one_view = JAXRodView(rod_one_state, first_metadata)
            rod_two_view = JAXRodView(rod_two_state, second_metadata)
            updated_views = operator(rod_one_view, rod_two_view, time)

            if updated_views is None:
                committed_rod_one_state = rod_one_view.commit()
                committed_rod_two_state = rod_two_view.commit()
            else:
                updated_rod_one_view, updated_rod_two_view = updated_views
                committed_rod_one_state = (
                    updated_rod_one_view.commit()
                    if isinstance(updated_rod_one_view, JAXRodView)
                    else rod_one_view.commit()
                )
                committed_rod_two_state = (
                    updated_rod_two_view.commit()
                    if isinstance(updated_rod_two_view, JAXRodView)
                    else rod_two_view.commit()
                )

            updated_states = list(states)
            updated_states[first_metadata.block_state_idx] = committed_rod_one_state
            updated_states[second_metadata.block_state_idx] = committed_rod_two_state
            return tuple(updated_states)

        return apply

    def _finalize_jax_rod2rod_ops(self) -> None:
        final_systems = tuple(self.final_systems())

        for jax_op in self._jax_rod2rod_ops_list:
            first_metadata, second_metadata = self._make_rod_pair_view_metadata(
                final_systems,
                jax_op.first_id(),
                jax_op.second_id(),
            )
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
                first_metadata=first_metadata,
                second_metadata=second_metadata,
                operator=method,
            )
            self._feature_group_synchronize.append_id(jax_op)
            self._feature_group_synchronize.add_operators(jax_op, [wrapped])

        self._jax_rod2rod_ops_list = []
        del self._jax_rod2rod_ops_list

    @classmethod
    def _rod_view_metadata_in_block(
        cls,
        *,
        block_state_idx: int,
        system: _CosseratRodMemoryBlock,
        local_idx: int,
    ) -> JAXRodViewMetadata:
        return JAXRodViewMetadata(
            block_state_idx=block_state_idx,
            node_slice=slice(
                int(system.start_idx_in_rod_nodes[local_idx]),
                int(system.end_idx_in_rod_nodes[local_idx]),
            ),
            element_slice=slice(
                int(system.start_idx_in_rod_elems[local_idx]),
                int(system.end_idx_in_rod_elems[local_idx]),
            ),
            voronoi_slice=slice(
                int(system.start_idx_in_rod_voronoi[local_idx]),
                int(system.end_idx_in_rod_voronoi[local_idx]),
            ),
        )

    @classmethod
    def _make_rod_pair_view_metadata(
        cls,
        final_systems: tuple[object, ...],
        first_sys_idx: SystemIdxType,
        second_sys_idx: SystemIdxType,
    ) -> tuple[JAXRodViewMetadata, JAXRodViewMetadata]:
        first_metadata: JAXRodViewMetadata | None = None
        second_metadata: JAXRodViewMetadata | None = None
        for block_state_idx, system in enumerate(final_systems):
            if not isinstance(system, _CosseratRodMemoryBlock):
                continue
            first_matches = np.where(system.system_idx_list == first_sys_idx)[0]
            if first_matches.size > 0 and first_metadata is None:
                first_metadata = cls._rod_view_metadata_in_block(
                    block_state_idx=block_state_idx,
                    system=system,
                    local_idx=int(first_matches[0]),
                )
            second_matches = np.where(system.system_idx_list == second_sys_idx)[0]
            if second_matches.size > 0 and second_metadata is None:
                second_metadata = cls._rod_view_metadata_in_block(
                    block_state_idx=block_state_idx,
                    system=system,
                    local_idx=int(second_matches[0]),
                )

        if first_metadata is None or second_metadata is None:
            raise RuntimeError(
                "Requested rod pair was not found in finalized "
                "_CosseratRodMemoryBlock systems."
            )
        return first_metadata, second_metadata


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
