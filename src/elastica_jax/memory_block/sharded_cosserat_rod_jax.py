"""Rod sharding wrapper around ``_CosseratRodMemoryBlock``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import jax
import jax.numpy as jnp
import numpy as np

from elastica_jax.execution_mesh import ExecutionMesh
from elastica_jax.memory_block.memory_block_rod_jax import (
    _CosseratRodMemoryBlock,
    _SYNCABLE_ATTRS,
)
from elastica.typing import RodType, SystemIdxType

SHARDED_STATE_KEY = "_eaj_sharded_state"
CONTACT_STATE_KEY_PREFIX = "capsule_contact_"


def is_sharded_block_state(state: dict[str, Any]) -> bool:
    return bool(state.get(SHARDED_STATE_KEY, False))


class _ShardedCosseratRodBlock:
    """
    Maps one logical rod bundle onto one or more device-local blocks.

    Configure with ``__init__``, then build during ``finalize()`` via
    ``block(systems, system_idx_list)``.
    """

    def __init__(
        self,
        *,
        mesh: ExecutionMesh,
        device_dtype: np.dtype,
        block_checkpoint: Path | str | None = None,
        inner_block_cls: Type[_CosseratRodMemoryBlock] = _CosseratRodMemoryBlock,
    ) -> None:
        self.mesh = mesh
        self.device_dtype = device_dtype
        self.block_checkpoint_path = (
            Path(block_checkpoint) if block_checkpoint is not None else None
        )
        self.inner_block_cls = inner_block_cls

    def __call__(
        self,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> _ShardedCosseratRodBlock:
        from elastica_jax.checkpoint.block_checkpoint import (
            infer_n_elements_per_rod,
            save_block_checkpoint,
            set_pending_block_checkpoint,
        )

        checkpoint_path = self.block_checkpoint_path
        if checkpoint_path is not None and checkpoint_path.is_file():
            set_pending_block_checkpoint(checkpoint_path)
        try:
            self._initialize_sharded_block(systems, system_idx_list)
        finally:
            set_pending_block_checkpoint(None)

        if checkpoint_path is not None and not checkpoint_path.is_file():
            save_block_checkpoint(
                self,
                checkpoint_path,
                n_elements_per_rod=infer_n_elements_per_rod(self),
            )
        return self

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def _initialize_sharded_block(
        self,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> None:
        n_rods = len(systems)
        mesh = self.mesh
        if mesh.rod_to_shard.size != n_rods:
            if mesh.n_shards == 1 and mesh.rod_to_shard.size == 0:
                mesh = ExecutionMesh(
                    devices=mesh.devices,
                    rod_to_shard=np.zeros(n_rods, dtype=np.int32),
                )
                self.mesh = mesh
            else:
                assert mesh.rod_to_shard.shape == (
                    n_rods,
                ), "ExecutionMesh.rod_to_shard must have one entry per rod."
        self._systems = tuple(systems)
        self.n_rods = n_rods
        self._shard_blocks: tuple[_CosseratRodMemoryBlock, ...] = tuple(
            self._build_shard_block(
                shard_index=shard_index,
                systems=systems,
                system_idx_list=system_idx_list,
            )
            for shard_index in range(mesh.n_shards)
        )
        if mesh.is_sharded:
            self._primary_block = self._shard_blocks[0]
            self._build_global_index_maps()
        else:
            self._primary_block = self._shard_blocks[0]
            self._mirror_primary_layout()

    def _build_shard_block(
        self,
        *,
        shard_index: int,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> _CosseratRodMemoryBlock:
        rod_indices = np.where(self.mesh.rod_to_shard == shard_index)[0]
        shard_systems = [systems[int(index)] for index in rod_indices]
        shard_system_idx = [system_idx_list[int(index)] for index in rod_indices]
        shard_block = self.inner_block_cls(
            device=self.mesh.devices[shard_index],
            device_dtype=self.device_dtype,
        )
        shard_block(shard_systems, shard_system_idx)
        return shard_block

    def _build_global_index_maps(self) -> None:
        node_offsets = [0]
        elem_offsets = [0]
        for block in self._shard_blocks:
            node_offsets.append(node_offsets[-1] + block.n_nodes)
            elem_offsets.append(elem_offsets[-1] + block.n_elems)
        self._global_node_offsets = np.asarray(node_offsets, dtype=np.int32)
        self._global_elem_offsets = np.asarray(elem_offsets, dtype=np.int32)
        self.n_nodes = int(self._global_node_offsets[-1])
        self.n_elems = int(self._global_elem_offsets[-1])
        self.n_systems = self.n_rods
        self.system_idx_list = np.asarray(
            [
                block.system_idx_list[local]
                for block in self._shard_blocks
                for local in range(block.n_rods)
            ],
            dtype=np.int32,
        )
        self.start_idx_in_rod_nodes = self._concatenate_rod_index_array(
            "start_idx_in_rod_nodes"
        )
        self.end_idx_in_rod_nodes = self._concatenate_rod_index_array(
            "end_idx_in_rod_nodes"
        )
        self.start_idx_in_rod_elems = self._concatenate_rod_index_array(
            "start_idx_in_rod_elems"
        )
        self.end_idx_in_rod_elems = self._concatenate_rod_index_array(
            "end_idx_in_rod_elems"
        )
        self.start_idx_in_rod_voronoi = self._concatenate_rod_index_array(
            "start_idx_in_rod_voronoi"
        )
        self.end_idx_in_rod_voronoi = self._concatenate_rod_index_array(
            "end_idx_in_rod_voronoi"
        )
        self.ring_rod_flag = any(block.ring_rod_flag for block in self._shard_blocks)

    def _mirror_primary_layout(self) -> None:
        block = self._primary_block
        self.n_nodes = block.n_nodes
        self.n_elems = block.n_elems
        self.n_systems = block.n_systems
        self.system_idx_list = block.system_idx_list
        self.start_idx_in_rod_nodes = block.start_idx_in_rod_nodes
        self.end_idx_in_rod_nodes = block.end_idx_in_rod_nodes
        self.start_idx_in_rod_elems = block.start_idx_in_rod_elems
        self.end_idx_in_rod_elems = block.end_idx_in_rod_elems
        self.start_idx_in_rod_voronoi = block.start_idx_in_rod_voronoi
        self.end_idx_in_rod_voronoi = block.end_idx_in_rod_voronoi
        self.ring_rod_flag = block.ring_rod_flag

    def __getattr__(self, attr: str) -> np.ndarray:
        if attr not in _SYNCABLE_ATTRS:
            raise AttributeError(attr)
        return self._get_syncable_attr(attr)

    def _get_syncable_attr(self, attr: str) -> np.ndarray:
        if not self.mesh.is_sharded:
            return getattr(self._primary_block, attr)
        chunks = [getattr(block, attr) for block in self._shard_blocks]
        if chunks[0].ndim == 1:
            return np.concatenate(chunks, axis=0)
        if chunks[0].ndim == 2:
            return np.concatenate(chunks, axis=1)
        if chunks[0].ndim == 3:
            return np.concatenate(chunks, axis=2)
        raise ValueError(
            f"Unsupported rank {chunks[0].ndim} for syncable attribute {attr!r}."
        )

    def _primary_device(self) -> jax.Device:
        return self.mesh.devices[0]

    def _concatenate_device_arrays(
        self,
        arrays: list[jax.Array],
        *,
        axis: int,
    ) -> jax.Array:
        primary_device = self._primary_device()
        transferred = [
            jax.device_put(array, device=primary_device) for array in arrays
        ]
        return jnp.concatenate(transferred, axis=axis)

    def _concatenate_rod_index_array(self, attr: str) -> np.ndarray:
        chunks = []
        node_offset = 0
        elem_offset = 0
        voronoi_offset = 0
        for block in self._shard_blocks:
            local = getattr(block, attr).copy()
            if attr.endswith("_nodes"):
                local += node_offset
            elif attr.endswith("_voronoi"):
                local += voronoi_offset
            else:
                local += elem_offset
            chunks.append(local)
            node_offset += block.n_nodes
            elem_offset += block.n_elems
            voronoi_offset += block.n_voronoi
        return np.concatenate(chunks)

    @property
    def position_collection_device(self) -> jax.Array:
        if not self.mesh.is_sharded:
            return self._primary_block.position_collection_device
        shards = [block.position_collection_device for block in self._shard_blocks]
        return self._concatenate_device_arrays(shards, axis=1)

    def _map_shard_states(
        self, state: dict[str, Any], method_name: str, *args: Any
    ) -> tuple[dict[str, Any], ...]:
        shard_states = state["shards"]
        method = getattr(self._shard_blocks[0], method_name)
        del method
        updated = []
        for block, shard_state in zip(self._shard_blocks, shard_states):
            updated.append(getattr(block, method_name)(shard_state, *args))
        return tuple(updated)

    def jax_get_state(self) -> dict[str, Any]:
        if not self.mesh.is_sharded:
            return self._primary_block.jax_get_state()
        return {
            SHARDED_STATE_KEY: True,
            "shards": tuple(block.jax_get_state() for block in self._shard_blocks),
        }

    def jax_set_state(self, state: dict[str, Any]) -> None:
        if not is_sharded_block_state(state):
            self._primary_block.jax_set_state(state)
            return
        for block, shard_state in zip(self._shard_blocks, state["shards"]):
            block.jax_set_state(shard_state)

    def jax_kinematic_step(
        self, state: dict[str, Any], time: np.float64, prefac: np.float64
    ) -> dict[str, Any]:
        if not self.mesh.is_sharded:
            return self._primary_block.jax_kinematic_step(state, time, prefac)
        return {
            SHARDED_STATE_KEY: True,
            "shards": self._map_shard_states(state, "jax_kinematic_step", time, prefac),
        }

    def jax_compute_internal_forces_and_torques(
        self, state: dict[str, Any], time: np.float64
    ) -> dict[str, Any]:
        if not self.mesh.is_sharded:
            return self._primary_block.jax_compute_internal_forces_and_torques(
                state, time
            )
        return {
            SHARDED_STATE_KEY: True,
            "shards": self._map_shard_states(
                state, "jax_compute_internal_forces_and_torques", time
            ),
        }

    def jax_dynamic_step(
        self, state: dict[str, Any], time: np.float64, dt: np.float64
    ) -> dict[str, Any]:
        if not self.mesh.is_sharded:
            return self._primary_block.jax_dynamic_step(state, time, dt)
        return {
            SHARDED_STATE_KEY: True,
            "shards": self._map_shard_states(state, "jax_dynamic_step", time, dt),
        }

    def jax_zero_external_loads(
        self, state: dict[str, Any], time: np.float64
    ) -> dict[str, Any]:
        if not self.mesh.is_sharded:
            return self._primary_block.jax_zero_external_loads(state, time)
        return {
            SHARDED_STATE_KEY: True,
            "shards": self._map_shard_states(state, "jax_zero_external_loads", time),
        }

    def merge_shard_states(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build a unified logical block state on the primary shard device."""
        assert is_sharded_block_state(
            state
        ), "merge_shard_states requires a sharded state."
        merged: dict[str, Any] = {}
        for key in state["shards"][0]:
            chunks = []
            for block, shard_state in zip(self._shard_blocks, state["shards"]):
                value = shard_state[key]
                if value.ndim == 1:
                    chunks.append(value)
                elif value.ndim == 2:
                    chunks.append(value)
                elif value.ndim == 3:
                    chunks.append(value)
                else:
                    raise ValueError(
                        f"Unsupported rank {value.ndim} for state key {key!r}."
                    )
            if state["shards"][0][key].ndim == 1:
                merged[key] = self._concatenate_device_arrays(chunks, axis=0)
            elif state["shards"][0][key].ndim == 2:
                merged[key] = self._concatenate_device_arrays(chunks, axis=1)
            else:
                merged[key] = self._concatenate_device_arrays(chunks, axis=2)
        return merged

    def scatter_merged_state(
        self, merged: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        """Scatter unified keys from ``merged`` back into shard states."""
        assert is_sharded_block_state(
            state
        ), "scatter_merged_state requires a sharded state."
        scattered_shards = []
        node_offset = 0
        elem_offset = 0
        for shard_index, (block, shard_state) in enumerate(
            zip(self._shard_blocks, state["shards"])
        ):
            updated = dict(shard_state)
            node_count = block.n_nodes
            elem_count = block.n_elems
            shard_device = block._initial_device
            for key, value in merged.items():
                if key not in shard_state and not key.startswith(
                    CONTACT_STATE_KEY_PREFIX
                ):
                    continue
                if key.startswith(CONTACT_STATE_KEY_PREFIX):
                    if shard_index == 0:
                        updated[key] = jax.device_put(value, device=shard_device)
                    continue
                current = shard_state[key]
                if current.ndim == 1:
                    shard_value = value[node_offset : node_offset + node_count]
                elif current.ndim == 2:
                    shard_value = value[:, node_offset : node_offset + node_count]
                else:
                    shard_value = value[:, :, elem_offset : elem_offset + elem_count]
                updated[key] = jax.device_put(shard_value, device=shard_device)
            scattered_shards.append(updated)
            node_offset += node_count
            elem_offset += elem_count
        return {SHARDED_STATE_KEY: True, "shards": tuple(scattered_shards)}
