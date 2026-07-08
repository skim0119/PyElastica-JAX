"""Rod sharding wrapper around ``_CosseratRodMemoryBlock``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence, Type

import jax
import jax.numpy as jnp
import numpy as np

from elastica.typing import RodType, SystemIdxType
from elastica_jax.memory_block.memory_block_rod_jax import (
    RodSyncTarget,
    _CosseratRodMemoryBlock,
    _ELEMENT_ATTRS,
    _NODE_ATTRS,
    _SYNCABLE_ATTRS,
    _VORONOI_ATTRS,
)

SHARDED_STATE_KEY = "_eaj_sharded_state"
CONTACT_STATE_KEY_PREFIX = "capsule_contact_"
_ATTR_DOMAINS: dict[str, str] = (
    {attr: "node" for attr in _NODE_ATTRS}
    | {attr: "element" for attr in _ELEMENT_ATTRS}
    | {attr: "voronoi" for attr in _VORONOI_ATTRS}
)


def _slice_merged_array_for_shard(
    *,
    key: str,
    value: jax.Array,
    current: jax.Array,
    node_offset: int,
    node_count: int,
    elem_offset: int,
    elem_count: int,
    voronoi_offset: int,
    voronoi_count: int,
) -> jax.Array:
    domain = _ATTR_DOMAINS[key]
    if domain == "node":
        offset, count = node_offset, node_count
    elif domain == "element":
        offset, count = elem_offset, elem_count
    else:
        offset, count = voronoi_offset, voronoi_count

    if current.ndim == 1:
        return value[offset : offset + count]
    if current.ndim == 2:
        return value[:, offset : offset + count]
    if current.ndim == 3:
        return value[:, :, offset : offset + count]
    raise ValueError(f"Unsupported rank {current.ndim} for sharded state key {key!r}.")


class _ShardedCosseratRodBlock:
    """
    Maps one logical rod bundle onto one or more device-local blocks.

    Configure with ``__init__``, then build during ``finalize()`` via
    ``block(systems, system_idx_list)``.
    """

    def __init__(
        self,
        *,
        devices: Sequence[jax.Device],
        device_dtype: np.dtype,
        block_checkpoint: Path | str | None = None,
        inner_block_cls: Type[_CosseratRodMemoryBlock] = _CosseratRodMemoryBlock,
    ) -> None:
        self._devices = tuple(devices)
        self._rod_to_shard = np.zeros(0, dtype=np.int32)
        self.device_dtype = device_dtype
        self.block_checkpoint_path = (
            Path(block_checkpoint) if block_checkpoint is not None else None
        )
        self.inner_block_cls = inner_block_cls

    @property
    def rod_to_shard(self) -> np.ndarray:
        return self._rod_to_shard

    @property
    def n_shards(self) -> int:
        return len(self._devices)

    def __call__(
        self,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> _ShardedCosseratRodBlock:
        # Only allow to build sharded block when:
        # Number of rods is greater than 2
        # Number of rods is divisible by number of devices.
        # TODO: Padding could resolve this issue. Not sure if the padding
        # should be implemented within the block implementation, or leave
        # it a user responsibility.
        assert len(systems) >= len(
            self._devices
        ), "Number of rods must be at least the number of devices."
        assert (
            len(systems) % len(self._devices) == 0
        ), "Number of rods must be divisible by number of devices."

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
        self._resolve_shard_layout(n_rods)
        self._systems = tuple(systems)
        self.n_rods = n_rods
        self._shard_blocks: tuple[_CosseratRodMemoryBlock, ...] = tuple(
            self._build_shard_block(
                shard_index=shard_index,
                systems=systems,
                system_idx_list=system_idx_list,
            )
            for shard_index in range(self.n_shards)
        )
        self._primary_block = self._shard_blocks[0]
        self._build_global_index_maps()

    def _resolve_shard_layout(self, n_rods: int) -> None:
        from elastica_jax.checkpoint.block_checkpoint import (
            read_block_checkpoint_layout,
        )

        n_shards = len(self._devices)
        checkpoint_path = self.block_checkpoint_path
        if checkpoint_path is not None and checkpoint_path.is_file():
            layout = read_block_checkpoint_layout(checkpoint_path)
            assert (
                layout.n_rods == n_rods
            ), "Block checkpoint rod count does not match appended rods."
            assert (
                layout.n_shards == n_shards
            ), "Block checkpoint shard count does not match configured devices."
            if layout.rod_to_shard is not None:
                self._rod_to_shard = np.asarray(layout.rod_to_shard, dtype=np.int32)
            else:
                self._rod_to_shard = np.arange(n_rods, dtype=np.int32) % n_shards
            return

        self._rod_to_shard = np.arange(n_rods, dtype=np.int32) % n_shards

    def _build_shard_block(
        self,
        *,
        shard_index: int,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> _CosseratRodMemoryBlock:
        rod_indices = np.where(self._rod_to_shard == shard_index)[0]
        shard_systems = [systems[int(index)] for index in rod_indices]
        shard_system_idx = [system_idx_list[int(index)] for index in rod_indices]
        shard_block = self.inner_block_cls(
            device=self._devices[shard_index],
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

    def __getattr__(self, attr: str) -> np.ndarray:
        if attr not in _SYNCABLE_ATTRS:
            raise AttributeError(attr)
        return self._get_syncable_attr(attr)

    def _get_syncable_attr(self, attr: str) -> np.ndarray:
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

    def _concatenate_device_arrays(
        self,
        arrays: list[jax.Array],
        *,
        axis: int,
    ) -> jax.Array:
        transferred = [
            jax.device_put(array, device=self._devices[0]) for array in arrays
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

    def _normalize_rod_sync_target(self, rods: RodSyncTarget) -> tuple[RodType, ...]:
        assert hasattr(
            self, "_systems"
        ), "Block must be built before synchronizing with rods."
        if rods == "all":
            return self._systems
        if isinstance(rods, (list, tuple)):
            return tuple(rods)
        return (rods,)

    def _shard_blocks_for_rods(
        self, rods: RodSyncTarget
    ) -> tuple[tuple[_CosseratRodMemoryBlock, RodSyncTarget], ...]:
        if rods == "all":
            return tuple((block, "all") for block in self._shard_blocks)

        rods_by_shard: dict[int, list[RodType]] = {}
        for rod in self._normalize_rod_sync_target(rods):
            global_index = self._systems.index(rod)
            shard_index = int(self._rod_to_shard[global_index])
            rods_by_shard.setdefault(shard_index, []).append(rod)

        return tuple(
            (
                self._shard_blocks[shard_index],
                shard_rods[0] if len(shard_rods) == 1 else tuple(shard_rods),
            )
            for shard_index, shard_rods in sorted(rods_by_shard.items())
        )

    def to_device(
        self,
        rods: RodSyncTarget = "all",
        *,
        variables: Iterable[str] | None = None,
    ) -> None:
        """
        Copy selected fields from host block memory and rod objects to device.

        Parameters
        ----------
        rods
            One rod, a sequence of rods, or ``"all"`` for every rod in the block.
        variables
            Block fields to synchronize. Defaults to all syncable fields.
        """
        for shard_block, shard_rods in self._shard_blocks_for_rods(rods):
            shard_block.to_device(shard_rods, variables=variables)

    def from_device(
        self,
        rods: RodSyncTarget = "all",
        *,
        variables: Iterable[str] | None = None,
        update_rods: bool = True,
    ) -> None:
        """
        Copy selected fields from device to host block memory and rod objects.

        Parameters
        ----------
        rods
            One rod, a sequence of rods, or ``"all"`` for every rod in the block.
        variables
            Block fields to synchronize. Defaults to all syncable fields.
        update_rods
            When ``False``, update block host memory only.
        """
        for shard_block, shard_rods in self._shard_blocks_for_rods(rods):
            shard_block.from_device(
                shard_rods,
                variables=variables,
                update_rods=update_rods,
            )

    @property
    def position_collection_device(self) -> jax.Array:
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
        return {
            SHARDED_STATE_KEY: True,
            "shards": tuple(block.jax_get_state() for block in self._shard_blocks),
        }

    def jax_set_state(self, state: dict[str, Any]) -> None:
        for block, shard_state in zip(self._shard_blocks, state["shards"]):
            block.jax_set_state(shard_state)

    def jax_kinematic_step(
        self, state: dict[str, Any], time: np.float64, prefac: np.float64
    ) -> dict[str, Any]:
        return {
            SHARDED_STATE_KEY: True,
            "shards": self._map_shard_states(state, "jax_kinematic_step", time, prefac),
        }

    def jax_compute_internal_forces_and_torques(
        self, state: dict[str, Any], time: np.float64
    ) -> dict[str, Any]:
        return {
            SHARDED_STATE_KEY: True,
            "shards": self._map_shard_states(
                state, "jax_compute_internal_forces_and_torques", time
            ),
        }

    def jax_dynamic_step(
        self, state: dict[str, Any], time: np.float64, dt: np.float64
    ) -> dict[str, Any]:
        return {
            SHARDED_STATE_KEY: True,
            "shards": self._map_shard_states(state, "jax_dynamic_step", time, dt),
        }

    def jax_zero_external_loads(
        self, state: dict[str, Any], time: np.float64
    ) -> dict[str, Any]:
        return {
            SHARDED_STATE_KEY: True,
            "shards": self._map_shard_states(state, "jax_zero_external_loads", time),
        }

    def merge_shard_states(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build a unified logical block state on the primary shard device."""
        assert state.get(
            SHARDED_STATE_KEY, False
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
        assert state.get(
            SHARDED_STATE_KEY, False
        ), "scatter_merged_state requires a sharded state."
        scattered_shards = []
        node_offset = 0
        elem_offset = 0
        voronoi_offset = 0
        for shard_index, (block, shard_state) in enumerate(
            zip(self._shard_blocks, state["shards"])
        ):
            updated = dict(shard_state)
            node_count = block.n_nodes
            elem_count = block.n_elems
            voronoi_count = block.n_voronoi
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
                assert (
                    key in _ATTR_DOMAINS
                ), f"Cannot scatter unknown sharded state key {key!r}."
                shard_value = _slice_merged_array_for_shard(
                    key=key,
                    value=value,
                    current=current,
                    node_offset=node_offset,
                    node_count=node_count,
                    elem_offset=elem_offset,
                    elem_count=elem_count,
                    voronoi_offset=voronoi_offset,
                    voronoi_count=voronoi_count,
                )
                updated[key] = jax.device_put(shard_value, device=shard_device)
            scattered_shards.append(updated)
            node_offset += node_count
            elem_offset += elem_count
            voronoi_offset += voronoi_count
        return {SHARDED_STATE_KEY: True, "shards": tuple(scattered_shards)}
