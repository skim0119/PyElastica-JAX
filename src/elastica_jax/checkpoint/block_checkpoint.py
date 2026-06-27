"""Save and load JAX Cosserat rod block state for fast simulation restart."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from elastica_jax.memory_block.memory_block_rod_jax import (
    _CosseratRodMemoryBlock,
    _SYNCABLE_ATTRS,
)
from elastica_jax.memory_block.sharded_cosserat_rod_jax import (
    SHARDED_STATE_KEY,
    _ShardedCosseratRodBlock,
    is_sharded_block_state,
)

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class BlockCheckpointLayout:
    n_rods: int
    n_elements_per_rod: int
    n_shards: int
    dtype: str
    rod_to_shard: np.ndarray | None = None


class _PendingBlockCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._next_shard = 0

    def take_shard_index(self) -> int:
        shard_index = self._next_shard
        self._next_shard += 1
        return shard_index

    @property
    def is_active(self) -> bool:
        return True


_pending_block_checkpoint: _PendingBlockCheckpoint | None = None


def set_pending_block_checkpoint(path: Path | str | None) -> None:
    """Arm the next finalized block(s) to load state from ``path`` instead of rods."""
    global _pending_block_checkpoint
    if path is None:
        _pending_block_checkpoint = None
        return
    _pending_block_checkpoint = _PendingBlockCheckpoint(Path(path))


def is_block_checkpoint_load_pending() -> bool:
    return _pending_block_checkpoint is not None


def _take_pending_shard_index() -> tuple[Path | None, int | None]:
    global _pending_block_checkpoint
    if _pending_block_checkpoint is None:
        return None, None
    path = _pending_block_checkpoint.path
    shard_index = _pending_block_checkpoint.take_shard_index()
    if (
        _pending_block_checkpoint._next_shard
        >= read_block_checkpoint_layout(path).n_shards
    ):
        _pending_block_checkpoint = None
    return path, shard_index


def consume_block_checkpoint_shard() -> tuple[Path, int] | None:
    path, shard_index = _take_pending_shard_index()
    if path is None or shard_index is None:
        return None
    return path, shard_index


def read_block_checkpoint_layout(path: Path | str) -> BlockCheckpointLayout:
    with h5py.File(path, "r") as handle:
        assert (
            int(handle.attrs["version"]) == CHECKPOINT_VERSION
        ), f"Unsupported block checkpoint version in {path!s}."
        rod_to_shard = None
        if "rod_to_shard" in handle:
            rod_to_shard = np.asarray(handle["rod_to_shard"], dtype=np.int32)
        return BlockCheckpointLayout(
            n_rods=int(handle.attrs["n_rods"]),
            n_elements_per_rod=int(handle.attrs["n_elements_per_rod"]),
            n_shards=int(handle.attrs["n_shards"]),
            dtype=str(handle.attrs["dtype"]),
            rod_to_shard=rod_to_shard,
        )


def layout_rods_for_block(
    *,
    n_rods: int,
    n_elements: int,
    length: float,
    radius: float,
    density: float,
    youngs_modulus: float,
    shear_modulus: float,
    spacing: float | None = None,
):
    """Create cheap placeholder rods that only define block layout."""
    import elastica as ea

    if spacing is None:
        spacing = 2.0 * length
    rods = []
    for rod_index in range(n_rods):
        start = np.array([0.0, 0.0, spacing * rod_index], dtype=np.float64)
        rod = ea.CosseratRod.straight_rod(
            n_elements,
            start,
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            length,
            radius,
            density,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )
        rods.append(rod)
    return rods


def execution_mesh_for_block_checkpoint(
    block_checkpoint: Path | str,
    *,
    mesh_name: str,
    backend: str,
    n_rods: int,
):
    """Resolve an execution mesh, using checkpoint layout when the file exists."""
    from elastica_jax.execution_mesh import ExecutionMesh
    from elastica_jax.memory_block.block_factory import resolve_backend_devices

    checkpoint_path = Path(block_checkpoint)
    if not checkpoint_path.is_file():
        if mesh_name == "unified":
            devices = resolve_backend_devices(backend)
            return ExecutionMesh.from_devices(devices[:1], n_rods=n_rods)
        if mesh_name == "auto":
            devices = resolve_backend_devices(backend)
            return ExecutionMesh.from_devices(devices, n_rods=n_rods)
        raise ValueError(f"Unsupported mesh option {mesh_name!r}.")

    layout = read_block_checkpoint_layout(checkpoint_path)
    if mesh_name == "unified":
        devices = resolve_backend_devices(backend)
        mesh = ExecutionMesh.from_devices(devices[:1], n_rods=layout.n_rods)
    elif mesh_name == "auto":
        devices = resolve_backend_devices(backend)
        mesh = ExecutionMesh.from_devices(devices, n_rods=layout.n_rods)
    else:
        raise ValueError(f"Unsupported mesh option {mesh_name!r}.")
    assert mesh.n_shards == layout.n_shards, (
        f"Checkpoint expects {layout.n_shards} shards, but mesh={mesh_name!r} "
        f"resolved to {mesh.n_shards}."
    )
    if layout.rod_to_shard is not None:
        assert np.array_equal(
            mesh.rod_to_shard, layout.rod_to_shard
        ), "Execution mesh rod sharding does not match the block checkpoint."
    return mesh


def _shard_group_name(shard_index: int) -> str:
    return f"shard_{shard_index:03d}"


def _collect_shard_payload(block: _CosseratRodMemoryBlock) -> dict[str, Any]:
    host = {attr: np.asarray(getattr(block, attr)) for attr in _SYNCABLE_ATTRS}
    jax_state = block.jax_get_state()
    return {"host": host, "jax": jax_state}


def infer_n_elements_per_rod(
    block: _CosseratRodMemoryBlock | _ShardedCosseratRodBlock,
) -> int:
    if isinstance(block, _ShardedCosseratRodBlock):
        n_elems_in_rods = block._primary_block.n_elems_in_rods
    else:
        n_elems_in_rods = block.n_elems_in_rods
    unique = np.unique(n_elems_in_rods)
    assert unique.size == 1, "Block checkpoint save requires uniform elements per rod."
    return int(unique[0])


def save_block_checkpoint(
    block: _CosseratRodMemoryBlock | _ShardedCosseratRodBlock,
    path: Path | str,
    *,
    n_elements_per_rod: int,
) -> None:
    """Write host and JAX block state to an HDF5 checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(block, _ShardedCosseratRodBlock):
        n_shards = block.mesh.n_shards
        n_rods = block.n_rods
        dtype = str(block.device_dtype)
        shard_payloads = [
            _collect_shard_payload(shard_block) for shard_block in block._shard_blocks
        ]
    else:
        n_shards = 1
        n_rods = block.n_rods
        dtype = str(block.device_dtype)
        shard_payloads = [_collect_shard_payload(block)]

    with h5py.File(path, "w") as handle:
        handle.attrs["version"] = CHECKPOINT_VERSION
        handle.attrs["n_rods"] = n_rods
        handle.attrs["n_elements_per_rod"] = n_elements_per_rod
        handle.attrs["n_shards"] = n_shards
        handle.attrs["dtype"] = dtype
        if isinstance(block, _ShardedCosseratRodBlock):
            rod_to_shard = block.mesh.rod_to_shard
        else:
            rod_to_shard = np.zeros(n_rods, dtype=np.int32)
        handle.create_dataset("rod_to_shard", data=rod_to_shard)
        for shard_index, payload in enumerate(shard_payloads):
            group = handle.create_group(_shard_group_name(shard_index))
            if isinstance(block, _ShardedCosseratRodBlock):
                group.attrs["n_rods"] = block._shard_blocks[shard_index].n_rods
            else:
                group.attrs["n_rods"] = block.n_rods
            host_group = group.create_group("host")
            for attr, array in payload["host"].items():
                host_group.create_dataset(attr, data=array)
            jax_group = group.create_group("jax")
            for key, value in payload["jax"].items():
                jax_group.create_dataset(key, data=np.asarray(value))


def validate_block_checkpoint_shard(
    *,
    path: Path,
    shard_index: int,
    n_rods: int,
    n_elements_per_rod: int | None = None,
) -> BlockCheckpointLayout:
    layout = read_block_checkpoint_layout(path)
    assert layout.n_shards > shard_index, (
        f"Checkpoint {path!s} has {layout.n_shards} shards, "
        f"but shard {shard_index} was requested."
    )
    with h5py.File(path, "r") as handle:
        shard_n_rods = int(handle[_shard_group_name(shard_index)].attrs["n_rods"])
    assert shard_n_rods == n_rods, (
        f"Checkpoint shard {shard_index} expects {shard_n_rods} rods, "
        f"but the constructed block has {n_rods}."
    )
    if n_elements_per_rod is not None:
        assert (
            layout.n_elements_per_rod == n_elements_per_rod
        ), "Checkpoint element count per rod does not match the simulator setup."
    return layout


def apply_block_checkpoint_to_memory_block(
    block: _CosseratRodMemoryBlock,
    path: Path | str,
    *,
    shard_index: int,
    device: jax.Device | None,
) -> None:
    """Overwrite an allocated block with checkpointed host and JAX arrays."""
    path = Path(path)
    layout = validate_block_checkpoint_shard(
        path=path,
        shard_index=shard_index,
        n_rods=block.n_rods,
    )
    with h5py.File(path, "r") as handle:
        group = handle[_shard_group_name(shard_index)]
        host_group = group["host"]
        for attr in _SYNCABLE_ATTRS:
            np.copyto(getattr(block, attr), np.asarray(host_group[attr]))
        jax_state = {}
        jax_group = group["jax"]
        for key in jax_group:
            array = np.asarray(jax_group[key], dtype=layout.dtype)
            if device is None:
                jax_state[key] = jnp.asarray(array, dtype=layout.dtype)
            else:
                jax_state[key] = jax.device_put(array, device=device)
    block.jax_set_state(jax_state)


def apply_block_checkpoint_to_sharded_block(
    block: _ShardedCosseratRodBlock,
    path: Path | str,
) -> None:
    path = Path(path)
    layout = read_block_checkpoint_layout(path)
    assert (
        layout.n_shards == block.mesh.n_shards
    ), "Checkpoint shard count does not match the execution mesh."
    assert (
        layout.n_rods == block.n_rods
    ), "Checkpoint rod count does not match the sharded block."
    for shard_index, shard_block in enumerate(block._shard_blocks):
        apply_block_checkpoint_to_memory_block(
            shard_block,
            path,
            shard_index=shard_index,
            device=block.mesh.devices[shard_index],
        )
    state = block.jax_get_state()
    if is_sharded_block_state(state):
        block.jax_set_state(state)
