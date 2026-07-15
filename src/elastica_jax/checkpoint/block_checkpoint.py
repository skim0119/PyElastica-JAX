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

CHECKPOINT_VERSION = 1
_LEGACY_GROUP_NAME = "shard_000"
_BLOCK_GROUP_NAME = "block"


@dataclass(frozen=True)
class BlockCheckpointLayout:
    """Metadata describing one packed rod-block checkpoint."""

    n_rods: int
    n_elements_per_rod: int
    dtype: str


def _checkpoint_group(handle: h5py.File) -> h5py.Group:
    if _BLOCK_GROUP_NAME in handle:
        return handle[_BLOCK_GROUP_NAME]
    if _LEGACY_GROUP_NAME in handle:
        return handle[_LEGACY_GROUP_NAME]
    raise AssertionError("Checkpoint does not contain a recognized block group.")


def read_block_checkpoint_layout(path: Path | str) -> BlockCheckpointLayout:
    """Read checkpoint metadata without materializing block arrays."""
    with h5py.File(path, "r") as handle:
        assert (
            int(handle.attrs["version"]) == CHECKPOINT_VERSION
        ), f"Unsupported block checkpoint version in {path!s}."
        return BlockCheckpointLayout(
            n_rods=int(handle.attrs["n_rods"]),
            n_elements_per_rod=int(handle.attrs["n_elements_per_rod"]),
            dtype=str(handle.attrs["dtype"]),
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
) -> list[Any]:
    """Create cheap placeholder rods that only define block layout."""
    import elastica as ea

    if spacing is None:
        spacing = 2.0 * length
    rods: list[Any] = []
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


def infer_n_elements_per_rod(block: _CosseratRodMemoryBlock) -> int:
    """Return the common element count per rod, asserting uniform discretization."""
    unique = np.unique(block.n_elems_in_rods)
    assert unique.size == 1, "Block checkpoint save requires uniform elements per rod."
    return int(unique[0])


def save_block_checkpoint(
    block: _CosseratRodMemoryBlock,
    path: Path | str,
    *,
    n_elements_per_rod: int,
) -> None:
    """Write host and JAX block state to an HDF5 checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    host_state = {attr: np.asarray(getattr(block, attr)) for attr in _SYNCABLE_ATTRS}
    jax_state = block.jax_get_state()

    with h5py.File(checkpoint_path, "w") as handle:
        handle.attrs["version"] = CHECKPOINT_VERSION
        handle.attrs["n_rods"] = block.n_rods
        handle.attrs["n_elements_per_rod"] = n_elements_per_rod
        handle.attrs["dtype"] = str(block.device_dtype)

        group = handle.create_group(_BLOCK_GROUP_NAME)
        group.attrs["n_rods"] = block.n_rods

        host_group = group.create_group("host")
        for attr, array in host_state.items():
            host_group.create_dataset(attr, data=array)

        jax_group = group.create_group("jax")
        for key, value in jax_state.items():
            jax_group.create_dataset(key, data=np.asarray(value))


def validate_block_checkpoint(
    *,
    path: Path,
    n_rods: int,
    n_elements_per_rod: int | None = None,
) -> BlockCheckpointLayout:
    """Validate checkpoint layout against the current simulator setup."""
    layout = read_block_checkpoint_layout(path)
    assert layout.n_rods == n_rods, (
        f"Checkpoint expects {layout.n_rods} rods, but the constructed block has "
        f"{n_rods}."
    )
    if n_elements_per_rod is not None:
        assert layout.n_elements_per_rod == n_elements_per_rod, (
            "Checkpoint element count per rod does not match the simulator setup."
        )
    with h5py.File(path, "r") as handle:
        group = _checkpoint_group(handle)
        group_n_rods = int(group.attrs["n_rods"])
    assert group_n_rods == n_rods, (
        f"Checkpoint block group expects {group_n_rods} rods, but the constructed "
        f"block has {n_rods}."
    )
    return layout


def apply_block_checkpoint_to_memory_block(
    block: _CosseratRodMemoryBlock,
    path: Path | str,
    *,
    device: jax.Device | None,
) -> None:
    """Overwrite an allocated block with checkpointed host and JAX arrays."""
    checkpoint_path = Path(path)
    layout = validate_block_checkpoint(
        path=checkpoint_path,
        n_rods=block.n_rods,
    )
    with h5py.File(checkpoint_path, "r") as handle:
        group = _checkpoint_group(handle)
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
