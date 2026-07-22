"""Shared HDF5 payload helpers for rod-block device state."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import h5py
import jax
import numpy as np

from elastica_jax.io.schema import fields_for_schema_level


class _RodBlockState(Protocol):
    n_rods: int
    device_dtype: np.dtype
    device_state: dict[str, jax.Array]

    def jax_get_state(self) -> dict[str, jax.Array]: ...

    def jax_set_state(self, state: dict[str, jax.Array]) -> None: ...


def _array_device(array: jax.Array) -> jax.Device:
    device = array.device
    if callable(device):
        device = device()
    return device


def current_device_metadata(block: _RodBlockState) -> tuple[str, int]:
    """Return ``(platform, device_id)`` from the block's device arrays."""
    device_state = block.device_state
    assert device_state, "Block has no device state to read device metadata from."
    sample = next(iter(device_state.values()))
    device = _array_device(sample)
    return str(device.platform), int(device.id)


def block_placement_device(block: _RodBlockState) -> jax.Device:
    """Return the device used to place loaded arrays for ``block``."""
    device_state = block.device_state
    assert device_state, "Block has no device state to place loaded arrays on."
    return _array_device(next(iter(device_state.values())))


def write_array_group(
    group: h5py.Group,
    arrays: dict[str, np.ndarray],
) -> None:
    """Create one dataset per array under ``group`` and fill immediately."""
    for name, array in arrays.items():
        group.create_dataset(name, data=array)


def create_empty_array_group(
    group: h5py.Group,
    arrays: dict[str, np.ndarray],
) -> None:
    """Create empty contiguous datasets matching ``arrays`` shapes."""
    for name, array in arrays.items():
        group.create_dataset(name, shape=array.shape, dtype=array.dtype)


def read_array_group(
    group: h5py.Group,
    field_names: Iterable[str],
) -> dict[str, np.ndarray]:
    """Read datasets named in ``field_names`` from ``group`` when present."""
    return {name: np.asarray(group[name]) for name in field_names if name in group}


def collect_block_arrays(
    block: _RodBlockState,
    *,
    schema_level: int,
) -> dict[str, np.ndarray]:
    """Return host copies of device collections selected by ``schema_level``."""
    field_names = fields_for_schema_level(schema_level)
    device_state = block.device_state
    arrays: dict[str, np.ndarray] = {}
    for name in field_names:
        assert name in device_state, f"Block device state is missing {name!r}."
        arrays[name] = np.asarray(device_state[name])
    return arrays


def write_block_group_metadata(
    group: h5py.Group,
    block: _RodBlockState,
) -> None:
    """Write block attrs (rod count, dtype, device metadata) onto ``group``."""
    platform, device_id = current_device_metadata(block)
    group.attrs["n_rods"] = int(block.n_rods)
    group.attrs["dtype"] = str(block.device_dtype)
    group.attrs["jax_platform"] = platform
    group.attrs["jax_device_id"] = device_id


def write_block_into(
    parent: h5py.Group,
    group_name: str,
    block: _RodBlockState,
    *,
    schema_level: int,
    fill_arrays: bool = True,
) -> dict[str, np.ndarray]:
    """Write one block's device state (including ghosts) under ``parent``.

    Parameters
    ----------
    fill_arrays :
        When True (default), datasets are created and filled. When False,
        empty datasets are created and the host arrays are returned for a
        later parallel fill.
    """
    arrays = collect_block_arrays(block, schema_level=schema_level)
    group = parent.create_group(group_name)
    write_block_group_metadata(group, block)
    if fill_arrays:
        write_array_group(group, arrays)
    else:
        create_empty_array_group(group, arrays)
    return arrays


def read_block_from(
    group: h5py.Group,
    block: _RodBlockState,
    *,
    check_device: bool,
) -> None:
    """Restore one block's device state from ``group``."""
    if check_device:
        platform, device_id = current_device_metadata(block)
        assert str(group.attrs["jax_platform"]) == platform, (
            f"Saved platform {group.attrs['jax_platform']!r} does not match "
            f"current platform {platform!r}."
        )
        assert int(group.attrs["jax_device_id"]) == device_id, (
            f"Saved device id {int(group.attrs['jax_device_id'])} does not match "
            f"current device id {device_id}."
        )

    dtype = str(group.attrs["dtype"])
    device = block_placement_device(block)
    updated = dict(block.jax_get_state())
    for name in group:
        array = np.asarray(group[name], dtype=dtype)
        updated[name] = jax.device_put(array, device=device)
    block.jax_set_state(updated)


def write_block_file_state(
    block: _RodBlockState,
    handle: h5py.File,
    *,
    schema_level: int,
) -> None:
    """Write a single-block file payload under ``blocks/0``."""
    from elastica_jax.io.schema import BLOCKS_GROUP

    parent = handle.create_group(BLOCKS_GROUP)
    write_block_into(parent, "0", block, schema_level=schema_level)


def read_block_file_state(
    block: _RodBlockState,
    handle: h5py.File,
    *,
    check_device: bool,
) -> None:
    """Read a single-block file payload from ``blocks/0``."""
    from elastica_jax.io.schema import BLOCKS_GROUP

    read_block_from(handle[BLOCKS_GROUP]["0"], block, check_device=check_device)
