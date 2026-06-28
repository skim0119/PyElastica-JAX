"""Convenience helpers for configuring rod memory blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Type

import jax
import numpy as np

from elastica_jax.memory_block.memory_block_rod_jax import _CosseratRodMemoryBlock
from elastica_jax.memory_block.sharded_cosserat_rod_jax import _ShardedCosseratRodBlock

DEFAULT_ROD_BLOCK_BACKEND = "cpu"
DEFAULT_ROD_BLOCK_DTYPE = np.dtype(np.float64)


def resolve_backend_devices(backend: str) -> tuple[jax.Device, ...]:
    """Return all JAX devices for an explicit backend name such as ``"cpu"``."""
    try:
        devices = tuple(jax.devices(backend))
    except Exception as exc:
        raise RuntimeError(f"Requested backend {backend!r} is unavailable.") from exc
    assert devices, f"Requested backend {backend!r} returned no devices."
    return devices


def _normalize_device_dtype(device_dtype: str | np.dtype) -> np.dtype:
    normalized = np.dtype(device_dtype)
    if normalized == np.dtype(np.float64) and not jax.config.x64_enabled:
        raise ValueError(
            "float64 device_dtype requires JAX x64 support. Enable it with "
            '`jax.config.update("jax_enable_x64", True)` or use float32.'
        )
    if normalized not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(
            "device_dtype must be one of float32 or float64 for Cosserat rod blocks."
        )
    return normalized


def configure_rod_block(
    *,
    device: str | jax.Device = DEFAULT_ROD_BLOCK_BACKEND,
    device_dtype: str | np.dtype = DEFAULT_ROD_BLOCK_DTYPE,
    inner_block_cls: Type[_CosseratRodMemoryBlock] = _CosseratRodMemoryBlock,
) -> _CosseratRodMemoryBlock:
    """
    Return a configured Cosserat rod block for ``enable_block_supports``.

    PyElastica builds the block by calling the returned instance as
    ``block(systems, system_idx_list)`` during ``finalize()``.
    """
    if isinstance(device, str):
        device = resolve_backend_devices(device)[0]
    return inner_block_cls(
        device=device,
        device_dtype=_normalize_device_dtype(device_dtype),
    )


def configure_rod_block_sharded(
    *,
    devices: Sequence[jax.Device] | None = None,
    device_dtype: str | np.dtype = DEFAULT_ROD_BLOCK_DTYPE,
    inner_block_cls: Type[_CosseratRodMemoryBlock] = _CosseratRodMemoryBlock,
    block_checkpoint: Path | str | None = None,
) -> _ShardedCosseratRodBlock:
    """
    Return a configured sharded Cosserat rod block for ``enable_block_supports``.

    Rods are split evenly across ``devices`` when the block is finalized.

    PyElastica builds the block by calling the returned instance as
    ``block(systems, system_idx_list)`` during ``finalize()``.
    """
    device_tuple = tuple(devices if devices is not None else jax.devices())
    assert len(device_tuple) >= 2, (
        "Sharded rod blocks require at least two devices. "
        "Use configure_rod_block(...) for a single-device memory block."
    )
    return _ShardedCosseratRodBlock(
        devices=device_tuple,
        device_dtype=_normalize_device_dtype(device_dtype),
        block_checkpoint=block_checkpoint,
        inner_block_cls=inner_block_cls,
    )
