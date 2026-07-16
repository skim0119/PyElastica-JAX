"""Convenience helpers for configuring rod memory blocks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Type

import jax
import numpy as np

from elastica_jax.memory_block.memory_block_rod_jax import _CosseratRodMemoryBlock
from elastica_jax.memory_block.mpi_cosserat_rod_jax import _MpiCosseratRodBlock
from elastica_jax.memory_block.memory_block_rod_vertical_jax import (
    _CosseratRodVerticalMemoryBlock,
)
from elastica_jax.memory_block.protocol import RodBlockProtocol

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
    if normalized == np.dtype(np.float64) and not jax.config.x64_enabled:  # type: ignore[attr-defined]
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
    device: str | jax.Device | Sequence[jax.Device] = DEFAULT_ROD_BLOCK_BACKEND,
    device_dtype: str | np.dtype = DEFAULT_ROD_BLOCK_DTYPE,
    inner_block_cls: Type[RodBlockProtocol] = _CosseratRodMemoryBlock,
) -> RodBlockProtocol:
    """
    Return a configured Cosserat rod block for ``enable_block_supports``.

    PyElastica builds the block by calling the returned instance as
    ``block(systems, system_idx_list)`` during ``finalize()``.
    """
    if isinstance(device, str):
        device = resolve_backend_devices(device)[0]  # TODO: handle multiple devices
    elif isinstance(device, Sequence):
        device = tuple(device)
        assert device, "configure_rod_block requires at least one device."
        if len(device) > 1:
            assert inner_block_cls is _CosseratRodVerticalMemoryBlock, (
                "Multiple devices are currently supported only by "
                "_CosseratRodVerticalMemoryBlock."
            )
    return inner_block_cls(
        device=device,
        device_dtype=_normalize_device_dtype(device_dtype),
    )  # type: ignore[call-arg]


def configure_rod_block_mpi(
    *,
    comm: Any,
    device_dtype: str | np.dtype = DEFAULT_ROD_BLOCK_DTYPE,
    inner_block_cls: Type[RodBlockProtocol] = _CosseratRodMemoryBlock,
    device: str | jax.Device = DEFAULT_ROD_BLOCK_BACKEND,
) -> RodBlockProtocol:
    """
    Return a configured MPI-local Cosserat rod block for ``enable_block_supports``.

    Each MPI rank appends only the rods it owns, typically using
    ``block.owns_rod(global_index)``, then builds one local memory block during
    ``finalize()``.
    """
    return _MpiCosseratRodBlock(
        comm=comm,
        device_dtype=_normalize_device_dtype(device_dtype),
        inner_block_cls=inner_block_cls,  # type: ignore[arg-type]
        device=device,
    )
