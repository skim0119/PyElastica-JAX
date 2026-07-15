"""MPI-rank-local rod block wrapper around ``_CosseratRodMemoryBlock``."""

from __future__ import annotations

from typing import Any, Type

import jax
import numpy as np

from elastica.typing import RodType, SystemIdxType
from elastica_jax.memory_block.memory_block_rod_jax import _CosseratRodMemoryBlock
from elastica_jax.memory_block.protocol import RodBlockProtocol


def _resolve_block_device(device: str | jax.Device) -> jax.Device:
    if not isinstance(device, str):
        return device
    devices = tuple(jax.devices(device))
    assert devices, f"Requested backend {device!r} returned no devices."
    return devices[0]


class _MpiCosseratRodBlock:
    """
    One MPI rank's rod memory block.

    Each rank builds only the rods it owns, then delegates rollout and host/device
    synchronization to a single local horizontal or vertical Cosserat rod block.
    """

    def __init__(
        self,
        *,
        comm: Any,
        device_dtype: np.dtype,
        inner_block_cls: Type[RodBlockProtocol] = _CosseratRodMemoryBlock,
        device: str | jax.Device = "cpu",
    ) -> None:
        self._comm = comm
        self.comm_rank = int(comm.Get_rank())
        self.comm_size = int(comm.Get_size())
        self.device_dtype = device_dtype
        self.inner_block_cls = inner_block_cls
        self._device = _resolve_block_device(device)
        self._inner_block: RodBlockProtocol | None = None
        self.global_rod_indices: np.ndarray = np.zeros(0, dtype=np.int32)

    @property
    def comm(self) -> Any:
        """MPI communicator bound at configuration time."""
        return self._comm

    def owns_rod(self, global_rod_index: int) -> bool:
        """
        Return whether ``global_rod_index`` belongs to this MPI rank.

        Parameters
        ----------
        global_rod_index
            Global rod index in the simulation.

        Returns
        -------
        bool
            ``True`` when this rank should create and append the rod.
        """
        return int(global_rod_index) % self.comm_size == self.comm_rank

    def __call__(
        self,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> _MpiCosseratRodBlock:
        assert len(systems) > 0, (
            f"MPI rank {self.comm_rank} received no rods. "
            "Increase the global rod count or reduce comm_size."
        )
        inner_block = self.inner_block_cls(
            device=self._device,
            device_dtype=self.device_dtype,
        )
        inner_block(systems, system_idx_list)
        self._inner_block = inner_block
        n_local = len(systems)
        self.global_rod_indices = np.arange(
            self.comm_rank,
            self.comm_rank + self.comm_size * n_local,
            self.comm_size,
            dtype=np.int32,
        )
        return self

    def _require_inner_block(self) -> RodBlockProtocol:
        assert self._inner_block is not None, (
            "MPI rod block must be built during finalize() before use."
        )
        return self._inner_block

    def __getattr__(self, name: str) -> Any:
        if name in {"_inner_block", "_comm", "_device"}:
            raise AttributeError(name)
        return getattr(self._require_inner_block(), name)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other
