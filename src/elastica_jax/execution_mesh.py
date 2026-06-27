"""Execution mesh for rod sharding across local workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import jax
import numpy as np


@dataclass(frozen=True)
class ExecutionMesh:
    """Static mapping from rods to shard indices and JAX devices."""

    devices: tuple[jax.Device, ...]
    rod_to_shard: np.ndarray

    @property
    def n_shards(self) -> int:
        return len(self.devices)

    @property
    def n_rods(self) -> int:
        return int(self.rod_to_shard.size)

    @property
    def is_sharded(self) -> bool:
        return self.n_shards > 1

    def rods_on_shard(self, shard_index: int) -> np.ndarray:
        return np.where(self.rod_to_shard == shard_index)[0].astype(np.int32)

    @classmethod
    def unified(cls, device: jax.Device | None = None) -> ExecutionMesh:
        if device is None:
            device = jax.devices()[0]
        return cls(devices=(device,), rod_to_shard=np.zeros(0, dtype=np.int32))

    @classmethod
    def for_n_rods(
        cls,
        n_rods: int,
        *,
        devices: Sequence[jax.Device],
        strategy: Literal["balance"] = "balance",
    ) -> ExecutionMesh:
        assert n_rods >= 0, "n_rods must be nonnegative."
        device_tuple = tuple(devices)
        assert len(device_tuple) >= 1, "At least one device is required."
        n_shards = min(len(device_tuple), max(1, n_rods))
        selected_devices = device_tuple[:n_shards]
        if n_rods == 0:
            return cls(
                devices=selected_devices, rod_to_shard=np.zeros(0, dtype=np.int32)
            )
        if strategy != "balance":
            raise ValueError(f"Unsupported rod sharding strategy {strategy!r}.")
        rod_to_shard = np.arange(n_rods, dtype=np.int32) % n_shards
        return cls(devices=selected_devices, rod_to_shard=rod_to_shard)

    @classmethod
    def from_devices(
        cls,
        devices: Sequence[jax.Device] | None = None,
        *,
        n_rods: int,
        strategy: Literal["balance"] = "balance",
    ) -> ExecutionMesh:
        if devices is None:
            devices = tuple(jax.devices())
        assert len(devices) >= 1, "No JAX devices are available."
        if len(devices) == 1:
            mesh = cls.unified(devices[0])
            if n_rods == 0:
                return mesh
            return cls(
                devices=mesh.devices,
                rod_to_shard=np.zeros(n_rods, dtype=np.int32),
            )
        return cls.for_n_rods(n_rods, devices=devices, strategy=strategy)
