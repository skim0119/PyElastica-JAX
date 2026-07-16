"""Simulator mixin that implements HDF5 Save/Load over finalized blocks."""

from __future__ import annotations

import h5py

from elastica_jax.io.block_state import (
    _RodBlockState,
    read_block_from,
    write_block_into,
)
from elastica_jax.io.schema import BLOCKS_GROUP, TARGET_SIMULATOR


class Hdf5IO:
    """Mixin for simulators: Save/Load all rod blocks into one HDF5 file.

    Mix into the simulator class alongside other JAX mixins::

        class Sim(ea.BaseSystemCollection, eaj.JAXOpsBlock, eaj.Hdf5IO):
            pass
    """

    @property
    def hdf5_target_kind(self) -> str:
        return TARGET_SIMULATOR

    def _hdf5_rod_blocks(self) -> list[_RodBlockState]:
        blocks: list[_RodBlockState] = []
        for system in self.final_systems():  # type: ignore[attr-defined]
            if callable(getattr(system, "write_hdf5_state", None)) and callable(
                getattr(system, "read_hdf5_state", None)
            ):
                blocks.append(system)  # type: ignore[arg-type]
        assert blocks, "Simulator has no HDF5-capable rod blocks to save or load."
        return blocks

    def write_hdf5_state(
        self,
        handle: h5py.File,
        *,
        schema_level: int,
    ) -> None:
        parent = handle.create_group(BLOCKS_GROUP)
        for index, block in enumerate(self._hdf5_rod_blocks()):
            write_block_into(parent, str(index), block, schema_level=schema_level)

    def read_hdf5_state(
        self,
        handle: h5py.File,
        *,
        schema_level: int,
        check_device: bool = True,
    ) -> None:
        del schema_level  # resume gate is enforced by load()
        blocks = self._hdf5_rod_blocks()
        blocks_group = handle[BLOCKS_GROUP]
        assert len(blocks_group) == len(blocks), (
            f"File has {len(blocks_group)} blocks but simulator has {len(blocks)}."
        )
        for index, block in enumerate(blocks):
            read_block_from(
                blocks_group[str(index)],
                block,
                check_device=check_device,
            )
