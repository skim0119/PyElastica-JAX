"""HDF5 adapter for upstream host ``CosseratRod`` objects."""

from __future__ import annotations

import h5py
import numpy as np

from elastica_jax.io.block_state import read_array_group, write_array_group
from elastica_jax.io.schema import ROD_GROUP, TARGET_ROD, fields_for_schema_level


class RodHdf5Adapter:
    """Adapt a host rod to :class:`~elastica_jax.io.protocol.Hdf5StateIO`.

    Upstream ``CosseratRod`` cannot grow methods in this package, so Save/Load
    wraps it in this adapter.
    """

    def __init__(self, rod: object) -> None:
        assert hasattr(rod, "position_collection") and hasattr(rod, "n_elems"), (
            "RodHdf5Adapter requires a host rod with position_collection and n_elems."
        )
        self._rod = rod

    @property
    def hdf5_target_kind(self) -> str:
        return TARGET_ROD

    def write_hdf5_state(
        self,
        handle: h5py.File,
        *,
        schema_level: int,
    ) -> None:
        field_names = fields_for_schema_level(schema_level)
        arrays: dict[str, np.ndarray] = {}
        for name in field_names:
            assert hasattr(self._rod, name), f"Rod is missing collection {name!r}."
            arrays[name] = np.asarray(getattr(self._rod, name))
        group = handle.create_group(ROD_GROUP)
        write_array_group(group, arrays)

    def read_hdf5_state(
        self,
        handle: h5py.File,
        *,
        schema_level: int,
        check_device: bool = True,
    ) -> None:
        del check_device  # rods have no device metadata
        field_names = fields_for_schema_level(schema_level)
        arrays = read_array_group(handle[ROD_GROUP], field_names)
        for name, array in arrays.items():
            np.copyto(getattr(self._rod, name), array)
