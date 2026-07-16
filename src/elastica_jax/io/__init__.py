"""Target-dependent HDF5 save and load for rods, blocks, and simulators."""

from __future__ import annotations

from elastica_jax.io.api import load, save
from elastica_jax.io.hdf5_io_mixin import Hdf5IO
from elastica_jax.io.protocol import Hdf5StateIO
from elastica_jax.io.schema import (
    IO_VERSION,
    SCHEMA_LEVEL_0,
    SCHEMA_LEVEL_1,
    fields_for_schema_level,
)

__all__ = [
    "Hdf5IO",
    "Hdf5StateIO",
    "IO_VERSION",
    "SCHEMA_LEVEL_0",
    "SCHEMA_LEVEL_1",
    "fields_for_schema_level",
    "load",
    "save",
]
