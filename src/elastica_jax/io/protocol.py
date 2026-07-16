"""Protocol for targets that can write/read HDF5 payload state."""

from __future__ import annotations

from typing import Protocol

import h5py


class Hdf5StateIO(Protocol):
    """Write or read simulation state into an open HDF5 file handle.

    File-level attributes (version, schema level, time, frame index) are owned
    by :func:`elastica_jax.io.save` / :func:`elastica_jax.io.load`. Implementations
    only create payload groups and datasets under ``handle``.
    """

    @property
    def hdf5_target_kind(self) -> str:
        """File ``target_kind`` attribute: ``rod``, ``block``, or ``simulator``."""
        ...

    def write_hdf5_state(
        self,
        handle: h5py.File,
        *,
        schema_level: int,
    ) -> None:
        """Write this target's payload groups into ``handle``."""
        ...

    def read_hdf5_state(
        self,
        handle: h5py.File,
        *,
        schema_level: int,
        check_device: bool = True,
    ) -> None:
        """Read this target's payload groups from ``handle``."""
        ...
