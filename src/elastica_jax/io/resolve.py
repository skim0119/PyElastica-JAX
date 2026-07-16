"""Resolve a Save/Load target to an :class:`Hdf5StateIO` implementation."""

from __future__ import annotations

from elastica_jax.io.protocol import Hdf5StateIO
from elastica_jax.io.rod_adapter import RodHdf5Adapter


def hdf5_io_for(target: object) -> Hdf5StateIO:
    """Return an HDF5 writer/reader for ``target``.

    Blocks and simulators that implement ``write_hdf5_state`` /
    ``read_hdf5_state`` are used directly. Host rods are wrapped in
    :class:`RodHdf5Adapter`.
    """
    write = getattr(target, "write_hdf5_state", None)
    read = getattr(target, "read_hdf5_state", None)
    if callable(write) and callable(read):
        return target  # type: ignore[return-value]
    return RodHdf5Adapter(target)
