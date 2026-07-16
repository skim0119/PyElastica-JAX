"""Public Save/Load orchestration for HDF5 simulation I/O."""

from __future__ import annotations

from pathlib import Path

import h5py

from elastica_jax.io.resolve import hdf5_io_for
from elastica_jax.io.schema import IO_VERSION, TARGET_ROD, fields_for_schema_level


def save(
    target: object,
    path: Path | str,
    *,
    time: float | None = None,
    frame_idx: int | None = None,
    verbose: int = 0,
) -> None:
    """Write ``target`` state to a single HDF5 file.

    Does not call ``from_device`` / ``to_device``. Opens the file, writes
    file-level attributes, then delegates payload groups to the target's
    :class:`~elastica_jax.io.protocol.Hdf5StateIO` implementation.

    Parameters
    ----------
    target :
        A rod, rod block, or simulator implementing HDF5 state I/O (host rods
        are adapted automatically).
    path :
        Destination HDF5 path.
    time :
        Optional simulation time stored as a file attribute.
    frame_idx :
        Optional frame index stored as a file attribute.
    verbose :
        Schema level (``0``, ``1``, or ``10``).

    Raises
    ------
    AssertionError
        If ``verbose`` is unsupported or ``target`` cannot be adapted.
    """
    schema_level = int(verbose)
    fields_for_schema_level(schema_level)  # validate early
    io = hdf5_io_for(target)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as handle:
        handle.attrs["version"] = IO_VERSION
        handle.attrs["schema_level"] = schema_level
        handle.attrs["target_kind"] = io.hdf5_target_kind
        if time is not None:
            handle.attrs["time"] = float(time)
        if frame_idx is not None:
            handle.attrs["frame_idx"] = int(frame_idx)
        io.write_hdf5_state(handle, schema_level=schema_level)


def load(
    target: object,
    path: Path | str,
    *,
    check_device: bool = True,
) -> None:
    """Restore ``target`` state from an HDF5 file written by :func:`save`.

    Does not call ``from_device`` / ``to_device``. Block and simulator loads
    require schema level ``10``.

    Parameters
    ----------
    target :
        A rod, rod block, or simulator matching the file layout.
    path :
        Source HDF5 path.
    check_device :
        When True (default), require matching JAX platform and device id for
        block/simulator files.

    Raises
    ------
    AssertionError
        If the file version/kind is unsupported, schema level is not ``10``
        for a block/simulator file, or device metadata does not match.
    """
    io = hdf5_io_for(target)
    in_path = Path(path)
    with h5py.File(in_path, "r") as handle:
        assert int(handle.attrs["version"]) == IO_VERSION, (
            f"Unsupported I/O version in {in_path}."
        )
        schema_level = int(handle.attrs["schema_level"])
        target_kind = str(handle.attrs["target_kind"])
        assert target_kind == io.hdf5_target_kind, (
            f"File target_kind {target_kind!r} does not match "
            f"target kind {io.hdf5_target_kind!r}."
        )
        if target_kind != TARGET_ROD:
            assert schema_level == 10, (
                f"Block/simulator Load requires schema level 10 for resume; "
                f"file has schema_level={schema_level}."
            )
        io.read_hdf5_state(
            handle,
            schema_level=schema_level,
            check_device=check_device,
        )
