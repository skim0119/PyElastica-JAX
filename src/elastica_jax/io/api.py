"""Public Save/Load orchestration for HDF5 simulation I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import h5py
import numpy as np

from elastica_jax.io.block_state import _RodBlockState, write_block_into
from elastica_jax.io.parallel import write_group_arrays_parallel
from elastica_jax.io.resolve import hdf5_io_for
from elastica_jax.io.schema import (
    BLOCKS_GROUP,
    IO_VERSION,
    TARGET_BLOCK,
    TARGET_ROD,
    TARGET_SIMULATOR,
    fields_for_schema_level,
)


class MpiComm(Protocol):
    """Minimal MPI communicator surface used by Save/Load coordination."""

    def Get_rank(self) -> int: ...

    def Barrier(self) -> None: ...


def _mpi_is_root(comm: MpiComm | None) -> bool:
    if comm is None:
        return True
    return int(comm.Get_rank()) == 0


def _mpi_barrier(comm: MpiComm | None) -> None:
    if comm is not None:
        comm.Barrier()


def _write_file_attrs(
    handle: h5py.File,
    *,
    schema_level: int,
    target_kind: str,
    time: float | None,
    frame_idx: int | None,
) -> None:
    handle.attrs["version"] = IO_VERSION
    handle.attrs["schema_level"] = schema_level
    handle.attrs["target_kind"] = target_kind
    if time is not None:
        handle.attrs["time"] = float(time)
    if frame_idx is not None:
        handle.attrs["frame_idx"] = int(frame_idx)


def _save_serial(
    target: object,
    out_path: Path,
    *,
    schema_level: int,
    time: float | None,
    frame_idx: int | None,
) -> None:
    io = hdf5_io_for(target)
    with h5py.File(out_path, "w") as handle:
        _write_file_attrs(
            handle,
            schema_level=schema_level,
            target_kind=io.hdf5_target_kind,
            time=time,
            frame_idx=frame_idx,
        )
        io.write_hdf5_state(handle, schema_level=schema_level)


def _simulator_rod_blocks(simulator: object) -> list[_RodBlockState]:
    getter = getattr(simulator, "_hdf5_rod_blocks", None)
    if callable(getter):
        return getter()  # type: ignore[no-any-return]
    blocks: list[_RodBlockState] = []
    for system in simulator.final_systems():  # type: ignore[attr-defined]
        if callable(getattr(system, "write_hdf5_state", None)) and callable(
            getattr(system, "read_hdf5_state", None)
        ):
            blocks.append(system)  # type: ignore[arg-type]
    assert blocks, "Simulator has no HDF5-capable rod blocks to save or load."
    return blocks


def _save_block_or_simulator_parallel(
    target: object,
    out_path: Path,
    *,
    schema_level: int,
    target_kind: str,
    time: float | None,
    frame_idx: int | None,
    n_workers: int,
) -> None:
    if target_kind == TARGET_BLOCK:
        blocks: list[_RodBlockState] = [target]  # type: ignore[list-item]
    else:
        assert target_kind == TARGET_SIMULATOR, (
            f"Parallel Save supports block/simulator; got {target_kind!r}."
        )
        blocks = _simulator_rod_blocks(target)

    pending: list[tuple[str, dict[str, np.ndarray]]] = []
    with h5py.File(out_path, "w") as handle:
        _write_file_attrs(
            handle,
            schema_level=schema_level,
            target_kind=target_kind,
            time=time,
            frame_idx=frame_idx,
        )
        parent = handle.create_group(BLOCKS_GROUP)
        for index, block in enumerate(blocks):
            arrays = write_block_into(
                parent,
                str(index),
                block,
                schema_level=schema_level,
                fill_arrays=False,
            )
            pending.append((f"{BLOCKS_GROUP}/{index}", arrays))

    for group_path, arrays in pending:
        write_group_arrays_parallel(
            out_path,
            group_path,
            arrays,
            n_workers=n_workers,
        )


def save(
    target: object,
    path: Path | str,
    *,
    time: float | None = None,
    frame_idx: int | None = None,
    n_workers: int = 1,
    comm: MpiComm | None = None,
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
    n_workers :
        Worker processes for chunked parallel writes of block/simulator
        payloads. ``1`` (default) is serial. Rod Saves always use the serial
        path.
    comm :
        Optional MPI communicator. When provided, ranks barrier around the
        operation and only rank 0 writes the file (shared-path consistency).
    verbose :
        Schema level (``0``, ``1``, or ``10``).

    Raises
    ------
    AssertionError
        If ``verbose`` is unsupported, ``n_workers < 1``, or ``target`` cannot
        be adapted.
    """
    assert n_workers >= 1, f"n_workers must be >= 1; got {n_workers}."
    schema_level = int(verbose)
    fields_for_schema_level(schema_level)  # validate early
    io = hdf5_io_for(target)
    out_path = Path(path)

    _mpi_barrier(comm)
    if not _mpi_is_root(comm):
        _mpi_barrier(comm)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_kind = io.hdf5_target_kind
    if n_workers > 1 and target_kind in {TARGET_BLOCK, TARGET_SIMULATOR}:
        _save_block_or_simulator_parallel(
            target,
            out_path,
            schema_level=schema_level,
            target_kind=target_kind,
            time=time,
            frame_idx=frame_idx,
            n_workers=n_workers,
        )
    else:
        _save_serial(
            target,
            out_path,
            schema_level=schema_level,
            time=time,
            frame_idx=frame_idx,
        )
    _mpi_barrier(comm)


def load(
    target: object,
    path: Path | str,
    *,
    check_device: bool = True,
    comm: MpiComm | None = None,
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
    comm :
        Optional MPI communicator. When provided, ranks barrier before and
        after Load so a shared-path resume stays consistent across ranks.

    Raises
    ------
    AssertionError
        If the file version/kind is unsupported, schema level is not ``10``
        for a block/simulator file, or device metadata does not match.
    """
    io = hdf5_io_for(target)
    in_path = Path(path)

    _mpi_barrier(comm)
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
    _mpi_barrier(comm)
