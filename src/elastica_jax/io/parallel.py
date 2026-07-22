"""Chunked parallel HDF5 dataset writes for large Saves."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np


class _Lock(Protocol):
    def __enter__(self) -> object: ...

    def __exit__(self, *args: object) -> object: ...


def _last_axis_indexer(ndim: int, start: int, end: int) -> tuple[slice, ...]:
    assert ndim >= 1, "Hyperslab writes require at least one axis."
    return (slice(None),) * (ndim - 1) + (slice(start, end),)


def write_dataset_chunk(
    path: str,
    dataset_path: str,
    start: int,
    end: int,
    chunk: np.ndarray,
    write_lock: _Lock,
) -> None:
    """Write one last-axis hyperslab into an existing dataset."""
    indexer = _last_axis_indexer(chunk.ndim, start, end)
    with write_lock:
        with h5py.File(path, "a") as handle:
            handle[dataset_path][indexer] = chunk


def write_group_arrays_parallel(
    path: Path | str,
    group_path: str,
    arrays: dict[str, np.ndarray],
    *,
    n_workers: int,
) -> None:
    """Fill pre-created datasets under ``group_path`` using worker processes.

    Arrays are partitioned along their last axis. When an array is too small to
    split, or ``n_workers <= 1``, the fill is serial.
    """
    assert n_workers >= 1, f"n_workers must be >= 1; got {n_workers}."
    out_path = Path(path)
    if n_workers == 1 or not arrays:
        with h5py.File(out_path, "a") as handle:
            group = handle[group_path]
            for name, array in arrays.items():
                group[name][...] = array
        return

    tasks: list[tuple[str, str, int, int, np.ndarray]] = []
    for name, array in arrays.items():
        dataset_path = f"{group_path}/{name}"
        if array.ndim == 0 or array.shape[-1] <= 1:
            with h5py.File(out_path, "a") as handle:
                handle[dataset_path][...] = array
            continue
        extent = int(array.shape[-1])
        chunk_count = min(n_workers, extent)
        span = max(1, (extent + chunk_count - 1) // chunk_count)
        for start in range(0, extent, span):
            end = min(start + span, extent)
            indexer = _last_axis_indexer(array.ndim, start, end)
            tasks.append(
                (
                    str(out_path),
                    dataset_path,
                    start,
                    end,
                    np.asarray(array[indexer]).copy(),
                )
            )

    if not tasks:
        return

    process_count = min(n_workers, len(tasks))
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        write_lock = manager.Lock()
        with ctx.Pool(processes=process_count) as pool:
            pool.starmap(
                write_dataset_chunk,
                [
                    (file_path, dataset_path, start, end, chunk, write_lock)
                    for file_path, dataset_path, start, end, chunk in tasks
                ],
            )
