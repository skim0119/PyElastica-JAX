"""Spawn-safe helpers for parallel HDF5 rod-chunk writes."""

from __future__ import annotations

import h5py
import numpy as np


def write_rod_chunk(
    frame_path: str,
    rod_start: int,
    rod_end: int,
    positions_chunk: np.ndarray,
    write_lock,
) -> int:
    with write_lock:
        with h5py.File(frame_path, "a") as handle:
            handle["positions"][rod_start:rod_end] = positions_chunk
    return rod_end - rod_start
