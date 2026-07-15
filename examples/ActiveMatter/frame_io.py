"""Parallel HDF5 frame I/O for active-matter rod stacks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    import elastica_jax as eaj


def extract_stacked_positions(
    block: eaj._CosseratRodMemoryBlock,
    *,
    n_snakes: int,
) -> np.ndarray:
    """Return rod node positions as ``(n_snakes, n_nodes, 3)``."""
    positions = np.asarray(block.device_state["position_collection"])
    assert positions.shape[0] == 3, "Rod positions must be stored as (3, n_nodes)."
    n_nodes_per_rod = int(
        block.end_idx_in_rod_nodes[0] - block.start_idx_in_rod_nodes[0]
    )
    stacked = np.empty((n_snakes, n_nodes_per_rod, 3), dtype=positions.dtype)
    for rod_idx in range(n_snakes):
        start = int(block.start_idx_in_rod_nodes[rod_idx])
        end = int(block.end_idx_in_rod_nodes[rod_idx])
        stacked[rod_idx] = positions[:, start:end].T
    return stacked


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


def save_frame_positions(
    frame_path: Path,
    positions: np.ndarray,
    *,
    time: float,
    frame_idx: int,
    attrs: dict[str, Any] | None = None,
    n_workers: int = 1,
) -> None:
    """Write one simulation frame with chunked parallel hyperslab I/O."""
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    n_snakes, n_nodes, spatial_dim = positions.shape
    assert spatial_dim == 3, "Positions must have trailing dimension 3."
    chunk_shape = (1, n_nodes, spatial_dim)

    with h5py.File(frame_path, "w") as handle:
        handle.create_dataset(
            "positions",
            shape=(n_snakes, n_nodes, spatial_dim),
            dtype=positions.dtype,
            chunks=chunk_shape,
        )
        handle.attrs["time"] = float(time)
        handle.attrs["frame_idx"] = int(frame_idx)
        handle.attrs["n_snakes"] = int(n_snakes)
        handle.attrs["n_nodes"] = int(n_nodes)
        if attrs is not None:
            for key, value in attrs.items():
                handle.attrs[key] = value

    if n_workers <= 1 or n_snakes <= 1:
        with h5py.File(frame_path, "a") as handle:
            handle["positions"][...] = positions
        return

    import multiprocessing as mp

    chunk_count = min(n_workers, n_snakes)
    rods_per_chunk = max(1, (n_snakes + chunk_count - 1) // chunk_count)
    tasks: list[tuple[str, int, int, np.ndarray]] = []
    for rod_start in range(0, n_snakes, rods_per_chunk):
        rod_end = min(rod_start + rods_per_chunk, n_snakes)
        tasks.append(
            (
                str(frame_path),
                rod_start,
                rod_end,
                positions[rod_start:rod_end].copy(),
            )
        )

    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        write_lock = manager.Lock()
        with ctx.Pool(processes=chunk_count) as pool:
            pool.starmap(
                write_rod_chunk,
                [
                    (path, start, end, chunk, write_lock)
                    for path, start, end, chunk in tasks
                ],
            )


def save_simulation_metadata(
    metadata_path: Path,
    *,
    case_name: str,
    parameters: Any,
    fps: float,
    steps_per_frame: int,
    frame_dt: float,
    wall_origins: np.ndarray,
    wall_normals: np.ndarray,
    seed: int,
    run_name: str | None = None,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_values = asdict(parameters)
    with h5py.File(metadata_path, "w") as handle:
        handle.attrs["case"] = case_name
        handle.attrs["fps"] = float(fps)
        handle.attrs["steps_per_frame"] = int(steps_per_frame)
        handle.attrs["frame_dt"] = float(frame_dt)
        handle.attrs["seed"] = int(seed)
        if run_name is not None:
            handle.attrs["run_name"] = run_name
        for key, value in parameter_values.items():
            if isinstance(value, (str, int, float, bool, np.number)):
                handle.attrs[f"param_{key}"] = value
        handle.create_dataset("wall_origins", data=np.asarray(wall_origins))
        handle.create_dataset("wall_normals", data=np.asarray(wall_normals))


def load_frame_positions(frame_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(frame_path, "r") as handle:
        positions = np.asarray(handle["positions"])
        attrs = {key: handle.attrs[key] for key in handle.attrs}
    return positions, attrs


def load_simulation_metadata(metadata_path: Path) -> dict[str, Any]:
    with h5py.File(metadata_path, "r") as handle:
        metadata: dict[str, Any] = {key: handle.attrs[key] for key in handle.attrs}
        metadata["wall_origins"] = np.asarray(handle["wall_origins"])
        metadata["wall_normals"] = np.asarray(handle["wall_normals"])
    return metadata


def list_frame_paths(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("frame_*.h5"))
