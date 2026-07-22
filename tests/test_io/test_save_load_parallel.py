"""Tests for parallel workers and MPI-coordinated Save/Load."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402


class _IoSimulator(ea.BaseSystemCollection):
    pass


class _FakeComm:
    """Minimal MPI communicator stand-in for unit tests."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size
        self.barrier_calls = 0

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def Barrier(self) -> None:
        self.barrier_calls += 1


def _layout_rods(*, n_rods: int, n_elements: int, length: float = 0.5) -> list:
    spacing = 2.0 * length
    rods = []
    for rod_index in range(n_rods):
        start = np.array([0.0, 0.0, spacing * rod_index], dtype=np.float64)
        rods.append(
            ea.CosseratRod.straight_rod(
                n_elements,
                start,
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
                length,
                0.01,
                1000.0,
                youngs_modulus=1.0e6,
                shear_modulus=1.0e6 / 1.5,
            )
        )
    return rods


def _build_block(
    *, n_rods: int = 4, n_elements: int = 4
) -> eaj._CosseratRodMemoryBlock:
    simulator = _IoSimulator()
    rod_block = eaj.configure_rod_block(
        device=jax.devices("cpu")[0],
        device_dtype=np.float64,
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    for rod in _layout_rods(n_rods=n_rods, n_elements=n_elements):
        simulator.append(rod)
    simulator.finalize()
    assert isinstance(rod_block, eaj._CosseratRodMemoryBlock)
    return rod_block


def _mutate_block(block: eaj._CosseratRodMemoryBlock) -> dict:
    state = block.jax_get_state()
    mutated = dict(state)
    mutated["position_collection"] = state["position_collection"] + 0.4
    block.jax_set_state(mutated)
    return mutated


def test_save_block_n_workers_matches_serial_schema_and_data(
    tmp_path: Path,
) -> None:
    block = _build_block()
    mutated = _mutate_block(block)
    serial_path = tmp_path / "serial.h5"
    parallel_path = tmp_path / "parallel.h5"

    eaj.save(block, serial_path, verbose=10, time=2.0, frame_idx=7, n_workers=1)
    eaj.save(block, parallel_path, verbose=10, time=2.0, frame_idx=7, n_workers=2)

    with (
        h5py.File(serial_path, "r") as serial,
        h5py.File(parallel_path, "r") as parallel,
    ):
        assert int(parallel.attrs["version"]) == int(serial.attrs["version"])
        assert int(parallel.attrs["schema_level"]) == int(serial.attrs["schema_level"])
        assert str(parallel.attrs["target_kind"]) == str(serial.attrs["target_kind"])
        assert float(parallel.attrs["time"]) == float(serial.attrs["time"])
        assert int(parallel.attrs["frame_idx"]) == int(serial.attrs["frame_idx"])
        serial_group = serial["blocks"]["0"]
        parallel_group = parallel["blocks"]["0"]
        assert set(parallel_group.keys()) == set(serial_group.keys())
        for name in serial_group:
            assert np.allclose(
                np.asarray(serial_group[name]),
                np.asarray(parallel_group[name]),
            )

    other = _build_block()
    eaj.load(other, parallel_path)
    loaded = other.jax_get_state()
    assert np.allclose(
        np.asarray(mutated["position_collection"]),
        np.asarray(loaded["position_collection"]),
    )


def test_save_with_comm_non_root_does_not_write(tmp_path: Path) -> None:
    block = _build_block(n_rods=2)
    path = tmp_path / "rank1.h5"
    non_root = _FakeComm(rank=1, size=2)

    eaj.save(block, path, verbose=10, comm=non_root)

    assert not path.exists()
    assert non_root.barrier_calls >= 2


def test_save_load_with_comm_root_roundtrip(tmp_path: Path) -> None:
    block = _build_block(n_rods=2)
    mutated = _mutate_block(block)
    path = tmp_path / "shared.h5"
    root = _FakeComm(rank=0, size=2)

    eaj.save(block, path, verbose=10, comm=root)
    assert path.exists()
    assert root.barrier_calls >= 2

    other = _build_block(n_rods=2)
    follower = _FakeComm(rank=1, size=2)
    eaj.load(other, path, comm=follower)
    assert follower.barrier_calls >= 2

    loaded = other.jax_get_state()
    assert np.allclose(
        np.asarray(mutated["position_collection"]),
        np.asarray(loaded["position_collection"]),
    )


class _Hdf5Simulator(ea.BaseSystemCollection, eaj.Hdf5IO):
    pass


def _finalize_simulator() -> tuple[_Hdf5Simulator, eaj._CosseratRodMemoryBlock]:
    simulator = _Hdf5Simulator()
    rod_block = eaj.configure_rod_block(
        device=jax.devices("cpu")[0],
        device_dtype=np.float64,
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    for rod in _layout_rods(n_rods=4, n_elements=3):
        simulator.append(rod)
    simulator.finalize()
    assert isinstance(rod_block, eaj._CosseratRodMemoryBlock)
    return simulator, rod_block


def test_save_simulator_n_workers_matches_serial(tmp_path: Path) -> None:
    simulator, block = _finalize_simulator()
    mutated = _mutate_block(block)
    serial_path = tmp_path / "sim_serial.h5"
    parallel_path = tmp_path / "sim_parallel.h5"

    eaj.save(simulator, serial_path, verbose=10, n_workers=1)
    eaj.save(simulator, parallel_path, verbose=10, n_workers=2)

    with (
        h5py.File(serial_path, "r") as serial,
        h5py.File(parallel_path, "r") as parallel,
    ):
        assert str(parallel.attrs["target_kind"]) == "simulator"
        assert set(parallel["blocks"]["0"].keys()) == set(serial["blocks"]["0"].keys())
        for name in serial["blocks"]["0"]:
            assert np.allclose(
                np.asarray(serial["blocks"]["0"][name]),
                np.asarray(parallel["blocks"]["0"][name]),
            )

    other_sim, other_block = _finalize_simulator()
    eaj.load(other_sim, parallel_path)
    assert np.allclose(
        np.asarray(mutated["position_collection"]),
        np.asarray(other_block.jax_get_state()["position_collection"]),
    )
