"""Tests for target-dependent HDF5 save/load (block seam)."""

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
    *, n_rods: int = 2, n_elements: int = 4
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


def test_save_load_block_schema_ten_roundtrip(tmp_path: Path) -> None:
    block = _build_block()
    state = block.jax_get_state()
    mutated = dict(state)
    mutated["position_collection"] = state["position_collection"] + 0.3
    block.jax_set_state(mutated)
    path = tmp_path / "block.h5"

    eaj.save(block, path, verbose=10, time=1.5, frame_idx=3)

    other = _build_block()
    eaj.load(other, path)

    loaded = other.jax_get_state()
    for key in mutated:
        assert np.allclose(np.asarray(mutated[key]), np.asarray(loaded[key]))

    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["schema_level"]) == 10
        assert handle.attrs["target_kind"] == "block"
        assert float(handle.attrs["time"]) == 1.5
        assert int(handle.attrs["frame_idx"]) == 3
        assert "jax_platform" in handle["blocks"]["0"].attrs
        assert "jax_device_id" in handle["blocks"]["0"].attrs


def test_save_block_schema_zero_omits_velocity(tmp_path: Path) -> None:
    block = _build_block()
    path = tmp_path / "block.h5"
    eaj.save(block, path, verbose=0)

    with h5py.File(path, "r") as handle:
        group = handle["blocks"]["0"]
        assert "position_collection" in group
        assert "velocity_collection" not in group


def test_load_block_rejects_device_mismatch(tmp_path: Path, monkeypatch) -> None:
    block = _build_block()
    path = tmp_path / "block.h5"
    eaj.save(block, path, verbose=10)

    other = _build_block()

    def _wrong_metadata(_block):
        return "cpu", 99

    monkeypatch.setattr(
        "elastica_jax.io.block_state.current_device_metadata",
        _wrong_metadata,
    )
    with pytest.raises(AssertionError, match="device id"):
        eaj.load(other, path, check_device=True)

    eaj.load(other, path, check_device=False)


def test_load_block_rejects_non_resume_schema(tmp_path: Path) -> None:
    block = _build_block()
    path = tmp_path / "block.h5"
    eaj.save(block, path, verbose=0)

    other = _build_block()
    with pytest.raises(AssertionError, match="schema level 10"):
        eaj.load(other, path)
