"""Tests for simulator target save/load (single file, multi-block ready)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402


def _layout_rods(*, n_rods: int, n_elements: int, length: float = 0.4) -> list:
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


def _finalize_simulator() -> tuple[eaj.Simulator, eaj._CosseratRodMemoryBlock]:
    simulator = eaj.Simulator()
    rod_block = eaj.configure_rod_block(
        device=jax.devices("cpu")[0],
        device_dtype=np.float64,
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    for rod in _layout_rods(n_rods=2, n_elements=3):
        simulator.append(rod)
    simulator.finalize()
    assert isinstance(rod_block, eaj._CosseratRodMemoryBlock)
    return simulator, rod_block


def test_save_load_simulator_roundtrip(tmp_path: Path) -> None:
    simulator, block = _finalize_simulator()
    state = block.jax_get_state()
    mutated = dict(state)
    mutated["position_collection"] = state["position_collection"] + 0.2
    block.jax_set_state(mutated)
    path = tmp_path / "sim.h5"

    eaj.save(simulator, path, verbose=10)

    other_sim, other_block = _finalize_simulator()
    eaj.load(other_sim, path)

    loaded = other_block.jax_get_state()
    assert np.allclose(
        np.asarray(mutated["position_collection"]),
        np.asarray(loaded["position_collection"]),
    )
    with h5py.File(path, "r") as handle:
        assert handle.attrs["target_kind"] == "simulator"
        assert "0" in handle["blocks"]
