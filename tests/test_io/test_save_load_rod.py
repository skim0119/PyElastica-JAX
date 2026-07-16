"""Tests for target-dependent HDF5 save/load (rod seam)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402


def _make_rod() -> ea.CosseratRod:
    return ea.CosseratRod.straight_rod(
        4,
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        1.0,
        0.01,
        1000.0,
        youngs_modulus=1.0e6,
        shear_modulus=1.0e6 / 1.5,
    )


def test_save_load_rod_schema_zero_roundtrip(tmp_path: Path) -> None:
    rod = _make_rod()
    rod.position_collection[0, 0] = 1.25
    path = tmp_path / "rod.h5"

    eaj.save(rod, path, verbose=0)

    other = _make_rod()
    eaj.load(other, path)

    assert np.allclose(other.position_collection, rod.position_collection)
    assert np.allclose(other.director_collection, rod.director_collection)
    assert np.allclose(other.radius, rod.radius)


def test_save_rod_schema_zero_omits_velocity(tmp_path: Path) -> None:
    rod = _make_rod()
    path = tmp_path / "rod.h5"
    eaj.save(rod, path, verbose=0)

    import h5py

    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["schema_level"]) == 0
        assert handle.attrs["target_kind"] == "rod"
        assert "velocity_collection" not in handle["rod"]
        assert "position_collection" in handle["rod"]
