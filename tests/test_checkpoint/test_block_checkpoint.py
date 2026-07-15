from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402


class _CheckpointSimulator(ea.BaseSystemCollection):
    pass


def _append_layout_rods(
    simulator: _CheckpointSimulator,
    *,
    n_rods: int,
    n_elements: int,
) -> None:
    for rod in eaj.layout_rods_for_block(
        n_rods=n_rods,
        n_elements=n_elements,
        length=0.5,
        radius=0.01,
        density=1000.0,
        youngs_modulus=1.0e6,
        shear_modulus=1.0e6 / 1.5,
    ):
        simulator.append(rod)


def _build_block(
    *,
    n_rods: int,
    n_elements: int,
    block_checkpoint: Path | None = None,
) -> eaj._CosseratRodMemoryBlock:
    simulator = _CheckpointSimulator()
    rod_block = eaj.configure_rod_block(
        device=jax.devices("cpu")[0],
        device_dtype=np.float64,
        block_checkpoint=block_checkpoint,
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    _append_layout_rods(simulator, n_rods=n_rods, n_elements=n_elements)
    simulator.finalize()
    assert isinstance(rod_block, eaj._CosseratRodMemoryBlock)
    return rod_block


def test_block_checkpoint_roundtrip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "block.h5"
    block = _build_block(
        n_rods=2,
        n_elements=4,
        block_checkpoint=checkpoint_path,
    )
    reference = block.jax_get_state()

    reloaded = _build_block(
        n_rods=2,
        n_elements=4,
        block_checkpoint=checkpoint_path,
    )
    loaded = reloaded.jax_get_state()
    for key in reference:
        assert np.allclose(
            np.asarray(reference[key]),
            np.asarray(loaded[key]),
        )


def test_finalize_skips_rod_packing_when_checkpoint_is_loaded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "block.h5"
    _build_block(
        n_rods=2,
        n_elements=4,
        block_checkpoint=checkpoint_path,
    )

    calls = {"count": 0}
    original = (
        eaj.memory_block.memory_block_rod_jax._compute_sigma_kappa_for_blockstructure
    )

    def _counting_compute_sigma_kappa(target):
        calls["count"] += 1
        return original(target)

    monkeypatch.setattr(
        eaj.memory_block.memory_block_rod_jax,
        "_compute_sigma_kappa_for_blockstructure",
        _counting_compute_sigma_kappa,
    )
    _build_block(
        n_rods=2,
        n_elements=4,
        block_checkpoint=checkpoint_path,
    )
    assert calls["count"] == 0


def test_read_block_checkpoint_layout_reads_saved_metadata(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "block.h5"
    _build_block(
        n_rods=3,
        n_elements=5,
        block_checkpoint=checkpoint_path,
    )

    layout = eaj.read_block_checkpoint_layout(checkpoint_path)
    assert layout.n_rods == 3
    assert layout.n_elements_per_rod == 5
    assert layout.dtype == "float64"
