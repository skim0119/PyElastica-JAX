from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import jax
import pytest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402


class _CheckpointSimulator(ea.BaseSystemCollection):
    pass


def _sharded_devices() -> tuple[jax.Device, ...]:
    devices = tuple(jax.devices("cpu")[:2])
    if len(devices) < 2:
        pytest.skip("requires at least two CPU devices")
    return devices


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
    devices: Sequence[jax.Device],
    block_checkpoint: Path | None = None,
) -> eaj._ShardedCosseratRodBlock:
    simulator = _CheckpointSimulator()
    rod_block_cls = eaj.configure_rod_block_sharded(
        devices=devices,
        device_dtype=np.float64,
        block_checkpoint=block_checkpoint,
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block_cls)
    _append_layout_rods(simulator, n_rods=n_rods, n_elements=n_elements)
    simulator.finalize()
    block = tuple(simulator.final_systems())[0]
    assert isinstance(block, eaj._ShardedCosseratRodBlock)
    return block


def test_block_checkpoint_roundtrip(tmp_path: Path) -> None:
    devices = _sharded_devices()
    checkpoint_path = tmp_path / "block.h5"
    block = _build_block(
        n_rods=2,
        n_elements=4,
        devices=devices,
        block_checkpoint=checkpoint_path,
    )
    reference = block.jax_get_state()

    reloaded = _build_block(
        n_rods=2,
        n_elements=4,
        devices=devices,
        block_checkpoint=checkpoint_path,
    )
    loaded = reloaded.jax_get_state()
    for shard_index, shard_state in enumerate(reference["shards"]):
        reloaded_shard = loaded["shards"][shard_index]
        for key in shard_state:
            assert np.allclose(
                np.asarray(shard_state[key]),
                np.asarray(reloaded_shard[key]),
            )


def test_finalize_skips_rod_packing_when_checkpoint_is_loaded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    devices = _sharded_devices()
    checkpoint_path = tmp_path / "block.h5"
    _build_block(
        n_rods=2,
        n_elements=4,
        devices=devices,
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
        devices=devices,
        block_checkpoint=checkpoint_path,
    )
    assert calls["count"] == 0


def test_execution_mesh_for_block_checkpoint_reads_layout(tmp_path: Path) -> None:
    devices = _sharded_devices()
    checkpoint_path = tmp_path / "block.h5"
    _build_block(
        n_rods=2,
        n_elements=4,
        devices=devices,
        block_checkpoint=checkpoint_path,
    )
    layout = eaj.read_block_checkpoint_layout(checkpoint_path)
    resolved = eaj.execution_mesh_for_block_checkpoint(
        checkpoint_path,
        mesh_name="auto",
        backend="cpu",
        n_rods=99,
    )
    assert len(resolved) == layout.n_shards == 2
    assert layout.rod_to_shard is not None
