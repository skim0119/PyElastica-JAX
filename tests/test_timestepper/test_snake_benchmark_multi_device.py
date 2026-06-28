"""Smoke tests for the snake benchmark multi-device builders."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "snake-self-activate-single-node"
)
sys.path.insert(0, str(BENCHMARK_DIR))

import _jax_snake_common as snake_common  # noqa: E402


def _two_cpu_devices() -> tuple[jax.Device, jax.Device]:
    devices = tuple(jax.devices("cpu")[:2])
    if len(devices) < 2:
        pytest.skip("requires at least two CPU devices")
    return devices[0], devices[1]


def test_gpu2x_builder_creates_two_explicit_blocks() -> None:
    devices = _two_cpu_devices()

    simulator, rod_blocks = snake_common.build_jax_sim_gpu2x(
        devices=devices,
        device_dtype=np.dtype(np.float64),
        n_snakes=2,
    )
    final_systems = tuple(simulator.final_systems())

    assert final_systems == rod_blocks
    assert tuple(block.position_collection_device.device for block in rod_blocks) == (
        devices
    )
    snake_common.integrate_jax_block_rollout(
        simulator,
        rod_blocks,
        steps=1,
        warmup_runs=1,
    )


def test_gpu2x_sharded_builder_creates_one_block_with_two_shards() -> None:
    devices = _two_cpu_devices()

    simulator, rod_block = snake_common.build_jax_sim(
        device=devices,
        device_dtype=np.dtype(np.float64),
        n_snakes=2,
        sharded=True,
    )
    shard_states = rod_block.jax_get_state()["shards"]

    assert tuple(simulator.final_systems()) == (rod_block,)
    assert len(shard_states) == 2
    assert (
        tuple(
            next(iter(jax.tree_util.tree_leaves(state)[0].devices()))
            for state in shard_states
        )
        == devices
    )
    snake_common.integrate_jax_block_rollout(
        simulator,
        (rod_block,),
        steps=1,
        warmup_runs=1,
    )
