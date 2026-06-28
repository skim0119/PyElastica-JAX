"""Tests for Position Verlet rollouts with blocks on different devices."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402
from elastica.modules import BaseSystemCollection  # noqa: E402


class _MultiBlockSimulator(BaseSystemCollection, eaj.JAXOpsBlock):
    pass


def _distinct_cosserat_rod_types() -> tuple[type[ea.CosseratRod], type[ea.CosseratRod]]:
    skip_attrs = {
        "__dict__",
        "__weakref__",
        "__module__",
        "__annotations__",
        "__doc__",
        "__qualname__",
    }
    rod_dict = {
        key: value
        for key, value in ea.CosseratRod.__dict__.items()
        if key not in skip_attrs
    }
    cr_type_1 = type("CR_ON_DEVICE_1", ea.CosseratRod.__bases__, rod_dict)
    cr_type_2 = type("CR_ON_DEVICE_2", ea.CosseratRod.__bases__, rod_dict)
    return cr_type_1, cr_type_2


def _build_rod(rod_type: type[ea.CosseratRod]) -> ea.CosseratRod:
    return rod_type.straight_rod(
        n_elements=8,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.0,
        base_radius=0.05,
        density=1_000.0,
        youngs_modulus=1.0e6,
    )


def test_position_verlet_integrates_two_blocks_on_separate_devices():
    devices = tuple(jax.devices("cpu")[:2])
    if len(devices) < 2:
        pytest.skip("requires at least two CPU devices")

    cr_type_1, cr_type_2 = _distinct_cosserat_rod_types()
    rod_block_1 = eaj.configure_rod_block(device=devices[0], device_dtype=np.float64)
    rod_block_2 = eaj.configure_rod_block(device=devices[1], device_dtype=np.float64)

    simulator = _MultiBlockSimulator()
    simulator.enable_block_supports(cr_type_1, rod_block_1)
    simulator.enable_block_supports(cr_type_2, rod_block_2)
    rod_1 = _build_rod(cr_type_1)
    rod_2 = _build_rod(cr_type_2)
    simulator.append(rod_1)
    simulator.append(rod_2)
    simulator.operate_block(rod_block_1).using(eaj.OneEndFixedJax)
    simulator.operate_block(rod_block_2).using(eaj.OneEndFixedJax)
    simulator.finalize()

    stepper = eaj.PositionVerletJAX()
    final_time = stepper.integrate(
        simulator,
        time=0.0,
        final_time=0.002,
        dt=0.001,
    )

    assert final_time == pytest.approx(0.002)
    assert sum(key[0] == "block" for key in stepper._compiled_rollout_cache) == 2
    jax.block_until_ready(rod_block_1.jax_get_state()["position_collection"])
    jax.block_until_ready(rod_block_2.jax_get_state()["position_collection"])

    simulator.jax_independent_block_executions = lambda: None
    with pytest.raises(AssertionError, match="Cross-block coupled operations"):
        stepper.integrate(
            simulator,
            time=0.002,
            final_time=0.003,
            dt=0.001,
        )


def test_position_verlet_compiles_each_sharded_block_partition():
    devices = tuple(jax.devices("cpu")[:2])
    if len(devices) < 2:
        pytest.skip("requires at least two CPU devices")

    simulator = _MultiBlockSimulator()
    rod_block = eaj.configure_rod_block_sharded(
        devices=devices,
        device_dtype=np.float64,
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    simulator.append(_build_rod(ea.CosseratRod))
    simulator.append(_build_rod(ea.CosseratRod))
    simulator.operate_block(rod_block).using(eaj.OneEndFixedJax)
    simulator.finalize()

    stepper = eaj.PositionVerletJAX()
    final_time = stepper.integrate(
        simulator,
        time=0.0,
        final_time=0.002,
        dt=0.001,
    )

    assert final_time == pytest.approx(0.002)
    assert sum(key[0] == "block" for key in stepper._compiled_rollout_cache) == 2
    for shard_state in rod_block.jax_get_state()["shards"]:
        jax.block_until_ready(shard_state["position_collection"])
