"""Tests for MPI-local Cosserat rod blocks."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica as ea
import elastica_jax as eaj


class _FakeComm:
    """Minimal MPI communicator stand-in for unit tests."""

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def Barrier(self) -> None:
        return None


class _RodBlockSimulator(eaj.Simulator):
    pass


def _build_rod(*, n_elements: int = 8) -> ea.CosseratRod:
    return ea.CosseratRod.straight_rod(
        n_elements=n_elements,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.0,
        base_radius=0.05,
        density=1_000.0,
        youngs_modulus=1.0e6,
    )


def test_configure_rod_block_mpi_owns_rod_round_robin() -> None:
    comm = _FakeComm(rank=1, size=4)
    rod_block = eaj.configure_rod_block_mpi(comm=comm)

    assert rod_block.comm_rank == 1
    assert rod_block.comm_size == 4
    assert rod_block.owns_rod(1)
    assert rod_block.owns_rod(5)
    assert not rod_block.owns_rod(0)
    assert not rod_block.owns_rod(2)


def test_mpi_block_finalize_and_integrate_local_gravity() -> None:
    comm = _FakeComm(rank=0, size=2)
    rod_block = eaj.configure_rod_block_mpi(comm=comm)
    simulator = _RodBlockSimulator()
    simulator.enable_block_supports(ea.CosseratRod, rod_block)

    n_rods_total = 4
    rods: list[ea.CosseratRod] = []
    for rod_index in range(n_rods_total):
        if rod_block.owns_rod(rod_index):
            rod = _build_rod()
            rods.append(rod)
            simulator.append(rod)

    dt = 1.0e-4
    simulator.operate_block(rod_block).using(eaj.OneEndFixedJax)
    simulator.operate_block(rod_block).using(
        eaj.GravityAnalyticalDamperJax,
        time_step=dt,
        uniform_damping_constant=0.5,
    )
    simulator.finalize()

    assert isinstance(rod_block, eaj._MpiCosseratRodBlock)
    assert rod_block.n_rods == 2
    np.testing.assert_array_equal(rod_block.global_rod_indices, np.array([0, 2]))

    initial_positions = [rod.position_collection.copy() for rod in rods]
    stepper = eaj.PositionVerletJAX()
    stepper.integrate(simulator, time=0.0, final_time=0.01, dt=dt)
    jax.block_until_ready(rod_block)

    rod_block.from_device()
    for rod, initial in zip(rods, initial_positions, strict=True):
        assert not np.allclose(rod.position_collection, initial)


def test_mpi_vertical_block_is_stacked_and_integrates() -> None:
    """MPI-wrapped vertical blocks must keep stacked layout through finalize."""
    from elastica_jax.modules.jax_ops_block import JAXOpsBlock

    comm = _FakeComm(rank=0, size=2)
    rod_block = eaj.configure_rod_block_mpi(
        comm=comm,
        inner_block_cls=eaj._CosseratRodVerticalMemoryBlock,
    )
    assert JAXOpsBlock._is_stacked_layout(rod_block)

    simulator = _RodBlockSimulator()
    simulator.enable_block_supports(ea.CosseratRod, rod_block)

    n_rods_total = 4
    rods: list[ea.CosseratRod] = []
    for rod_index in range(n_rods_total):
        if rod_block.owns_rod(rod_index):
            rod = _build_rod()
            rods.append(rod)
            simulator.append(rod)

    dt = 1.0e-4
    simulator.operate_block(rod_block).using(eaj.OneEndFixedJax)
    simulator.operate_block(rod_block).using(
        eaj.GravityAnalyticalDamperJax,
        time_step=dt,
        uniform_damping_constant=0.5,
    )
    simulator.finalize()

    assert isinstance(rod_block, eaj._MpiCosseratRodBlock)
    assert isinstance(rod_block._inner_block, eaj._CosseratRodVerticalMemoryBlock)
    assert JAXOpsBlock._is_stacked_layout(rod_block)
    assert rod_block.position_collection.ndim == 3

    initial_positions = [rod.position_collection.copy() for rod in rods]
    stepper = eaj.PositionVerletJAX()
    stepper.integrate(simulator, time=0.0, final_time=0.01, dt=dt)
    jax.block_until_ready(rod_block)

    rod_block.from_device()
    for rod, initial in zip(rods, initial_positions, strict=True):
        assert not np.allclose(rod.position_collection, initial)
