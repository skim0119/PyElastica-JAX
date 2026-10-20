"""Registration must preserve rod-local physics across memory layouts."""

from unittest.mock import Mock

import jax
import numpy as np
import pytest

import elastica as ea
import elastica_jax as eaj
from elastica_jax.memory_block.protocol import RodBlockProtocol, RodView

jax.config.update("jax_enable_x64", True)


class PairLoad(eaj.NoRodRodBlockOpJax):
    """Add opposite loads and read pending writes from a paired view."""

    def __init__(self, *, return_views: bool, **kwargs: object) -> None:
        self.return_views = return_views

    def jax_operation(
        self, first: RodView, second: RodView, time: float
    ) -> tuple[RodView, RodView] | None:
        """Return updated views, or commit through their shared mutations."""
        first.external_forces = first.external_forces + 1.0
        second.external_forces = second.external_forces - 2.0
        first.external_forces = first.external_forces + 3.0
        return (first, second) if self.return_views else None


def build_rod() -> ea.CosseratRod:
    """Return a small rod for one-step registration checks."""
    return ea.CosseratRod.straight_rod(
        4,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        0.5,
        0.01,
        1000.0,
        youngs_modulus=1.0e4,
    )


@pytest.mark.parametrize(
    "block_cls", [eaj._CosseratRodMemoryBlock, eaj._CosseratRodVerticalMemoryBlock]
)
@pytest.mark.parametrize("mpi_wrapper", [False, True])
def test_single_rod_boundary_preserves_other_rods(
    block_cls: type[RodBlockProtocol],
    mpi_wrapper: bool,
) -> None:
    """Fix one rod while a neighboring rod continues to translate."""
    simulator = eaj.Simulator()
    block: RodBlockProtocol
    if mpi_wrapper:
        comm = Mock()
        comm.Get_rank.return_value = 0
        comm.Get_size.return_value = 1
        block = eaj.configure_rod_block_mpi(comm=comm, inner_block_cls=block_cls)
    else:
        block = eaj.configure_rod_block(inner_block_cls=block_cls)
    simulator.enable_block_supports(ea.CosseratRod, block)
    fixed, moving = build_rod(), build_rod()
    fixed.velocity_collection[0] = 1.0
    moving.velocity_collection[0] = 1.0
    initial_positions = fixed.position_collection.copy()
    simulator.append(fixed)
    simulator.append(moving)
    simulator.operate(fixed).using(eaj.OneEndFixedJax)
    simulator.finalize()

    eaj.PositionVerletJAX().integrate(simulator, time=0.0, final_time=1.0e-5, dt=1.0e-5)
    block.from_device()

    np.testing.assert_allclose(fixed.position_collection[:, 0], initial_positions[:, 0])
    np.testing.assert_allclose(fixed.velocity_collection[:, 0], 0.0)
    np.testing.assert_allclose(moving.position_collection[0], 1.0e-5)


@pytest.mark.parametrize(
    "layouts, same_block",
    [
        ((eaj._CosseratRodMemoryBlock, eaj._CosseratRodMemoryBlock), True),
        (
            (eaj._CosseratRodVerticalMemoryBlock, eaj._CosseratRodVerticalMemoryBlock),
            True,
        ),
        ((eaj._CosseratRodMemoryBlock, eaj._CosseratRodMemoryBlock), False),
        ((eaj._CosseratRodMemoryBlock, eaj._CosseratRodVerticalMemoryBlock), False),
        ((eaj._CosseratRodVerticalMemoryBlock, eaj._CosseratRodMemoryBlock), False),
        (
            (eaj._CosseratRodVerticalMemoryBlock, eaj._CosseratRodVerticalMemoryBlock),
            False,
        ),
    ],
)
@pytest.mark.parametrize("return_views", [True, False])
def test_pairwise_loads_preserve_both_updates(
    layouts: tuple[type[RodBlockProtocol], type[RodBlockProtocol]],
    same_block: bool,
    return_views: bool,
) -> None:
    """Couple rods within a block or across blocks with different layouts."""
    simulator = eaj.Simulator()
    first_block = eaj.configure_rod_block(inner_block_cls=layouts[0])
    first, second = build_rod(), build_rod()
    if same_block:
        simulator.enable_block_supports(ea.CosseratRod, first_block)
    else:

        class SecondRod(ea.CosseratRod):
            """Select a separate block through the existing type registration."""

        second = SecondRod.straight_rod(
            4,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            0.5,
            0.01,
            1000.0,
            youngs_modulus=1.0e4,
        )
        second_block = eaj.configure_rod_block(inner_block_cls=layouts[1])
        simulator.enable_block_supports(ea.CosseratRod, first_block)
        simulator.enable_block_supports(SecondRod, second_block)
    simulator.append(first)
    simulator.append(second)
    simulator.pairwise_interaction(first, second).using(
        PairLoad, return_views=return_views
    )
    simulator.finalize()
    final_blocks = tuple(simulator.final_systems())
    initial_states = tuple(block.jax_get_state() for block in final_blocks)

    updated_states = jax.jit(simulator.jax_synchronize)(initial_states, 0.0)
    for state in initial_states:
        np.testing.assert_array_equal(state["external_forces"], 0.0)
    for block, state in zip(final_blocks, updated_states):
        block.jax_set_state(state)
        block.from_device()

    np.testing.assert_allclose(first.external_forces, 4.0)
    np.testing.assert_allclose(second.external_forces, -2.0)
