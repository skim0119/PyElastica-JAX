from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import elastica as ea
import elastica_jax as eaj


class _MarkerRodRodOp(eaj.NoRodRodBlockOpJax):
    def __init__(
        self, increment: float, *, _first_system=None, _second_system=None
    ) -> None:
        del _first_system, _second_system
        self.increment = increment

    def jax_operation(self, rod_one_view, rod_two_view, time):
        del time
        rod_one_view.external_forces = rod_one_view.external_forces.at[:, 0].add(
            self.increment
        )
        rod_two_view.external_forces = rod_two_view.external_forces.at[:, 0].add(
            2.0 * self.increment
        )
        return rod_one_view, rod_two_view


class _RodRodTestSimulator(ea.BaseSystemCollection, eaj.JAXInteraction):
    pass


def test_jax_rod2rod_op_updates_paired_rods() -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        simulator = _RodRodTestSimulator()
        rod_block = eaj.configure_rod_block()
        simulator.enable_block_supports(ea.CosseratRod, rod_block)

        rod_one = ea.CosseratRod.straight_rod(
            4,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            0.5,
            0.01,
            1000.0,
            youngs_modulus=1.0e6,
            shear_modulus=1.0e6 / 1.5,
        )
        rod_two = ea.CosseratRod.straight_rod(
            4,
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            0.5,
            0.01,
            1000.0,
            youngs_modulus=1.0e6,
            shear_modulus=1.0e6 / 1.5,
        )

        simulator.append(rod_one)
        simulator.append(rod_two)
        simulator.pairwise_interaction(rod_one, rod_two).using(_MarkerRodRodOp, 3.0)
        simulator.finalize()

        block = tuple(simulator.final_systems())[0]
        state = block.jax_get_state()
        updated_states = simulator.jax_synchronize((state,), np.float64(0.0))
        updated_state = updated_states[0]

        first_node = int(block.start_idx_in_rod_nodes[0])
        second_node = int(block.start_idx_in_rod_nodes[1])

        assert np.isclose(
            updated_state["external_forces"][0, first_node], 3.0
        ), "Rod-to-rod op did not update the first rod as expected."
        assert np.isclose(
            updated_state["external_forces"][1, second_node], 6.0
        ), "Rod-to-rod op did not update the second rod as expected."
