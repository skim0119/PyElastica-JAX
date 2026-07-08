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
    cr_type_1 = type("CR_BLOCK_ONE", ea.CosseratRod.__bases__, rod_dict)
    cr_type_2 = type("CR_BLOCK_TWO", ea.CosseratRod.__bases__, rod_dict)
    return cr_type_1, cr_type_2


def _build_test_rod(
    rod_type: type[ea.CosseratRod],
    *,
    start: np.ndarray | None = None,
) -> ea.CosseratRod:
    if start is None:
        start = np.zeros(3)
    return rod_type.straight_rod(
        4,
        start,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        0.5,
        0.01,
        1000.0,
        youngs_modulus=1.0e6,
        shear_modulus=1.0e6 / 1.5,
    )


def test_jax_rod2rod_op_updates_paired_rods() -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        simulator = _RodRodTestSimulator()
        rod_block = eaj.configure_rod_block()
        simulator.enable_block_supports(ea.CosseratRod, rod_block)

        rod_one = _build_test_rod(ea.CosseratRod)
        rod_two = _build_test_rod(
            ea.CosseratRod,
            start=np.array([0.1, 0.0, 0.0]),
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


def test_jax_rod2rod_op_updates_paired_rods_across_blocks() -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        body_type, muscle_type = _distinct_cosserat_rod_types()
        simulator = _RodRodTestSimulator()
        body_block = eaj.configure_rod_block()
        muscle_block = eaj.configure_rod_block()
        simulator.enable_block_supports(body_type, body_block)
        simulator.enable_block_supports(muscle_type, muscle_block)

        rod_one = _build_test_rod(body_type)
        rod_two = _build_test_rod(
            muscle_type,
            start=np.array([0.1, 0.0, 0.0]),
        )

        simulator.append(rod_one)
        simulator.append(rod_two)
        simulator.pairwise_interaction(rod_one, rod_two).using(_MarkerRodRodOp, 3.0)
        simulator.finalize()

        body_state = body_block.jax_get_state()
        muscle_state = muscle_block.jax_get_state()
        updated_states = simulator.jax_synchronize(
            (body_state, muscle_state),
            np.float64(0.0),
        )
        updated_body_state, updated_muscle_state = updated_states

        first_node = int(body_block.start_idx_in_rod_nodes[0])
        second_node = int(muscle_block.start_idx_in_rod_nodes[0])

        assert np.isclose(
            updated_body_state["external_forces"][0, first_node], 3.0
        ), "Cross-block rod-to-rod op did not update the first rod as expected."
        assert np.isclose(
            updated_muscle_state["external_forces"][1, second_node], 6.0
        ), "Cross-block rod-to-rod op did not update the second rod as expected."
