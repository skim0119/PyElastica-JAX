"""Combined gravity and damping must match separate operator registration."""

import jax
import numpy as np
import pytest

import elastica as ea
import elastica_jax as eaj
from elastica_jax.memory_block.protocol import JAXBlockState, RodView

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def rod() -> ea.CosseratRod:
    """Return a small rod for numerical damping comparisons."""
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
    "parameters",
    [
        {"damping_constant": 0.2},
        {"uniform_damping_constant": 0.3},
        {"translational_damping_constant": 0.4, "rotational_damping_constant": 0.5},
    ],
)
def test_combined_matches_separate_operators_under_jit(
    rod: ea.CosseratRod, parameters: dict[str, float]
) -> None:
    """Compare gravity and both velocity fields at nonuniform dilatation."""
    simulator = eaj.Simulator()
    block = eaj.configure_rod_block()
    simulator.enable_block_supports(ea.CosseratRod, block)
    simulator.append(rod)
    simulator.finalize()
    state = block.jax_get_state()
    for field in ("velocity_collection", "omega_collection", "external_forces"):
        state[field] = jax.numpy.ones_like(state[field])
    state["dilatation"] = jax.numpy.linspace(0.8, 1.2, rod.n_elems)
    gravity = eaj.GravityForcesJax()
    damper = eaj.AnalyticalLinearDamperJax(
        time_step=np.float64(1.0e-3), _system=rod, **parameters
    )
    combined = eaj.GravityAnalyticalDamperJax(
        time_step=1.0e-3, acc_gravity=None, _system=rod, **parameters
    )

    def apply_separately(state: JAXBlockState) -> JAXBlockState:
        view: RodView = block.rod_view(state, 0)
        view = gravity.jax_operate_synchronize(view, np.float64(0.0))
        return damper.jax_operate_constrain_rates(view, np.float64(0.0)).commit()

    def apply_combined(state: JAXBlockState) -> JAXBlockState:
        view: RodView = block.rod_view(state, 0)
        view = combined.jax_operate_synchronize(view, np.float64(0.0))
        return combined.jax_operate_constrain_rates(view, np.float64(0.0)).commit()

    expected = jax.jit(apply_separately)(state)
    actual = jax.jit(apply_combined)(state)

    for field in ("external_forces", "velocity_collection", "omega_collection"):
        np.testing.assert_allclose(actual[field], expected[field], rtol=1.0e-14)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"damping_constant": 0.1, "uniform_damping_constant": 0.2},
        {"translational_damping_constant": 0.1},
        {"rotational_damping_constant": 0.1},
    ],
)
def test_combined_rejects_invalid_damping_parameters(
    rod: ea.CosseratRod, parameters: dict[str, float]
) -> None:
    """Keep the standalone damper's parameter validation on the combined API."""
    with pytest.raises(ValueError):
        eaj.GravityAnalyticalDamperJax(
            time_step=1.0e-3, acc_gravity=None, _system=rod, **parameters
        )
