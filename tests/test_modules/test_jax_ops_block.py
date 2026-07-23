import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from elastica.rod.cosserat_rod import CosseratRod  # noqa: E402
from elastica_jax import NoBlockOpJax, Simulator, configure_rod_block  # noqa: E402


class _ScaleExternalForces(NoBlockOpJax):
    def __init__(self, scale: float, *, _system) -> None:
        self.scale = np.float64(scale)

    def jax_block_operate_synchronize(self, state, time):
        del time
        updated = dict(state)
        updated["external_forces"] = state["external_forces"] * self.scale
        return updated


def _build_rod(n_elems: int = 8) -> CosseratRod:
    return CosseratRod.straight_rod(
        n_elements=n_elems,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.0,
        base_radius=0.05,
        density=1_000.0,
        youngs_modulus=1.0e6,
    )


def test_operate_block_accepts_configured_block_instance():
    with jax.default_device(jax.devices("cpu")[0]):
        simulator = Simulator()
        rod_block = configure_rod_block()
        simulator.enable_block_supports(CosseratRod, rod_block)
        simulator.append(_build_rod())
        simulator.operate_block(rod_block).using(_ScaleExternalForces, scale=3.0)
        simulator.finalize()

        block = tuple(simulator.final_systems())[0]
        assert block is rod_block

        initial_state = block.jax_get_state()
        updated_state = simulator.jax_synchronize((initial_state,), np.float64(0.0))[0]
        np.testing.assert_allclose(
            np.asarray(updated_state["external_forces"]),
            3.0 * np.asarray(initial_state["external_forces"]),
        )
