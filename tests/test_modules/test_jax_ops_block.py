import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from elastica.rod.cosserat_rod import CosseratRod  # noqa: E402
from elastica_jax import (  # noqa: E402
    NoBlockOpJax,
    OneEndFixedJax,
    RodSystemLike,
    Simulator,
    configure_rod_block,
)
from elastica_jax.memory_block.protocol import JAXBlockState, RodView  # noqa: E402


class _ScaleExternalForces(NoBlockOpJax):
    def __init__(self, scale: float, **kwargs: object) -> None:
        del kwargs
        self.scale = np.float64(scale)

    def jax_block_operate_synchronize(
        self,
        state: JAXBlockState,
        time: np.float64,
    ) -> JAXBlockState:
        del time
        updated = dict(state)
        updated["external_forces"] = state["external_forces"] * self.scale
        return updated


class _ScaleExternalForcesPerRod(NoBlockOpJax):
    def __init__(self, scale: float, **kwargs: object) -> None:
        del kwargs
        self.scale = np.float64(scale)

    def jax_per_rod_operate_synchronize(
        self,
        rod_view: RodView,
        time: np.float64,
    ) -> RodView:
        del time
        rod_view.external_forces = rod_view.external_forces * self.scale
        return rod_view


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


def test_operate_block_accepts_configured_block_instance() -> None:
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
        forces = initial_state["external_forces"]
        initial_state["external_forces"] = jax.numpy.arange(
            1, forces.size + 1, dtype=forces.dtype
        ).reshape(forces.shape)
        updated_state = simulator.jax_synchronize((initial_state,), np.float64(0.0))[0]
        np.testing.assert_allclose(
            np.asarray(updated_state["external_forces"]),
            3.0 * np.asarray(initial_state["external_forces"]),
        )


def test_operate_block_per_rod_uses_map_rods() -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        simulator = Simulator()
        rod_block = configure_rod_block()
        simulator.enable_block_supports(CosseratRod, rod_block)
        simulator.append(_build_rod())
        simulator.append(_build_rod(n_elems=8))
        simulator.operate_block(rod_block).using(_ScaleExternalForcesPerRod, scale=4.0)
        simulator.finalize()

        block = tuple(simulator.final_systems())[0]
        initial_state = block.jax_get_state()
        forces = initial_state["external_forces"]
        initial_state["external_forces"] = jax.numpy.arange(
            1, forces.size + 1, dtype=forces.dtype
        ).reshape(forces.shape)
        updated_state = simulator.jax_synchronize((initial_state,), np.float64(0.0))[0]
        for rod_idx in range(2):
            start = int(block.start_idx_in_rod_nodes[rod_idx])
            end = int(block.end_idx_in_rod_nodes[rod_idx])
            np.testing.assert_allclose(
                np.asarray(updated_state["external_forces"])[:, start:end],
                4.0 * np.asarray(initial_state["external_forces"])[:, start:end],
            )


@pytest.mark.parametrize("operator_cls", [_ScaleExternalForcesPerRod, OneEndFixedJax])
def test_per_rod_block_operators_reject_incompatible_rods_at_finalize(
    operator_cls: type[NoBlockOpJax] | type[OneEndFixedJax],
) -> None:
    """Reject unsupported batching before the first integration call."""
    simulator = Simulator()
    block = configure_rod_block()
    simulator.enable_block_supports(CosseratRod, block)
    simulator.append(_build_rod())
    second = _build_rod(n_elems=6)
    simulator.append(second)
    kwargs = {"scale": 2.0} if operator_cls is _ScaleExternalForcesPerRod else {}
    simulator.operate_block(block).using(operator_cls, **kwargs)

    with pytest.raises(AssertionError, match=r"simulator\.operate\(rod\)"):
        simulator.finalize()


@pytest.mark.parametrize("per_rod_registration", [False, True])
def test_unequal_rods_support_unbatched_operations(
    per_rod_registration: bool,
) -> None:
    """Keep native Block hooks and separate rod registrations available."""
    simulator = Simulator()
    block = configure_rod_block()
    simulator.enable_block_supports(CosseratRod, block)
    rods = [_build_rod(), _build_rod(n_elems=6)]
    for rod in rods:
        simulator.append(rod)
        if per_rod_registration:
            simulator.operate(rod).using(OneEndFixedJax)
    if not per_rod_registration:
        simulator.operate_block(block).using(_ScaleExternalForces, scale=2.0)
    simulator.finalize()

    state = block.jax_get_state()
    forces = jax.numpy.ones_like(state["external_forces"])
    state["external_forces"] = forces
    result = jax.jit(simulator.jax_synchronize)((state,), 0.0)[0]

    expected_scale = 1.0 if per_rod_registration else 2.0
    np.testing.assert_allclose(result["external_forces"], expected_scale * forces)


def test_reused_operator_is_constructed_once_per_rod() -> None:
    """Inspect existing instances without constructing the first rod twice."""
    constructed: list[RodSystemLike] = []

    class CountingBoundary(OneEndFixedJax):
        """Record constructor calls while keeping the fixed-end physics."""

        def __init__(self, *, _system: RodSystemLike) -> None:
            """Capture the target rod and its initial boundary."""
            constructed.append(_system)
            super().__init__(_system=_system)

    simulator = Simulator()
    block = configure_rod_block()
    simulator.enable_block_supports(CosseratRod, block)
    rods = [_build_rod(), _build_rod()]
    for index, rod in enumerate(rods):
        rod.position_collection[0] += float(index)
        simulator.append(rod)
    simulator.operate_block(block).using(CountingBoundary)
    simulator.finalize()

    assert len(constructed) == len(rods), "Construct exactly one operator per rod."
    assert all(target is rod for target, rod in zip(constructed, rods))
    state = block.jax_get_state()
    state["position_collection"] += 2.0
    result = jax.jit(simulator.jax_constrain_values)((state,), 0.0)[0]
    for index, start in enumerate(block.start_idx_in_rod_nodes):
        np.testing.assert_allclose(
            result["position_collection"][:, start], np.array([float(index), 0.0, 0.0])
        )
