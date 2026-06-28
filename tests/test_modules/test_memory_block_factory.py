import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from elastica.modules import BaseSystemCollection  # noqa: E402
from elastica.rod.cosserat_rod import CosseratRod  # noqa: E402
from elastica_jax.memory_block.block_factory import (  # noqa: E402
    configure_rod_block,
)
from elastica_jax.memory_block.memory_block_rod_jax import (  # noqa: E402
    _CosseratRodMemoryBlock,
)


class _DummySimulator(BaseSystemCollection):
    pass


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


def test_configure_rod_block_builds_block_at_finalize():
    simulator = _DummySimulator()
    rod_block = configure_rod_block(device_dtype=np.float64)
    simulator.enable_block_supports(CosseratRod, rod_block)

    rods = [_build_rod(), _build_rod()]
    for rod in rods:
        simulator.append(rod)

    simulator.finalize()

    assert isinstance(rod_block, _CosseratRodMemoryBlock)
    assert rod_block.n_rods == 2
    assert rod_block is tuple(simulator.final_systems())[0]
