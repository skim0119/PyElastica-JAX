"""Execute the introductory README examples as users copy them."""

from pathlib import Path
import re
from unittest.mock import Mock
from typing import cast

import jax

import numpy as np
import pytest

import elastica_jax as eaj


@pytest.mark.parametrize("block_operators", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_readme_tutorial_reaches_final_frame(
    block_operators: bool, stacked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run assembly, stepping, and sampling without writing HDF5 files."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    snippets = re.findall(r"```py\n(.*?)```", readme, flags=re.DOTALL)
    setup = next(code for code in snippets if code.startswith("import jax\n"))
    integration = next(code for code in snippets if code.startswith("timestepper ="))
    collection = next(code for code in snippets if code.startswith("# Continue"))
    block_example = next(
        code for code in snippets if code.startswith("class GravityBlockOp")
    )
    save = Mock()
    monkeypatch.setattr(eaj, "save", save)
    namespace: dict[str, object] = {}
    if stacked:
        setup = setup.replace(
            "eaj.configure_rod_block()",
            "eaj.configure_rod_block(inner_block_cls=eaj._CosseratRodVerticalMemoryBlock)",
        )
    if block_operators:
        setup = setup.replace("simulator.finalize()", "")
        setup = setup.replace(
            "    simulator.operate(rod).using(eaj.OneEndFixedJax)\n", ""
        ).replace("    simulator.operate(rod).using(eaj.GravityForcesJax)\n", "")
    exec(setup, namespace)
    if block_operators:
        exec(block_example + "\nsimulator.finalize()", namespace)
    exec(integration, namespace)
    # Exercise a final chunk shorter than steps_per_frame, too.
    exec(collection.replace("current_time + 0.01", "current_time + 0.011"), namespace)

    times = [call.kwargs["time"] for call in save.call_args_list]
    np.testing.assert_allclose(times, [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.021])
    assert namespace["current_time"] == pytest.approx(0.021)
    exec(
        "assert rods[0].position_collection[1, -1] < 0.0\n"
        "np.testing.assert_allclose(rods[0].position_collection[:, 0], 0.0)",
        namespace,
    )


def test_readme_custom_block_dissipates_axial_motion() -> None:
    """Check the documented viscous extension under JIT on two packed rods."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    snippets = re.findall(r"```py\n(.*?)```", readme, flags=re.DOTALL)
    setup = next(code for code in snippets if code.startswith("import jax\n"))
    custom_block = next(
        code for code in snippets if "class AxialViscousBlock" in code
    ).split("# Use this Block")[0]
    namespace: dict[str, object] = {"eaj": eaj, "jax": jax}
    exec(custom_block, namespace)
    setup = setup.replace(
        "eaj.configure_rod_block()",
        "eaj.configure_rod_block(inner_block_cls=AxialViscousBlock)",
    ).replace(
        "simulator.finalize()",
        "for rod in rods:\n"
        "    rod.velocity_collection[0] = np.linspace(0.0, 0.1, rod.n_nodes)\n"
        "simulator.finalize()",
    )
    exec(setup, namespace)
    block = cast(eaj._CosseratRodMemoryBlock, namespace["rod_block"])
    state = block.jax_get_state()

    elastic = eaj._CosseratRodMemoryBlock.jax_compute_internal_forces_and_torques(
        block, state, np.float64(0.0)
    )
    viscous = jax.jit(block.jax_compute_internal_forces_and_torques)(state, 0.0)
    added_forces = np.asarray(viscous["internal_forces"] - elastic["internal_forces"])

    for start, end in zip(block.start_idx_in_rod_nodes, block.end_idx_in_rod_nodes):
        forces = added_forces[:, start:end]
        velocities = np.asarray(state["velocity_collection"])[:, start:end]
        np.testing.assert_allclose(forces.sum(axis=-1), 0.0, atol=1.0e-12)
        assert np.sum(forces * velocities) < 0.0, (
            "Axial viscous loads must dissipate energy within each rod."
        )
    np.testing.assert_allclose(viscous["internal_torques"], elastic["internal_torques"])
