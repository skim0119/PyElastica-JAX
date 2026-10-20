"""Execute the introductory README examples as users copy them."""

from pathlib import Path
import re
from unittest.mock import Mock

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
