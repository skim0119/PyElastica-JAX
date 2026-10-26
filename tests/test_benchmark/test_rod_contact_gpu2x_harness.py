"""Tests for single-node two-GPU (gpu2x) rod-contact scaling helpers."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "rod-contact-scaling"
if str(BENCH_DIR) not in sys.path:
    sys.path.append(str(BENCH_DIR))

from _rod_contact_scaling_sweep import (  # noqa: E402
    ScalingCase,
    export_scaling_csv,
    series_label,
    sweep_backend,
)
from jax_rod_contact_throughput import ThroughputConfig  # noqa: E402
from sweep_jax_rod_contact_gpu2x_throughput import main as gpu2x_main  # noqa: E402


def test_series_label_encodes_multidevice_layouts() -> None:
    assert series_label("cuda", vertical=True, n_devices=2) == "jax-cuda-vertical-2x"
    assert series_label("cuda", vertical=False, n_devices=2) == "jax-cuda-horizontal-2x"
    assert series_label("cuda", vertical=True, n_devices=1) == "jax-cuda-vertical"


def test_export_scaling_csv_includes_n_devices(tmp_path: Path) -> None:
    cases = [
        ScalingCase(
            backend="cuda",
            vertical=True,
            n_devices=2,
            points=[(2, 4, 0.1, 0.2)],
        ),
    ]
    output = tmp_path / "gpu2x.csv"
    export_scaling_csv(
        cases,
        steps=200,
        n_elements=10,
        steps_between_detection=0,
        broad_phase="spatial_hash",
        output=output,
    )
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["backend"] == "cuda"
    assert rows[0]["vertical"] == "1"
    assert rows[0]["n_devices"] == "2"
    assert rows[0]["n_rods"] == "4"


def test_sweep_backend_forwards_n_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _fake_run(*, backend: str, config: ThroughputConfig) -> tuple[float, float]:
        seen["backend"] = backend
        seen["vertical"] = config.vertical
        seen["n_devices"] = config.n_devices
        seen["n_rods"] = config.n_rods
        return 0.01, 0.02

    monkeypatch.setattr("_rod_contact_scaling_sweep.run_throughput", _fake_run)
    points = sweep_backend(
        backend="cuda",
        min_exp=1,
        max_exp=1,
        steps=5,
        warmup_runs=0,
        n_elements=4,
        steps_between_detection=0,
        broad_phase="all_pairs",
        vertical=True,
        n_devices=2,
        verbose=False,
    )
    assert points == [(1, 2, 0.01, 0.02)]
    assert seen["backend"] == "cuda"
    assert seen["vertical"] is True
    assert seen["n_devices"] == 2
    assert seen["n_rods"] == 2


def test_gpu2x_cli_rejects_min_exp_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sweep_jax_rod_contact_gpu2x_throughput._backend_available",
        lambda _name: True,
    )
    with pytest.raises(AssertionError, match="min_exp"):
        CliRunner().invoke(
            gpu2x_main,
            [
                "--backend",
                "cpu",
                "--layout",
                "vertical",
                "--min-exp",
                "0",
                "--max-exp",
                "0",
                "--quiet",
            ],
            catch_exceptions=False,
        )


def test_gpu2x_cli_forwards_n_devices_to_scaling_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def _fake_run_scaling_benchmark(**kwargs: object) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(
        "sweep_jax_rod_contact_gpu2x_throughput._backend_available",
        lambda _name: True,
    )
    monkeypatch.setattr(
        "sweep_jax_rod_contact_gpu2x_throughput.run_scaling_benchmark",
        _fake_run_scaling_benchmark,
    )
    plot = tmp_path / "gpu2x.png"
    result = CliRunner().invoke(
        gpu2x_main,
        [
            "--backend",
            "cpu",
            "--layout",
            "vertical",
            "--min-exp",
            "1",
            "--max-exp",
            "1",
            "--steps",
            "5",
            "--warmup-runs",
            "0",
            "--output",
            str(plot),
            "--quiet",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert seen["backend"] == "cpu"
    assert seen["vertical"] is True
    assert seen["n_devices"] == 2
    assert seen["output_plot"] == plot


def test_horizontal_sweep_reuses_mpi_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split rods across two ranks and report the slowest rank's rollout."""
    from sweep_jax_rod_contact_gpu2x_throughput import sweep_horizontal_mpi
    from sweep_jax_rod_contact_mpi_throughput import RolloutPoint

    seen: dict[str, object] = {}

    def fake_run_mpi_point(**kwargs: object) -> RolloutPoint:
        seen.update(kwargs)
        return 2, 8, 16, np.asarray([0.3, 0.2])

    monkeypatch.setattr(
        "sweep_jax_rod_contact_gpu2x_throughput.run_mpi_point", fake_run_mpi_point
    )
    points = sweep_horizontal_mpi(
        "cuda",
        4,
        4,
        python_executable="/usr/bin/python",
        steps=10,
        warmup_runs=0,
        n_elements=10,
        steps_between_detection=0,
        broad_phase="spatial_hash",
        verbose=False,
    )

    assert seen == {
        "mpi_size": 2,
        "rods_per_rank_exp": 3,
        "rods_per_rank_multiplier": 1,
        "vertical": False,
        "python_executable": "/usr/bin/python",
        "steps": 10,
        "warmup_runs": 0,
        "backend": "cuda",
        "n_elements": 10,
        "steps_between_detection": 0,
        "broad_phase": "spatial_hash",
    }
    assert points == [(4, 16, 0.0, 0.3)]


@pytest.mark.parametrize("layout", ["vertical", "horizontal", "both"])
@pytest.mark.parametrize("explicit_csv", [False, True])
def test_gpu2x_output_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, layout: str, explicit_csv: bool
) -> None:
    import sweep_jax_rod_contact_gpu2x_throughput as worker

    outputs: list[tuple[Path, Path | None]] = []

    def record_output(**kwargs: object) -> None:
        outputs.append((kwargs["output_plot"], kwargs["output_csv"]))

    monkeypatch.setattr(worker, "_backend_available", lambda _name: True)
    monkeypatch.setattr(worker, "run_scaling_benchmark", record_output)
    monkeypatch.setattr(worker, "sweep_horizontal_mpi", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        worker,
        "_export_horizontal_case",
        lambda points, **kwargs: record_output(**kwargs),
    )
    args = ["--layout", layout, "--output", str(tmp_path / "timing.png"), "--quiet"]
    if explicit_csv:
        args.extend(["--csv-output", str(tmp_path / "samples.csv")])
    result = CliRunner().invoke(worker.main, args, catch_exceptions=False)
    assert result.exit_code == 0
    suffixes = ["_vertical", "_horizontal"] if layout == "both" else [""]
    assert outputs == [
        (
            tmp_path / f"timing{suffix}.png",
            tmp_path / f"samples{suffix}.csv" if explicit_csv else None,
        )
        for suffix in suffixes
    ]
