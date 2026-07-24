"""Tests for single-node rod-contact scaling harness helpers."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "rod-contact-scaling"
if str(BENCH_DIR) not in sys.path:
    sys.path.append(str(BENCH_DIR))

from _rod_contact_scaling_sweep import (  # noqa: E402
    ScalingCase,
    export_scaling_csv,
    series_label,
    sweep_backend,
)
from jax_rod_contact_throughput import (  # noqa: E402
    ThroughputConfig,
    run as run_throughput,
)


def test_series_label_encodes_backend_and_layout() -> None:
    assert series_label("pyelastica", vertical=False) == "pyelastica"
    assert series_label("cpu", vertical=False) == "jax-cpu-horizontal"
    assert series_label("cpu", vertical=True) == "jax-cpu-vertical"
    assert series_label("cuda", vertical=True) == "jax-cuda-vertical"


def test_export_scaling_csv_is_long_form_with_vertical(
    tmp_path: Path,
) -> None:
    cases = [
        ScalingCase(
            backend="cpu",
            vertical=False,
            points=[(1, 2, 0.1, 0.2)],
        ),
        ScalingCase(
            backend="cpu",
            vertical=True,
            points=[(1, 2, 0.15, 0.25)],
        ),
        ScalingCase(
            backend="pyelastica",
            vertical=False,
            points=[(1, 2, 0.3, 0.4)],
        ),
    ]
    output = tmp_path / "scaling.csv"
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
    assert [row["backend"] for row in rows] == ["cpu", "cpu", "pyelastica"]
    assert [row["vertical"] for row in rows] == ["0", "1", "0"]
    assert rows[1]["rollout_walltime_s"] == "0.25"
    assert rows[0]["steps"] == "200"
    assert rows[0]["n_elements"] == "10"
    assert rows[0]["broad_phase"] == "spatial_hash"


def test_sweep_backend_forwards_vertical_through_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _fake_run(*, backend: str, config: ThroughputConfig) -> tuple[float, float]:
        seen["backend"] = backend
        seen["vertical"] = config.vertical
        seen["n_rods"] = config.n_rods
        seen["broad_phase"] = config.broad_phase
        return 0.01, 0.02

    monkeypatch.setattr(
        "jax_rod_contact_throughput.run",
        _fake_run,
    )
    monkeypatch.setattr(
        "_rod_contact_scaling_sweep.run_throughput",
        _fake_run,
    )
    points = sweep_backend(
        backend="cpu",
        min_exp=1,
        max_exp=1,
        steps=5,
        warmup_runs=0,
        n_elements=4,
        steps_between_detection=0,
        broad_phase="all_pairs",
        vertical=True,
        verbose=False,
    )
    assert points == [(1, 2, 0.01, 0.02)]
    assert seen["backend"] == "cpu"
    assert seen["n_rods"] == 2
    assert seen["vertical"] is True
    assert seen["broad_phase"] == "all_pairs"


def test_throughput_worker_rejects_vertical_pyelastica() -> None:
    with pytest.raises(AssertionError, match="vertical"):
        run_throughput(
            backend="pyelastica",
            config=ThroughputConfig(
                n_rods_exp=1,
                steps=1,
                warmup_runs=0,
                vertical=True,
            ),
        )


def test_throughput_worker_rejects_multidevice_without_vertical() -> None:
    with pytest.raises(AssertionError, match="vertical"):
        run_throughput(
            backend="cpu",
            config=ThroughputConfig(
                n_rods_exp=1,
                steps=1,
                warmup_runs=0,
                vertical=False,
                n_devices=2,
            ),
        )
