"""Tests for single-node two-GPU (gpu2x) rod-contact scaling helpers."""

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
from jax_rod_contact_throughput import ThroughputConfig  # noqa: E402


def test_series_label_encodes_multidevice_vertical() -> None:
    assert series_label("cuda", vertical=True, n_devices=2) == "jax-cuda-vertical-2x"
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
