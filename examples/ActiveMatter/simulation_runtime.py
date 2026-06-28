"""Shared helpers for active-matter examples."""

from __future__ import annotations

from pathlib import Path


def resolve_output_dir(run_name: str | None) -> Path:
    if run_name is None:
        return Path("output")
    return Path(f"output_{run_name}")


def png_dir_for(output_dir: Path) -> Path:
    return output_dir / "png"


def video_path_for(run_name: str | None) -> Path:
    if run_name is None:
        return Path("output.mp4")
    return Path(f"output_{run_name}.mp4")
