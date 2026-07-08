"""MPI-parallel renderer for active-matter snake-pit HDF5 frame dumps.

Reads chunked ``frame_*.h5`` dumps and ``metadata.h5`` from ``data/`` (or
``data_<run-name>/``), renders top/side planar views per frame to
``render/png/``, and assembles ``render/output.mp4`` with ffmpeg. Rendering is
split across MPI ranks; run standalone or under ``mpiexec``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import click

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from frame_io import (  # noqa: E402
    list_frame_paths,
    load_frame_positions,
    load_simulation_metadata,
)

from mpi4py import MPI  # noqa: E402


def render_all(
    *,
    run_name: str | None,
    dpi: int = 150,
    skip_ffmpeg: bool = False,
) -> None:
    """Render every frame to PNGs and assemble an mp4 across MPI ranks.

    Parameters
    ----------
    run_name : str | None
        Selects the ``data``/``render`` pair (``data_<run-name>`` when set).
    dpi : int, optional
        Figure resolution for the PNG frames.
    skip_ffmpeg : bool, optional
        Render PNG frames only and skip the mp4 assembly.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    input_dir = _EXAMPLE_DIR / ("data" if run_name is None else f"data_{run_name}")
    render_root = _EXAMPLE_DIR / (
        "render" if run_name is None else f"render_{run_name}"
    )
    frame_png_dir = render_root / "png"
    metadata_path = input_dir / "metadata.h5"
    assert metadata_path.is_file(), f"Missing simulation metadata at {metadata_path}."

    metadata = load_simulation_metadata(metadata_path)
    fps = float(metadata["fps"])
    frame_paths = list_frame_paths(input_dir)
    assert frame_paths, f"No frame_*.h5 files found in {input_dir}."

    local_frames = [
        path for index, path in enumerate(frame_paths) if index % size == rank
    ]
    total_frames = len(frame_paths)
    if rank == 0:
        print(f"rendering {total_frames} frames from {input_dir} at {fps:g} fps")
        print(f"MPI ranks={size}")

    for run_index, frame_path in enumerate(local_frames):
        frame_idx = int(frame_path.stem.split("_")[-1])
        out_png = frame_png_dir / f"frame_{frame_idx:06d}.png"
        render_frame_png(frame_path, out_png, metadata=metadata, dpi=dpi)
        # if total_frames > 0 and run_index % max(1, total_frames // 50) == 0:
        #     percent = 100.0 * (run_index + 1) / total_frames
        #     print(f"rank={rank} wrote {out_png} ({percent:.1f}%)", flush=True)

    comm.Barrier()

    if rank == 0 and not skip_ffmpeg:
        out_video = render_root / "output.mp4"
        _run_ffmpeg(png_root=frame_png_dir, output_video=out_video, fps=fps)
        print(f"wrote {out_video}")


def render_frame_png(
    frame_path: Path,
    out_png: Path,
    *,
    metadata: dict[str, object],
    dpi: int,
) -> None:
    """Render one frame as side-by-side top and side planar views."""
    positions, frame_attrs = load_frame_positions(frame_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    top_view, side_view = _view_definitions()
    wall_origins = np.asarray(metadata["wall_origins"])
    wall_normals = np.asarray(metadata["wall_normals"])

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    _configure_planar_axes(
        ax_top,
        positions,
        view=top_view,
        wall_origins=wall_origins,
        wall_normals=wall_normals,
    )
    _configure_planar_axes(
        ax_side,
        positions,
        view=side_view,
        wall_origins=wall_origins,
        wall_normals=wall_normals,
    )

    time_value = float(frame_attrs.get("time", 0.0))
    fig.suptitle(f"t = {time_value:.4f} s")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _view_definitions() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {"axis_a": 0, "axis_b": 1, "label_a": "x", "label_b": "y", "name": "Top view"},
        {"axis_a": 0, "axis_b": 2, "label_a": "x", "label_b": "z", "name": "Side view"},
    )


def _third_axis(axis_a: int, axis_b: int) -> int:
    return ({0, 1, 2} - {axis_a, axis_b}).pop()


def _planar_limits(
    positions: np.ndarray,
    axis_a: int,
    axis_b: int,
    *,
    padding_ratio: float = 0.1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    flat = positions.reshape(-1, 3)
    a_values = flat[:, axis_a]
    b_values = flat[:, axis_b]
    a_center = 0.5 * (a_values.min() + a_values.max())
    b_center = 0.5 * (b_values.min() + b_values.max())
    half_span = 0.5 * max(
        float(a_values.max() - a_values.min()),
        float(b_values.max() - b_values.min()),
    )
    half_span = max(half_span * (1.0 + padding_ratio), 5.0e-7)
    return (
        (a_center - half_span, a_center + half_span),
        (b_center - half_span, b_center + half_span),
    )


def _draw_walls_2d(
    ax,
    wall_origins: np.ndarray,
    wall_normals: np.ndarray,
    *,
    axis_a: int,
    axis_b: int,
    a_lim: tuple[float, float],
    b_lim: tuple[float, float],
) -> None:
    axis_k = _third_axis(axis_a, axis_b)
    for origin, normal in zip(wall_origins, wall_normals, strict=True):
        normal = np.asarray(normal, dtype=float)
        origin = np.asarray(origin, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1.0e-12:
            continue
        normal = normal / normal_norm
        na = float(normal[axis_a])
        nb = float(normal[axis_b])
        nk = float(normal[axis_k])
        if abs(na) < 1.0e-12 and abs(nb) < 1.0e-12:
            continue
        plane_offset = float(np.dot(normal, origin))
        a_values = np.array(a_lim, dtype=float)
        if abs(nb) > abs(na):
            b_values = (plane_offset - na * a_values) / nb
        else:
            b_values = np.array(b_lim, dtype=float)
            a_values = (plane_offset - nb * b_values) / na
        ax.plot(a_values, b_values, color="0.5", linewidth=1.0, alpha=0.8, zorder=1)
        if abs(nk) < 1.0e-8:
            tangent = np.zeros(3, dtype=float)
            tangent[axis_a] = -nb
            tangent[axis_b] = na
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1.0e-12:
                tangent = tangent / tangent_norm
                span = max(a_lim[1] - a_lim[0], b_lim[1] - b_lim[0])
                hatch_origin = origin + 0.02 * span * normal
                hatch_end = hatch_origin + 0.12 * span * tangent
                ax.plot(
                    [hatch_origin[axis_a], hatch_end[axis_a]],
                    [hatch_origin[axis_b], hatch_end[axis_b]],
                    color="0.45",
                    linewidth=0.8,
                    alpha=0.6,
                    zorder=1,
                )


def _configure_planar_axes(
    ax,
    positions: np.ndarray,
    *,
    view: dict[str, object],
    wall_origins: np.ndarray,
    wall_normals: np.ndarray,
) -> None:
    axis_a = int(view["axis_a"])
    axis_b = int(view["axis_b"])
    a_lim, b_lim = _planar_limits(positions, axis_a, axis_b)
    ax.set_xlim(*a_lim)
    ax.set_ylim(*b_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(str(view["label_a"]))
    ax.set_ylabel(str(view["label_b"]))
    ax.set_title(str(view["name"]))
    ax.grid(True, alpha=0.25)
    if wall_origins.size > 0:
        _draw_walls_2d(
            ax,
            wall_origins,
            wall_normals,
            axis_a=axis_a,
            axis_b=axis_b,
            a_lim=a_lim,
            b_lim=b_lim,
        )
    for rod_positions in positions:
        ax.plot(
            rod_positions[:, axis_a],
            rod_positions[:, axis_b],
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
        )


def _run_ffmpeg(*, png_root: Path, output_video: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None, "ffmpeg is required to assemble the mp4 video."
    output_video.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(png_root / "frame_%06d.png")
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    subprocess.run(command, check=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--run-name",
    default=None,
    help='Read from "data" or "data_<run-name>" (must match the simulation run).',
)
@click.option("--dpi", default=150, show_default=True)
@click.option(
    "--skip-ffmpeg",
    is_flag=True,
    help="Only render PNG frames and skip ffmpeg assembly.",
)
def main(run_name: str | None, dpi: int, skip_ffmpeg: bool) -> None:
    """Render snake-pit frames from ``data/`` into ``render/``."""
    render_all(run_name=run_name, dpi=dpi, skip_ffmpeg=skip_ffmpeg)


if __name__ == "__main__":
    main()
