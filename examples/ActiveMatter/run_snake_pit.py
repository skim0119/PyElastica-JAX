"""GPU JAX reproduction of the enclosed snake-pit active-matter case.

Randomly packed active snakes are enclosed by a floor and four side walls, driven
by a traveling-wave internal torque, and resolved with capsule-capsule rod-rod
contact and capsule-half-space wall contact. Node positions are streamed to
chunked HDF5 frames under ``data/`` for offline rendering (see
``post_processing.py``). Reference physics live in ``CASE_DESCRIPTION.md``.

Configuration lives in :class:`SnakePitParameters` below; ``--smoke`` downscales
the run while still exercising every operator.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from environment import build_simulation  # noqa: E402
from frame_io import (  # noqa: E402
    extract_stacked_positions,
    save_frame_positions,
    save_simulation_metadata,
)

import elastica_jax as eaj  # noqa: E402
import jax  # noqa: E402
from jax import config as jax_config  # noqa: E402

jax_config.update("jax_enable_x64", True)


@dataclass(frozen=True, kw_only=True)
class SnakePitParameters:
    """Physical and numerical parameters for the snake-pit case.

    ``n_snakes`` and ``final_time`` are supplied by the CLI (or :meth:`smoke`);
    the remaining fields default to the reference case and can be edited here to
    change physics or scale.
    """

    n_snakes: int
    final_time: float

    n_elements: int = 20
    length: float = 0.35
    radius_ratio: float = 0.011
    density: float = 1000.0
    youngs_modulus: float = 1.0e5  # * 10
    time_period: float = 2.0
    wave_length: float = 1.0
    contact_stiffness: float = 1.0e3  # * 10
    contact_damping: float = 1.0e-3
    gravitational_acc: float = -0.1
    damping_rate: float = 1.0e-4
    time_step: float = 2.0e-4  # 5.0e-5
    activation_start_time_nd: float = 5.0
    packing_initial_vertical_span_ratio: float = 1.0
    packing_initial_radial_span_ratio: float = 1.0
    wall_distance_ratio: float = 2.0
    steps_between_detection: int = 0

    @property
    def radius(self) -> float:
        return self.radius_ratio * self.length

    def pit_walls(self) -> tuple[np.ndarray, np.ndarray]:
        """Return floor and side-wall ``(origins, inward_normals)`` half-spaces."""
        half = 0.5 * self.wall_distance_ratio * self.length
        origins = np.array(
            [
                [0.0, 0.0, 0.0],
                [-half, 0.0, 0.0],
                [half, 0.0, 0.0],
                [0.0, -half, 0.0],
                [0.0, half, 0.0],
            ]
        )
        normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        return origins, normals

    @classmethod
    def smoke(cls) -> SnakePitParameters:
        """Return a downscaled copy for a fast smoke test of the full stack."""
        return cls(n_snakes=4, final_time=0.5)


def run_simulation(
    parameters: SnakePitParameters,
    *,
    backend: str,
    mesh: str,
    seed: int,
    run_name: str | None,
    fps: float,
    save_workers: int,
    block_checkpoint: Path,
) -> Path:
    """Build the simulator, roll it out, and stream HDF5 frames to ``data/``."""
    output_dir = _EXAMPLE_DIR / ("data" if run_name is None else f"data_{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    devices = eaj.execution_mesh_for_block_checkpoint(
        block_checkpoint,
        mesh_name=mesh,
        backend=backend,
        n_rods=parameters.n_snakes,
    )
    simulator, block = build_simulation(
        parameters,
        devices=devices,
        seed=seed,
        block_checkpoint=block_checkpoint,
    )
    wall_origins, wall_normals = parameters.pit_walls()

    frame_dt = 1.0 / fps
    steps_per_frame = int(round(frame_dt / parameters.time_step))
    frame_dt = steps_per_frame * parameters.time_step
    n_frames = int(round(parameters.final_time / frame_dt)) + 1
    save_simulation_metadata(
        output_dir / "metadata.h5",
        case_name="snake-pit",
        parameters=parameters,
        fps=fps,
        steps_per_frame=steps_per_frame,
        frame_dt=frame_dt,
        wall_origins=wall_origins,
        wall_normals=wall_normals,
        seed=seed,
        run_name=run_name,
    )

    stepper = eaj.PositionVerletJAX()
    start = time.perf_counter()
    current_time = 0.0
    for frame_idx in tqdm(range(n_frames), desc="Snake-pit rollout"):
        jax.block_until_ready(block.position_collection_device)
        stacked_positions = extract_stacked_positions(
            block, n_snakes=parameters.n_snakes
        )
        save_frame_positions(
            output_dir / f"frame_{frame_idx:06d}.h5",
            stacked_positions,
            time=current_time,
            frame_idx=frame_idx,
            attrs={"case": "snake-pit", "radius": parameters.radius},
            n_workers=save_workers,
        )
        if frame_idx == n_frames - 1:
            break
        chunk_final_time = current_time + frame_dt
        stepper.integrate(
            simulator,
            time=current_time,
            final_time=chunk_final_time,
            dt=parameters.time_step,
        )
        current_time = chunk_final_time

    elapsed = time.perf_counter() - start
    positions = np.asarray(block.position_collection_device)
    print(
        f"case=snake-pit steps={int(round(parameters.final_time / parameters.time_step))}"
    )
    print(
        f"frames={n_frames} fps={fps} steps_per_frame={steps_per_frame} "
        f"frame_dt={frame_dt:.6e}"
    )
    print(f"saved_frames={output_dir}")
    print(f"elapsed={elapsed:.6f}s finite={np.isfinite(positions).all()}")
    print(
        f"backend={backend} mesh={mesh} shards={getattr(block, 'n_shards', 1)} "
        f"dtype={block.device_dtype}"
    )
    return output_dir


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-N", "--n-snakes", type=int, default=4, show_default=True)
@click.option("-T", "--final-time", type=float, default=20.0, show_default=True)
@click.option("--gpu", is_flag=True, help="Run on the CUDA backend (default CPU).")
@click.option(
    "--smoke", is_flag=True, help="Downscaled fast run exercising all features."
)
@click.option("--render", is_flag=True, help="Render frames and mp4 after the run.")
@click.option(
    "--mesh",
    type=click.Choice(("unified", "auto"), case_sensitive=False),
    default="auto",
    show_default=True,
    help=(
        "Execution mesh: unified keeps one shard; auto uses one shard per "
        "local device when multiple are available."
    ),
)
@click.option("--seed", default=2026, show_default=True)
@click.option(
    "--run-name",
    default=None,
    help='Output suffix: writes to "data" or "data_<run-name>".',
)
@click.option(
    "--fps", default=25, show_default=True, help="Frame capture rate for HDF5 output."
)
@click.option(
    "--save-workers",
    default=4,
    show_default=True,
    help="Worker processes for parallel HDF5 rod-chunk writes.",
)
@click.option(
    "--block-checkpoint",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help=(
        "HDF5 block state path: load when the file exists, otherwise pack rods "
        "and save after construction. Defaults to <data-dir>/block_checkpoint.h5."
    ),
)
def main(
    n_snakes: int,
    final_time: float,
    gpu: bool,
    smoke: bool,
    render: bool,
    mesh: str,
    seed: int,
    run_name: str | None,
    fps: float,
    save_workers: int,
    block_checkpoint: Path | None,
) -> None:
    """Build, integrate, and stream the JAX snake-pit case to ``data/``."""
    parameters = (
        SnakePitParameters.smoke()
        if smoke
        else SnakePitParameters(n_snakes=n_snakes, final_time=final_time)
    )
    backend = "cuda" if gpu else "cpu"
    if block_checkpoint is None:
        data_root = _EXAMPLE_DIR / ("data" if run_name is None else f"data_{run_name}")
        block_checkpoint = data_root / "block_checkpoint.h5"
    block_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    run_simulation(
        parameters,
        backend=backend,
        mesh=mesh,
        seed=seed,
        run_name=run_name,
        fps=fps,
        save_workers=save_workers,
        block_checkpoint=block_checkpoint,
    )

    if render:
        from post_processing import render_all

        render_all(run_name=run_name, dpi=150, skip_ffmpeg=False)


if __name__ == "__main__":
    main()
