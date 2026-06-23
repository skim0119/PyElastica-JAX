"""Device selection and frame-capture integration loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
import jax
import numpy as np
from tqdm import tqdm
import elastica_jax as eaj

from frame_io import (
    extract_stacked_positions,
    save_frame_positions,
    save_simulation_metadata,
)

DEFAULT_FPS = 25.0
DEFAULT_SAVE_WORKERS = 4

F = TypeVar("F", bound=Callable[..., Any])


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


def override_parameters(parameters: Any, /, **replacements: Any) -> Any:
    updates = {key: value for key, value in replacements.items() if value is not None}
    if not updates:
        return parameters
    return type(parameters)(**{**parameters.__dict__, **updates})


def run_output_options(func: F) -> F:
    func = click.option(
        "--run-name",
        default=None,
        help='Output directory suffix: writes to "output" or "output_<run-name>".',
    )(func)
    func = click.option(
        "--fps",
        default=DEFAULT_FPS,
        show_default=True,
        help="Frame capture rate for HDF5 output.",
    )(func)
    func = click.option(
        "--save-workers",
        default=DEFAULT_SAVE_WORKERS,
        show_default=True,
        help="Worker processes for parallel HDF5 rod-chunk writes.",
    )(func)
    return func


def simulation_run_options(*, backends: tuple[str, ...]) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        func = click.option(
            "--backend",
            type=click.Choice(backends, case_sensitive=False),
            default="auto",
            show_default=True,
        )(func)
        func = click.option("--n-elements", type=int, default=None)(func)
        func = click.option("--n-snakes", type=int, default=None)(func)
        func = click.option("--final-time", type=float, default=None)(func)
        func = click.option("--time-step", type=float, default=None)(func)
        func = click.option("--seed", default=2026, show_default=True)(func)
        func = run_output_options(func)
        return func

    return decorator


def render_output_options(func: F) -> F:
    func = click.option(
        "--run-name",
        default=None,
        help='Read from "output" or "output_<run-name>" (must match the simulation run).',
    )(func)
    func = click.option("--dpi", default=150, show_default=True)(func)
    func = click.option(
        "--skip-ffmpeg",
        is_flag=True,
        help="Only render PNG frames and skip ffmpeg assembly.",
    )(func)
    return func


def available_platforms() -> dict[str, jax.Device]:
    result = {}
    for name in ("cpu", "gpu", "cuda", "metal", "mps"):
        try:
            devices = jax.devices(name)
        except Exception:
            continue
        if devices:
            result.setdefault(name, devices[0])
            result.setdefault(devices[0].platform.lower(), devices[0])
    if "gpu" in result:
        result.setdefault("cuda", result["gpu"])
    if "metal" in result:
        result.setdefault("mps", result["metal"])
    return result


def select_device(backend: str) -> tuple[str, jax.Device]:
    platforms = available_platforms()
    if backend == "auto":
        for candidate in ("cuda", "mps", "gpu", "cpu"):
            if candidate in platforms:
                return candidate, platforms[candidate]
        raise RuntimeError("No JAX devices are available.")
    assert (
        backend in platforms
    ), f"Requested backend {backend!r} is unavailable; found {sorted(platforms)}."
    return backend, platforms[backend]


def frame_schedule(
    *,
    final_time: float,
    time_step: float,
    fps: float,
) -> tuple[int, int, float, int]:
    assert fps > 0.0, "FPS must be positive."
    frame_dt = 1.0 / fps
    steps_per_frame = int(round(frame_dt / time_step))
    assert steps_per_frame >= 1, "FPS is too high for the chosen time step."
    frame_dt = steps_per_frame * time_step
    n_frames = int(round(final_time / frame_dt))
    assert n_frames >= 1, "Simulation must produce at least one frame interval."
    assert np.isclose(
        n_frames * frame_dt, final_time
    ), "Final time must be an integer multiple of the frame interval."
    return n_frames, steps_per_frame, frame_dt, n_frames + 1


def integrate_with_frame_capture(
    *,
    simulator,
    block,
    parameters: Any,
    case_name: str,
    n_snakes: int,
    radius: float,
    final_time: float,
    time_step: float,
    output_dir: Path,
    fps: float,
    save_workers: int,
    wall_origins: np.ndarray,
    wall_normals: np.ndarray,
    seed: int,
    run_name: str | None = None,
) -> float:
    steps = int(round(final_time / time_step))
    assert np.isclose(
        steps * time_step, final_time
    ), "Final time must be an integer multiple of the time step."

    n_frames, steps_per_frame, frame_dt, n_frame_files = frame_schedule(
        final_time=final_time,
        time_step=time_step,
        fps=fps,
    )
    save_simulation_metadata(
        output_dir / "metadata.h5",
        case_name=case_name,
        parameters=parameters,
        fps=fps,
        steps_per_frame=steps_per_frame,
        frame_dt=frame_dt,
        wall_origins=wall_origins,
        wall_normals=wall_normals,
        seed=seed,
        run_name=run_name,
    )

    stepper = eaj.PositionVerletGPU()
    start = time.perf_counter()
    current_time = 0.0
    for frame_idx in tqdm(range(n_frame_files)):
        jax.block_until_ready(block.position_collection_device)
        stacked_positions = extract_stacked_positions(block, n_snakes=n_snakes)
        save_frame_positions(
            output_dir / f"frame_{frame_idx:06d}.h5",
            stacked_positions,
            time=current_time,
            frame_idx=frame_idx,
            attrs={"case": case_name, "radius": radius},
            n_workers=save_workers,
        )
        if frame_idx == n_frames:
            break
        chunk_final_time = current_time + frame_dt
        stepper.integrate(
            simulator,
            time=current_time,
            final_time=chunk_final_time,
            dt=time_step,
        )
        current_time = chunk_final_time

    elapsed = time.perf_counter() - start
    positions = np.asarray(block.position_collection_device)
    print(f"case={case_name} steps={steps}")
    print(
        f"frames={n_frame_files} fps={fps} steps_per_frame={steps_per_frame} frame_dt={frame_dt:.6e}"
    )
    print(f"saved_frames={output_dir}")
    print(f"elapsed={elapsed:.6f}s finite={np.isfinite(positions).all()}")
    return elapsed
