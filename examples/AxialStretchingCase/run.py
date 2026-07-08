"""Run the JAX axial stretching example and save diagnostics."""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from operations import (
    AxialStretchingParameters,
    build_simulator,
    extract_centerline,
    plot_centerline_evolution,
    plot_final_centerline,
    save_simulation_data,
)

import elastica_jax as eaj
import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(("cpu", "cuda"), case_sensitive=False),
    default="cpu",
    show_default=True,
)
@click.option("--n-elem", type=int, default=19, show_default=True)
@click.option("--final-time", type=float, default=0.2, show_default=True)
@click.option("--time-step", type=float, default=0.1 / 19.0, show_default=True)
@click.option("--fps", type=float, default=30.0, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default="output",
    show_default=True,
)
def main(
    backend: str,
    n_elem: int,
    final_time: float,
    time_step: float,
    fps: float,
    output_dir: Path,
) -> None:
    """Run the axial stretching simulation and save centerline plots."""
    parameters = AxialStretchingParameters(
        n_elem=n_elem,
        final_time=final_time,
        time_step=time_step,
    )

    simulator, block = build_simulator(parameters, backend=backend)
    stepper = eaj.PositionVerletJAX()

    time_step = parameters.time_step
    total_steps = int(round(parameters.final_time / time_step))
    snapped_final_time = total_steps * time_step
    frame_dt = 1.0 / fps
    steps_per_frame = max(1, int(round(frame_dt / time_step)))
    frame_dt = steps_per_frame * time_step
    n_frames = int(round(snapped_final_time / frame_dt)) + 1

    callback_data: dict[str, list] = defaultdict(list)

    print("Starting axial stretching simulation...")
    print(f"  Backend: {backend}")
    print(f"  Time step: {parameters.time_step:.3e} s")
    print(f"  Final time: {snapped_final_time:.3f} s")
    print(f"  Capture every {steps_per_frame} steps")

    start = time.perf_counter()
    current_time = 0.0

    for frame_idx in tqdm(range(n_frames), desc="Axial stretching rollout"):
        jax.block_until_ready(block.position_collection_device)
        s_coords, x_coords = extract_centerline(block)
        callback_data["time"].append(current_time)
        callback_data["centerline_s"].append(s_coords)
        callback_data["centerline_x"].append(x_coords)

        if frame_idx == n_frames - 1:
            if current_time < snapped_final_time:
                stepper.integrate(
                    simulator,
                    time=current_time,
                    final_time=snapped_final_time,
                    dt=time_step,
                )
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
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "axial_stretching_data.npz"
    save_simulation_data(
        data_path,
        times=callback_data["time"],
        centerline_s=callback_data["centerline_s"],
        centerline_x=callback_data["centerline_x"],
        step_skip=steps_per_frame,
    )

    times = np.array(callback_data["time"])
    centerline_s = np.array(callback_data["centerline_s"], dtype=object)
    centerline_x = np.array(callback_data["centerline_x"], dtype=object)
    plot_centerline_evolution(
        output_dir / "centerline_evolution.png",
        times=times,
        centerline_s=centerline_s,
        centerline_x=centerline_x,
    )
    plot_final_centerline(
        output_dir / "final_centerline.png",
        centerline_s=centerline_s,
        centerline_x=centerline_x,
    )

    print("Simulation completed.")
    print(f"  Elapsed: {elapsed:.3f} s")
    print(f"  Saved {len(times)} frames to {data_path}")
    print(f"  Figures written to {output_dir}")


if __name__ == "__main__":
    main()
