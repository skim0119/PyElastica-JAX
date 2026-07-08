"""Run the JAX Timoshenko beam example and save diagnostics."""

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
    TimoshenkoParameters,
    build_simulator,
    extract_centerline,
    plot_final_deflection,
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
@click.option("--n-elem", type=int, default=100, show_default=True)
@click.option("--final-time", type=float, default=10.0, show_default=True)
@click.option("--fps", type=float, default=10.0, show_default=True)
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
    fps: float,
    output_dir: Path,
) -> None:
    """Run the Timoshenko beam simulation and save the deflection plot."""
    parameters = TimoshenkoParameters(n_elem=n_elem, final_time=final_time)

    simulator, block, rod = build_simulator(parameters, backend=backend)
    stepper = eaj.PositionVerletJAX()

    time_step = parameters.time_step
    total_steps = int(round(parameters.final_time / time_step))
    snapped_final_time = total_steps * time_step
    frame_dt = 1.0 / fps
    steps_per_frame = max(1, int(round(frame_dt / time_step)))
    frame_dt = steps_per_frame * time_step
    n_frames = int(round(snapped_final_time / frame_dt)) + 1

    callback_data: dict[str, list] = defaultdict(list)

    print("Starting Timoshenko beam simulation...")
    print(f"  Backend: {backend}")
    print(f"  Time step: {parameters.time_step:.3e} s")
    print(f"  Final time: {snapped_final_time:.3f} s")
    print(f"  Capture every {steps_per_frame} steps")

    start = time.perf_counter()
    current_time = 0.0

    for frame_idx in tqdm(range(n_frames), desc="Timoshenko rollout"):
        jax.block_until_ready(block.position_collection_device)
        s_coords, deflection = extract_centerline(block)
        callback_data["time"].append(current_time)
        callback_data["centerline_s"].append(s_coords)
        callback_data["centerline_deflection"].append(deflection)

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

    data_path = output_dir / "timoshenko_data.npz"
    save_simulation_data(
        data_path,
        times=callback_data["time"],
        centerline_s=callback_data["centerline_s"],
        centerline_deflection=callback_data["centerline_deflection"],
        step_skip=steps_per_frame,
    )

    block.from_device(update_rods=True)

    centerline_s = np.array(callback_data["centerline_s"], dtype=object)
    centerline_deflection = np.array(
        callback_data["centerline_deflection"], dtype=object
    )
    plot_final_deflection(
        output_dir / "timoshenko_deflection.png",
        rod=rod,
        end_force=parameters.end_force,
        centerline_s=centerline_s,
        centerline_deflection=centerline_deflection,
    )

    print("Simulation completed.")
    print(f"  Elapsed: {elapsed:.3f} s")
    print(f"  Saved {len(callback_data['time'])} frames to {data_path}")
    print(f"  Figures written to {output_dir}")


if __name__ == "__main__":
    main()
