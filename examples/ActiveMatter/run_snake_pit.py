"""GPU JAX reproduction of the enclosed snake-pit active-matter case."""

from __future__ import annotations

import click
from collections.abc import Sequence
from dataclasses import dataclass
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from forcing import ActiveMatterForcingJax, spline_actuation_amplitude
from frame_io import (
    extract_stacked_positions,
    save_frame_positions,
    save_simulation_metadata,
)
from simulation_runtime import resolve_output_dir
from utils import instantiate_rods_in_cylinder

import elastica as ea
import elastica_jax as eaj
import jax
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)


class SnakePitSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    pass


@dataclass
class SnakePitParameters:
    n_elements: int = 20
    n_snakes: int = 4
    length: float = 0.35
    radius_ratio: float = 0.011
    density: float = 1000.0
    youngs_modulus: float = 1.0e6
    time_period: float = 2.0
    wave_length: float = 1.0
    contact_stiffness: float = 1.0e4
    contact_damping: float = 1.0e-3
    gravitational_acc: float = -0.1
    damping_rate: float = 1.0e-4
    time_step: float = 5.0e-5
    final_time: float = 20.0
    activation_start_time_nd: float = 5.0
    packing_initial_vertical_span_ratio: float = 1.0
    packing_initial_radial_span_ratio: float = 1.0
    wall_distance_ratio: float = 2.0
    steps_between_detection: int = 0

    @property
    def radius(self) -> float:
        return self.radius_ratio * self.length

    def pit_walls(self) -> tuple[np.ndarray, np.ndarray]:
        origins = [[0.0, 0.0, 0.0]]
        normals = [[0.0, 0.0, 1.0]]
        half = 0.5 * self.wall_distance_ratio * self.length
        origins += [
            [-half, 0.0, 0.0],
            [half, 0.0, 0.0],
            [0.0, -half, 0.0],
            [0.0, half, 0.0],
        ]
        normals += [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
        return np.asarray(origins), np.asarray(normals)


def build_simulator(
    parameters: SnakePitParameters,
    *,
    devices: Sequence[jax.Device],
    dtype: np.dtype,
    seed: int,
    block_checkpoint: Path,
):
    wall_origins, wall_normals = parameters.pit_walls()

    simulator = SnakePitSimulator()
    rod_block_cls = eaj.configure_rod_block_sharded(
        devices=devices,
        device_dtype=dtype,
        block_checkpoint=block_checkpoint,
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block_cls)

    for start, direction in instantiate_rods_in_cylinder(parameters, seed):
        # Calculate unit-normal vector
        normal_seed = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(normal_seed, direction)) > 0.9:
            normal_seed = np.array([0.0, 1.0, 0.0])
        normal = normal_seed - np.dot(normal_seed, direction) * direction
        normal /= np.linalg.norm(normal)
        # Define rod
        rod = ea.CosseratRod.straight_rod(
            parameters.n_elements,
            start,
            direction,
            normal,
            parameters.length,
            parameters.radius,
            parameters.density,
            youngs_modulus=parameters.youngs_modulus,
            # shear modulus is not used in the simulation, and it is
            # slightly controversial, so avoid explicit setting for now.
            # shear_modulus=parameters.youngs_modulus / 1.5,
        )
        simulator.append(rod)

    # Define forcing (active actuation)
    spline_amplitude = spline_actuation_amplitude(parameters.n_elements)
    simulator.operate_block(rod_block_cls).using(
        ActiveMatterForcingJax,
        parameters=parameters,
        n_snakes=parameters.n_snakes,
        n_elements=parameters.n_elements,
        gravity_axis=np.array([0.0, 0.0, 1.0]),
        spline_amplitude=spline_amplitude,
        ramp="sinusoidal",
    )

    # Rod-to-rod contact
    simulator.operate_block(rod_block_cls).using(
        eaj.CapsuleContactOp,
        n_elements_per_rod=parameters.n_elements,
        contact_stiffness=parameters.contact_stiffness,
        contact_damping=parameters.contact_damping,
        steps_between_detection=parameters.steps_between_detection,
        time_step=parameters.time_step,
    )
    simulator.operate_block(rod_block_cls).using(
        eaj.WallContactOp,
        n_elements_per_rod=parameters.n_elements,
        wall_origins=wall_origins,
        wall_normals=wall_normals,
        contact_stiffness=parameters.contact_stiffness,
        contact_damping=parameters.contact_damping,
    )
    simulator.operate_block(rod_block_cls).using(
        eaj.AnalyticalLinearDamperJax,
        damping_constant=parameters.damping_rate,
        time_step=parameters.time_step,
    )
    simulator.finalize()
    block = tuple(simulator.final_systems())[0]
    metadata = eaj.build_block_capsule_metadata(
        block, n_elements_per_rod=parameters.n_elements
    )
    eaj.install_capsule_contact_state(
        block,
        metadata,
        device=execution_mesh.devices[0],
        dtype=dtype,
    )
    return simulator, block


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(("cpu", "cuda"), case_sensitive=False),
    default="cpu",
    show_default=True,
)
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
@click.option("--n-elements", type=int, default=20, show_default=True)
@click.option("--n-snakes", type=int, default=4, show_default=True)
@click.option("--final-time", type=float, default=20.0, show_default=True)
@click.option("--time-step", type=float, default=5.0e-5, show_default=True)
@click.option("--seed", default=2026, show_default=True)
@click.option(
    "--run-name",
    default=None,
    help='Output directory suffix: writes to "output" or "output_<run-name>".',
)
@click.option(
    "--fps",
    default=25,
    show_default=True,
    help="Frame capture rate for HDF5 output.",
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
    default="block_checkpoint.h5",
    show_default=True,
    help=(
        "HDF5 block state path: load when the file exists, otherwise pack rods "
        "and save after construction."
    ),
)
def main(
    backend: str,
    mesh: str,
    n_elements: int,
    n_snakes: int,
    final_time: float,
    time_step: float,
    seed: int,
    run_name: str | None,
    fps: float,
    save_workers: int,
    block_checkpoint: Path,
) -> None:
    parameters = SnakePitParameters(
        n_elements=n_elements,
        n_snakes=n_snakes,
        final_time=final_time,
        time_step=time_step,
    )

    devices = eaj.execution_mesh_for_block_checkpoint(
        block_checkpoint,
        mesh_name=mesh,
        backend=backend,
        n_rods=parameters.n_snakes,
    )
    dtype = np.dtype(np.float64)
    simulator, block = build_simulator(
        parameters,
        devices=devices,
        dtype=dtype,
        seed=seed,
        block_checkpoint=block_checkpoint,
    )
    wall_origins, wall_normals = parameters.pit_walls()
    output_dir = resolve_output_dir(run_name)

    steps = int(round(parameters.final_time / parameters.time_step))

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
    for frame_idx in tqdm(range(n_frames)):
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
    print(f"case=snake-pit steps={steps}")
    print(
        f"frames={n_frames} fps={fps} steps_per_frame={steps_per_frame} frame_dt={frame_dt:.6e}"
    )
    print(f"saved_frames={output_dir}")
    print(f"elapsed={elapsed:.6f}s finite={np.isfinite(positions).all()}")
    print(
        "backend={backend} mesh={mesh} shards={shards} dtype={dtype}".format(
            backend=backend,
            mesh=mesh,
            shards=block.n_shards,
            dtype=dtype,
        )
    )


if __name__ == "__main__":
    main()
