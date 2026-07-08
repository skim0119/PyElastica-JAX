"""Post-processing and visualization for the muscular snake case.

Provides the light analysis path for the case: persist the recorded rollout to
an ``.npz`` archive, plot the projected center-of-mass velocity, and render a
top-down (x-y) locomotion animation of the body and muscle rods to an mp4. The
rendering helpers are called inline by ``run_muscular_snake.py --render`` or
standalone against a saved archive.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import click
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from tqdm import tqdm

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

DATA_DIR = _EXAMPLE_DIR / "data"
RENDER_DIR = _EXAMPLE_DIR / "render"


def save_simulation_data(
    output_path: Path,
    *,
    callback_history: Sequence[dict[str, list]],
    fps: float,
    period: float,
) -> None:
    """Persist the full rod rollout and body kinematics to an ``.npz`` archive.

    Rod node positions are stored per rod (their node counts differ), while the
    body center-of-mass and average velocity are stored once from rod ``0``.
    """
    body_history = callback_history[0]
    payload: dict[str, np.ndarray] = {
        "time": np.asarray(body_history["time"]),
        "n_rods": np.asarray(len(callback_history)),
        "fps": np.asarray(fps),
        "period": np.asarray(period),
        "avg_velocity": np.asarray(body_history["avg_velocity"]),
        "center_of_mass": np.asarray(body_history["center_of_mass"]),
    }
    for rod_idx, rod_history in enumerate(callback_history):
        payload[f"position__{rod_idx}"] = np.asarray(rod_history["position"])
        payload[f"radius__{rod_idx}"] = np.asarray(rod_history["radius"][0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def load_simulation_data(
    input_path: Path,
) -> tuple[list[dict[str, list]], float, float]:
    """Reconstruct the per-rod history and ``(fps, period)`` from an archive."""
    with np.load(input_path) as archive:
        n_rods = int(archive["n_rods"])
        fps = float(archive["fps"])
        period = float(archive["period"])
        times = archive["time"]
        callback_history: list[dict[str, list]] = [
            defaultdict(list) for _ in range(n_rods)
        ]
        for rod_idx in range(n_rods):
            positions = archive[f"position__{rod_idx}"]
            radius = archive[f"radius__{rod_idx}"]
            history = callback_history[rod_idx]
            history["time"] = list(times)
            history["position"] = list(positions)
            history["radius"] = [radius] * len(positions)
        body_history = callback_history[0]
        body_history["avg_velocity"] = list(archive["avg_velocity"])
        body_history["center_of_mass"] = list(archive["center_of_mass"])
    return callback_history, fps, period


def plot_snake_velocity(
    output_path: Path,
    *,
    body_history: dict[str, list],
    period: float,
) -> None:
    """Plot the projected snake center-of-mass velocity components."""
    time_per_period = np.array(body_history["time"]) / period
    avg_velocity = np.array(body_history["avg_velocity"])
    velocity_forward, velocity_lateral = _compute_projected_velocity(
        body_history, period
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    ax.plot(time_per_period, velocity_forward[:, 0], "r-", label="forward")
    ax.plot(
        time_per_period,
        velocity_lateral[:, 1],
        color=to_rgb("xkcd:bluish"),
        label="lateral",
    )
    ax.plot(time_per_period, avg_velocity[:, 2], "k-", label="normal")
    ax.set_xlabel("Time / period")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Muscular snake velocity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_animation_xy(
    output_path: Path,
    *,
    callback_history: Sequence[dict[str, list]],
    fps: int = 30,
    verbose: bool = True,
) -> Path:
    """Render a top-down animation of the snake body and muscle rods.

    Axis limits enclose the full trajectory so the net forward locomotion of the
    snake is visible as it slides across the frame. Returns the written path
    (extension may fall back to ``.gif`` when ffmpeg is unavailable).
    """
    sim_time = np.array(callback_history[0]["time"])
    n_rods = len(callback_history)

    try:
        writer = manimation.writers["ffmpeg"](fps=fps)
    except (KeyError, RuntimeError):
        writer = manimation.writers["pillow"](fps=fps)
        output_path = output_path.with_suffix(".gif")

    x_limits, y_limits = _trajectory_limits(callback_history)
    max_axis_length = max(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0])
    scaling_factor = (2 * 0.1) / max_axis_length * 2.6e3

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")

    rod_lines = []
    scatters = []
    for rod_idx in range(n_rods):
        position, radius = _unpack_rod_frame(callback_history, rod_idx, 0)
        color = "tab:red" if rod_idx == 0 else "tab:blue"
        rod_lines.append(ax.plot(position[0], position[1], color=color, lw=0.5)[0])
        scatters.append(
            ax.scatter(
                position[0],
                position[1],
                s=np.pi * (scaling_factor * radius) ** 2,
                color=color,
            )
        )
    com = callback_history[0]["center_of_mass"][0]
    (com_line,) = ax.plot(com[0], com[1], "k--", lw=2.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output_path), dpi=100):
        for frame_idx in tqdm(
            range(len(sim_time)),
            desc="Rendering muscular snake",
            disable=not verbose,
        ):
            for rod_idx in range(n_rods):
                position, radius = _unpack_rod_frame(
                    callback_history, rod_idx, frame_idx
                )
                rod_lines[rod_idx].set_data(position[0], position[1])
                scatters[rod_idx].set_offsets(position[:2].T)
                scatters[rod_idx].set_sizes(np.pi * (scaling_factor * radius) ** 2)
            com = callback_history[0]["center_of_mass"][frame_idx]
            com_line.set_data([com[0]], [com[1]])
            ax.set_title(f"Muscular snake (t = {sim_time[frame_idx]:.3f} s)")
            writer.grab_frame()

    plt.close(fig)
    return output_path


def render_outputs(
    output_dir: Path,
    *,
    callback_history: Sequence[dict[str, list]],
    fps: float,
    period: float,
    verbose: bool = True,
) -> None:
    """Write the velocity plot and locomotion animation for a completed run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    body_history = callback_history[0]
    time_span = float(body_history["time"][-1]) - float(body_history["time"][0])
    if len(body_history["time"]) >= 3 and time_span >= 2.0 * period:
        plot_snake_velocity(
            output_dir / "muscular_snake_velocity.png",
            body_history=body_history,
            period=period,
        )
    video_path = plot_animation_xy(
        output_dir / "muscular_snake_xy.mp4",
        callback_history=callback_history,
        fps=int(round(fps)),
        verbose=verbose,
    )
    if verbose:
        print(f"Rendered {len(body_history['time'])} frames to {video_path}")


def _unpack_rod_frame(
    callback_history: Sequence[dict[str, list]],
    rod_idx: int,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position, radius)`` with positions collocated at elements."""
    position = np.asarray(callback_history[rod_idx]["position"][frame_idx])
    radius = np.asarray(callback_history[rod_idx]["radius"][frame_idx])
    if position.shape[1] != radius.shape[0]:
        position = 0.5 * (position[..., 1:] + position[..., :-1])
    return position, radius


def _trajectory_limits(
    callback_history: Sequence[dict[str, list]],
    *,
    padding: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(x_limits, y_limits)`` enclosing every rod across all frames."""
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    for rod_history in callback_history:
        for position in rod_history["position"]:
            position = np.asarray(position)
            x_values.append(position[0])
            y_values.append(position[1])
    x_all = np.concatenate(x_values)
    y_all = np.concatenate(y_values)
    x_pad = padding * max(np.ptp(x_all), 1e-3)
    y_pad = padding * max(np.ptp(y_all), 1e-3)
    return (
        (float(x_all.min() - x_pad), float(x_all.max() + x_pad)),
        (float(y_all.min() - y_pad), float(y_all.max() + y_pad)),
    )


def _compute_projected_velocity(
    body_history: dict[str, list],
    period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose the body velocity into forward and lateral components."""
    time_per_period = np.array(body_history["time"]) / period
    avg_velocity = np.array(body_history["avg_velocity"])
    center_of_mass = np.array(body_history["center_of_mass"])

    period_step = int(period / (time_per_period[-1] - time_per_period[-2])) + 1
    number_of_period = int(time_per_period.shape[0] / period_step)
    center_of_mass_per_period = np.zeros((number_of_period - 2, 3))
    for period_idx in range(1, number_of_period - 1):
        center_of_mass_per_period[period_idx - 1] = np.mean(
            center_of_mass[
                (period_idx + 1) * period_step : (period_idx + 2) * period_step
            ]
            - center_of_mass[period_idx * period_step : (period_idx + 1) * period_step],
            axis=0,
        )

    direction_of_rod = np.mean(center_of_mass_per_period, axis=0)
    direction_of_rod /= np.linalg.norm(direction_of_rod, ord=2)

    velocity_mag_in_direction_of_rod = np.einsum(
        "ji,i->j", avg_velocity, direction_of_rod
    )
    velocity_in_direction_of_rod = np.einsum(
        "j,i->ji", velocity_mag_in_direction_of_rod, direction_of_rod
    )
    velocity_in_rod_roll_dir = avg_velocity - velocity_in_direction_of_rod
    return velocity_in_direction_of_rod, velocity_in_rod_roll_dir


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "data_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DATA_DIR / "muscular_snake_data.npz",
    show_default=True,
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Playback frame rate override (defaults to the run metadata).",
)
def main(data_path: Path, fps: float | None) -> None:
    """Render the velocity plot and locomotion mp4 from a saved archive."""
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run run_muscular_snake.py first.")
        return
    callback_history, saved_fps, period = load_simulation_data(data_path)
    render_outputs(
        RENDER_DIR,
        callback_history=callback_history,
        fps=fps if fps is not None else saved_fps,
        period=period,
    )


if __name__ == "__main__":
    main()
