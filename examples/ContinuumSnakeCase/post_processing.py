"""Comparison and rendering for the reduced continuum snake GPU case.

This validation example compares the JAX rollout against the PyElastica
reference and can render the final centerlines and an mp4 animation of the JAX
snake gait. Functions are called directly by the run script; the ``main`` CLI
re-runs comparison and rendering from a saved ``.npz`` archive.
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_COMPARISON_KEYS = (
    "position_collection",
    "director_collection",
    "velocity_collection",
    "omega_collection",
    "internal_forces",
    "internal_torques",
    "sigma",
    "kappa",
)


def max_abs_diff(first: np.ndarray, second: np.ndarray) -> float:
    """Return the maximum absolute element-wise difference of two arrays."""
    return float(np.max(np.abs(first - second)))


def summarize_results(
    cpu_state: dict[str, np.ndarray], jax_state: dict[str, np.ndarray]
) -> dict[str, float]:
    """Return per-field maximum absolute differences between two rod states."""
    return {
        key: max_abs_diff(cpu_state[key], jax_state[key]) for key in _COMPARISON_KEYS
    }


def print_summary(diffs: dict[str, float]) -> None:
    """Print the field-wise maximum absolute differences."""
    print("Max absolute differences vs CPU reference:")
    for key, value in diffs.items():
        print(f"  {key}: {value:.3e}")


def save_comparison(
    output_path: str | Path,
    *,
    cpu_state: dict[str, np.ndarray],
    jax_state: dict[str, np.ndarray],
    diffs: dict[str, float],
    trajectory: dict[str, np.ndarray],
) -> None:
    """Persist final centerlines, diffs, and the JAX gait trajectory.

    Storing the trajectory lets :func:`animate_snake_gait` be re-run offline
    from the archive without repeating the rollout.
    """
    np.savez(
        output_path,
        cpu_position=cpu_state["position_collection"],
        jax_position=jax_state["position_collection"],
        diff_keys=np.array(list(diffs.keys())),
        diff_values=np.array(list(diffs.values())),
        trajectory_time=trajectory["time"],
        trajectory_position=trajectory["position"],
    )


def load_comparison(input_path: str | Path) -> dict:
    """Load a comparison archive written by :func:`save_comparison`."""
    data = np.load(input_path, allow_pickle=True)
    diffs = dict(
        zip(
            [str(key) for key in data["diff_keys"]],
            [float(value) for value in data["diff_values"]],
            strict=True,
        )
    )
    return {
        "cpu_position": data["cpu_position"],
        "jax_position": data["jax_position"],
        "diffs": diffs,
        "trajectory": {
            "time": data["trajectory_time"],
            "position": data["trajectory_position"],
        },
    }


def plot_final_centerlines(
    cpu_position: np.ndarray,
    jax_position: np.ndarray,
    *,
    filename: str | Path = "final_centerlines.png",
    save_figure: bool = True,
    show: bool = False,
) -> None:
    """Plot the final snake centerlines (top view) for both backends.

    The rod crawls on the plane spanned by the initial tangent (``z``) and the
    lateral (``x``) directions, so a top-down ``x`` vs ``z`` view captures the
    gait shape.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        cpu_position[2],
        cpu_position[0],
        "o-",
        color="tab:blue",
        markersize=3,
        label="CPU reference",
    )
    ax.plot(
        jax_position[2],
        jax_position[0],
        "x--",
        color="tab:orange",
        markersize=4,
        label="JAX rollout",
    )
    ax.set_xlabel("z (m)", fontsize=12)
    ax.set_ylabel("x (m)", fontsize=12)
    ax.set_title("Final snake centerline (top view)", fontsize=14)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_figure:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {filename}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_snake_gait(
    trajectory: dict[str, np.ndarray],
    *,
    video_path: str | Path = "render/snake_gait.mp4",
    fps: int = 30,
    dpi: int = 100,
) -> None:
    """Render a top-view mp4 animation of the JAX snake gait.

    Parameters
    ----------
    trajectory
        Mapping with ``time`` of shape ``(n_frames,)`` and ``position`` of shape
        ``(n_frames, 3, n_nodes)`` giving the rod centerline per captured frame.
    video_path
        Output mp4 path; the parent ``render/`` directory is created if needed.
    fps
        Frames per second of the encoded video.
    dpi
        Rendering resolution.
    """
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(trajectory["time"])
    positions = np.asarray(trajectory["position"])
    z_coords = positions[:, 2, :]
    x_coords = positions[:, 0, :]

    pad_z = 0.05 * (float(z_coords.max() - z_coords.min()) + 1.0e-9)
    pad_x = 0.05 * (float(x_coords.max() - x_coords.min()) + 1.0e-9)

    fig, ax = plt.subplots(figsize=(10, 4))
    (line,) = ax.plot([], [], "o-", color="tab:orange", markersize=3)
    ax.set_xlim(float(z_coords.min()) - pad_z, float(z_coords.max()) + pad_z)
    ax.set_ylim(float(x_coords.min()) - pad_x, float(x_coords.max()) + pad_x)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("z (m)", fontsize=12)
    ax.set_ylabel("x (m)", fontsize=12)
    ax.grid(True, alpha=0.3)

    writer = manimation.writers["ffmpeg"](
        fps=fps,
        metadata={"title": "Continuum snake gait", "artist": "PyElastica"},
    )
    with writer.saving(fig, str(video_path), dpi=dpi):
        for frame in tqdm(range(positions.shape[0]), desc="Rendering snake gait"):
            line.set_data(z_coords[frame], x_coords[frame])
            ax.set_title(
                f"JAX snake gait (top view) - t = {times[frame]:.3f} s", fontsize=14
            )
            writer.grab_frame()
    plt.close(fig)
    print(f"Animation saved to {video_path}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default="snake_comparison.npz",
    show_default=True,
)
@click.option("--fps", type=int, default=30, show_default=True)
@click.option(
    "--show/--no-show",
    default=False,
    show_default=True,
    help="Display the figure interactively after saving.",
)
def main(input_path: Path, fps: int, show: bool) -> None:
    """Re-render the snake comparison from a saved ``.npz`` archive."""
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        print("Run run_continuum_snake_gpu.py --render first to generate it.")
        return
    data = load_comparison(input_path)
    print_summary(data["diffs"])
    plot_final_centerlines(data["cpu_position"], data["jax_position"], show=show)
    animate_snake_gait(
        data["trajectory"],
        video_path=input_path.parent / "render" / "snake_gait.mp4",
        fps=fps,
    )


if __name__ == "__main__":
    main()
