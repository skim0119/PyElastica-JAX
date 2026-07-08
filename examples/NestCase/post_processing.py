"""Post-processing and visualization for the JAX nest packing example."""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

DEFAULT_CYLINDER_RADIUS = 69.85


def load_simulation_data(filename: str | Path = "nest_simulation_data.npz") -> dict:
    """Load simulation diagnostics from a saved ``.npz`` archive.

    Parameters
    ----------
    filename
        Path to the saved simulation data file.

    Returns
    -------
    dict
        Dictionary containing simulation data.
    """
    data = np.load(filename, allow_pickle=True)
    result = {
        "time": data["time"],
        "nest_height_max": data["nest_height_max"],
        "nest_height_min": data["nest_height_min"],
        "total_energy": data["total_energy"],
        "rod_positions": data["rod_positions"],
    }
    if "step" in data:
        result["step"] = data["step"]
    if "step_skip" in data:
        result["step_skip"] = data["step_skip"]
    if "cylinder_radius" in data:
        result["cylinder_radius"] = float(np.asarray(data["cylinder_radius"]).item())
    else:
        result["cylinder_radius"] = DEFAULT_CYLINDER_RADIUS
    return result


def plot_nest_height_evolution(
    data: dict,
    *,
    save_figure: bool = True,
    filename: str | Path = "nest_height_evolution.png",
    show: bool = False,
) -> None:
    """Plot the evolution of nest height over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    time = data["time"]
    height_max = data["nest_height_max"]
    height_min = data["nest_height_min"]

    ax.plot(time, height_max, label="Max height", linewidth=2)
    ax.plot(time, height_min, label="Min height", linewidth=2)
    ax.fill_between(time, height_min, height_max, alpha=0.3, label="Height range")

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Height (mm)", fontsize=14)
    ax.set_title("Nest Height Evolution", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_figure:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {filename}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_energy_evolution(
    data: dict,
    *,
    save_figure: bool = True,
    filename: str | Path = "energy_evolution.png",
    show: bool = False,
) -> None:
    """Plot the evolution of total mechanical energy over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    time = data["time"]
    energy = data["total_energy"]

    ax.plot(time, energy, linewidth=2, color="red")
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Total Energy (g mm^2 / s^2)", fontsize=14)
    ax.set_title("Total Energy Evolution", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    if save_figure:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {filename}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_3d_animation(
    data: dict,
    *,
    video_name: str | Path = "nest_packing_3d.mp4",
    fps: int = 30,
    cylinder_radius: float | None = None,
    step: int = 1,
    dpi: int = 100,
) -> None:
    """Create a 3D animation of the nest packing process."""
    rod_positions = data["rod_positions"]
    time = data["time"]
    if cylinder_radius is None:
        cylinder_radius = data.get("cylinder_radius", DEFAULT_CYLINDER_RADIUS)

    filtered_frames = list(range(0, len(rod_positions), step))
    num_frames = len(filtered_frames)

    print(f"Creating 3D animation with {num_frames} frames at {fps} fps...")
    print(f"  Total frames available: {len(rod_positions)}")
    print(f"  Using every {step} frame(s)")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    max_height = np.max(data["nest_height_max"]) * 1.2
    ax.set_xlim(-cylinder_radius * 1.2, cylinder_radius * 1.2)
    ax.set_ylim(-cylinder_radius * 1.2, cylinder_radius * 1.2)
    ax.set_zlim(0, max_height)

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_zlabel("Z (mm)", fontsize=12)

    theta = np.linspace(0, 2 * np.pi, 50)
    z_boundary = np.linspace(0, max_height, 20)
    theta_grid, z_grid = np.meshgrid(theta, z_boundary)
    x_boundary = cylinder_radius * np.cos(theta_grid)
    y_boundary = cylinder_radius * np.sin(theta_grid)
    ax.plot_surface(
        x_boundary,
        y_boundary,
        z_grid,
        alpha=0.1,
        color="gray",
        edgecolor="none",
    )

    num_rods = len(rod_positions[0])
    rod_lines = []
    for _ in range(num_rods):
        (line,) = ax.plot([], [], [], "b-", linewidth=1.5, alpha=0.6)
        rod_lines.append(line)

    try:
        ffmpeg_writer = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Nest Packing Simulation",
            artist="PyElastica",
            comment="Nest simulation animation",
        )
        writer = ffmpeg_writer(fps=fps, metadata=metadata)
    except (KeyError, RuntimeError) as exc:
        print(f"Error setting up FFMpeg writer: {exc}")
        print("Trying alternative writer...")
        try:
            pillow_writer = manimation.writers["pillow"]
            writer = pillow_writer(fps=fps)
            video_name = Path(video_name).with_suffix(".gif")
            print(f"Using Pillow writer, output will be {video_name}")
        except (KeyError, RuntimeError) as exc2:
            print(f"Error: Could not set up any video writer: {exc2}")
            print("Please install ffmpeg or pillow for video generation.")
            return

    with writer.saving(fig, str(video_name), dpi=dpi):
        for frame_idx in tqdm(filtered_frames, desc="Rendering frames"):
            frame_data = rod_positions[frame_idx]
            current_time = time[frame_idx]

            for rod_idx, line in enumerate(rod_lines):
                if rod_idx < len(frame_data):
                    positions = frame_data[rod_idx]
                    line.set_data(positions[0, :], positions[1, :])
                    line.set_3d_properties(positions[2, :])
                else:
                    line.set_data([], [])
                    line.set_3d_properties([])

            ax.set_title(
                f"Nest Packing Simulation - Time: {current_time:.4f} s",
                fontsize=14,
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Animation saved to {video_name}")


def plot_final_configuration(
    data: dict,
    *,
    cylinder_radius: float | None = None,
    save_figure: bool = True,
    filename: str | Path = "final_nest_configuration.png",
    show: bool = False,
) -> None:
    """Plot the final configuration of the nest."""
    if cylinder_radius is None:
        cylinder_radius = data.get("cylinder_radius", DEFAULT_CYLINDER_RADIUS)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    final_positions = data["rod_positions"][-1]
    for rod_pos in final_positions:
        ax.plot(
            rod_pos[0, :],
            rod_pos[1, :],
            rod_pos[2, :],
            "b-",
            linewidth=1.5,
            alpha=0.6,
        )

    theta = np.linspace(0, 2 * np.pi, 50)
    z_boundary = np.linspace(0, np.max(data["nest_height_max"]) * 1.2, 20)
    theta_grid, z_grid = np.meshgrid(theta, z_boundary)
    x_boundary = cylinder_radius * np.cos(theta_grid)
    y_boundary = cylinder_radius * np.sin(theta_grid)
    ax.plot_surface(
        x_boundary,
        y_boundary,
        z_grid,
        alpha=0.1,
        color="gray",
        edgecolor="none",
    )

    max_height = np.max(data["nest_height_max"])
    ax.set_xlim(-cylinder_radius * 1.2, cylinder_radius * 1.2)
    ax.set_ylim(-cylinder_radius * 1.2, cylinder_radius * 1.2)
    ax.set_zlim(0, max_height * 1.2)

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_zlabel("Z (mm)", fontsize=12)
    ax.set_title("Final Nest Configuration", fontsize=14)

    plt.tight_layout()
    if save_figure:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {filename}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def render_outputs(
    data: dict,
    *,
    output_dir: str | Path,
    fps: int = 30,
    animate: bool = True,
) -> None:
    """Write diagnostic plots and an optional mp4 animation to ``output_dir``.

    Parameters
    ----------
    data : dict
        Diagnostic history (as returned by the run script or loaded from npz).
    output_dir : str | pathlib.Path
        Directory where figures and the animation are written.
    fps : int, optional
        Animation frame rate.
    animate : bool, optional
        Whether to render the 3D mp4 animation (requires ffmpeg).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_nest_height_evolution(data, filename=output_dir / "nest_height_evolution.png")
    plot_energy_evolution(data, filename=output_dir / "energy_evolution.png")
    plot_final_configuration(data, filename=output_dir / "final_nest_configuration.png")
    if not animate:
        return
    num_frames = len(data["rod_positions"])
    step = max(1, num_frames // 500) if num_frames > 1000 else 1
    try:
        plot_3d_animation(
            data,
            video_name=output_dir / "nest_packing_3d.mp4",
            fps=fps,
            step=step,
        )
    except OSError as exc:
        print(f"Warning: could not create animation (ffmpeg missing?): {exc}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path(__file__).resolve().parent / "data" / "nest_diagnostics.npz",
    show_default=True,
)
@click.option("--fps", type=int, default=30, show_default=True)
@click.option(
    "--skip-animation/--no-skip-animation",
    default=False,
    show_default=True,
    help="Skip the 3D animation export.",
)
def main(input_path: Path, fps: int, skip_animation: bool) -> None:
    """Generate nest diagnostic plots and optional animation from saved data."""
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run run_nest.py first.")
        return
    data = load_simulation_data(input_path)
    render_outputs(
        data,
        output_dir=Path(__file__).resolve().parent / "render",
        fps=fps,
        animate=not skip_animation,
    )


if __name__ == "__main__":
    main()
