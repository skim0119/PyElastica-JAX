# Continuum Snake GPU Case

A reduced, JAX-backed prototype of the continuum snake case used to validate the
device-side state updates. The same reduced problem is built twice, once on the
host with PyElastica and once with the JAX stepper, and their final rod states
are compared. It keeps the rod initialization, gravity, muscle-torque actuation,
anisotropic rod-plane friction, and analytical damping from the original snake
case, and is intended as a framework-validation example rather than a
replacement for the full continuum snake benchmark.

## Layout

```
ContinuumSnakeGPUCase/
├── README.md
├── run_continuum_snake_gpu.py   # entry point: SnakeParameters, rollouts, comparison
├── environment.py               # build_simulation / build_reference_simulation
├── post_processing.py           # difference summary, centerline plot, gait animation
├── render/                      # generated mp4 animation (gitignored)
└── operators/
    ├── muscle_torques.py         # SnakeMuscleTorquesJax + gravity/muscle actuation
    └── plane_contact.py          # SnakePlaneContactJax + friction helpers
```

## Usage

```bash
# CPU (default), full rollout
uv run --no-sync python run_continuum_snake_gpu.py

# GPU rollout
uv run --no-sync python run_continuum_snake_gpu.py --backend gpu

# Quick end-to-end check
uv run --no-sync python run_continuum_snake_gpu.py --smoke

# Save the comparison figure, gait animation (render/snake_gait.mp4), and archive
uv run --no-sync python run_continuum_snake_gpu.py --render --fps 30

# Re-render figure and animation from a saved archive
uv run --no-sync python post_processing.py --input snake_comparison.npz
```

The script reports the maximum absolute difference between the JAX and
PyElastica final states for position, director, velocity, angular velocity,
internal forces, internal torques, `sigma`, and `kappa`. With `--render`, the
JAX rollout is captured at `--fps` and encoded to `render/snake_gait.mp4` (a
top-view animation of the snake gait) via ffmpeg.
