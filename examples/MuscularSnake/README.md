# Muscular Snake Case (JAX)

A continuum musculoskeletal snake: a single Cosserat-rod body actuated by eight
muscle rods (four antagonist pairs) bound side-by-side to the body and driven by
a head-to-tail traveling wave. Sliding on an anisotropic-friction ground plane,
the lateral undulation is rectified into net forward locomotion.

Reference: Zhang, Chan, Parthasarathy, Gazzola, "Modeling and simulation of
complex dynamic musculoskeletal architectures", *Nat. Commun.* 10, 4825 (2019).

## Two-block layout

The body and muscles are packed into **separate memory blocks** on the same
device. This demonstrates same-device **cross-block rod-to-rod interaction**:

| Block | Rods | Operators |
|-------|------|-----------|
| Body | 1 | `GravityAnalyticalDamperJax`, `MuscularSnakePlaneContactJax` |
| Muscles | 8 | per-rod `AnalyticalLinearDamperJax`, `MuscularSnakeMuscleForcesBlockJax` |

Eight ``pairwise_interaction`` surface joints couple the body block to the muscle
block. Gravity and dissipation use built-in ``elastica_jax`` block operators.

## Physics

- **Muscle actuation**: traveling-wave contraction on all eight muscle rods
  (`operators/muscle_forces.py`).
- **Body forcing**: built-in gravity plus case-specific anisotropic ground contact
  (`operators/plane_contact.py`).
- **Dissipation**: built-in analytical dampers (block on body, per-rod on muscles).
- **Surface joints**: cross-block muscle-to-body coupling
  (`operators/surface_joint.py`).

## Structure

```
MuscularSnake/
├── run_muscular_snake.py   # config (MuscularSnakeParameters) + CLI entry point
├── environment.py          # two-block simulator assembly
├── block_utils.py          # rod types, packed-block extraction, body kinematics
├── post_processing.py      # npz I/O, velocity plot, locomotion animation
├── operators/
│   ├── muscle_forces.py     # traveling-wave actuation on the muscle block
│   ├── plane_contact.py     # anisotropic ground contact on the body block
│   └── surface_joint.py     # side-by-side surface joint (rod-rod)
├── data/                   # npz rollout archive, gitignored
└── render/                 # velocity plot + mp4, gitignored
```

## Usage

Smoke test (short horizon, exercises the full operator stack):

```bash
uv run --no-sync python examples/MuscularSnake/run_muscular_snake.py --smoke --render
```

Full locomotion run (default 16 s) with inline rendering:

```bash
uv run --no-sync python examples/MuscularSnake/run_muscular_snake.py --render
```

Add `--gpu` for the CUDA backend. Scale and physics are configured by editing
`MuscularSnakeParameters` at the top of `run_muscular_snake.py`; the CLI only
selects run mode and frame cadence.

Render from a saved archive standalone:

```bash
uv run --no-sync python examples/MuscularSnake/post_processing.py \
  --input examples/MuscularSnake/data/muscular_snake_data.npz
```
