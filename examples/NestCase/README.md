# Nest Packing Case (JAX)

Randomly oriented rods settle under gravity inside a cylinder and pack into a nest-like structure. This case reproduces the C++ `nest-simulator` (`Nest.cpp`).

> This case is written in its native **mm-g-s** unit system, differing only in how dissipation is handled.

Reference: Bhosale, Weiner, Butler, Kim, Gazzola & King, "Micromechanical origin of plasticity and hysteresis in nestlike packings", *Phys. Rev. Lett.* 128, 198003 (2022).

## Physics

- **Gravity**: uniform body force on every node.
- **Rod-rod contact**: spatial-hash capsule contact with the Hertzian
  `F = zeta * gamma^1.5` normal law and `gamma^0.5` normal damping, a soft-to-hard
  stiffness ramp at `t = 0.2 s`, and isotropic kinetic Coulomb friction after
  `t = 0.5 s` (`CapsuleContactOp`, `hertzian=True`).
- **Substrate**: node-based ground plane and cylindrical wall using the same
  Hertzian law plus kinetic friction (`operators/substrate_interaction.py`).
- **Dissipation**: analytical linear damper (`AnalyticalLinearDamperJax`).

## Units (mm-g-s)

All quantities are in millimetres, grams, and seconds, matching the C++ source
so its constants are used verbatim.

| Quantity | Value |
|----------|-------|
| Rod length / radius | `75.74 mm` / `1.2075 mm` |
| Density | `8.94e-3 g/mm^3` |
| Young's modulus | `12e9 g/(mm s^2)` (= 12 GPa) |
| Gravity | `981 mm/s^2` |
| Cylinder radius | `69.85 mm` |
| Time step | `8.8e-7 s` |
| Rod-rod stiffness (soft / hard) | `1500` / `1e6` |
| Ground/wall stiffness | `1e6` |

## Structure

```
NestCase/
├── run_nest.py            # config (NestParameters) + CLI entry point
├── environment.py         # NestSimulator, build_simulation(), rod placement
├── post_processing.py     # plots + mp4 rendering
├── operators/
│   └── substrate_interaction.py# ground + cylinder wall (Hertzian + friction)
├── data/                  # diagnostics (.npz), gitignored
└── render/                # figures + mp4, gitignored
```

## Usage

Smoke test (few rods, short horizon, exercises the full operator stack):

```bash
uv run --no-sync python examples/NestCase/run_nest.py --smoke
```

Full-size run (default 455 rods to 50 s; use a GPU):

```bash
uv run --no-sync python examples/NestCase/run_nest.py --gpu
```

Scale and physics are configured by editing `NestParameters` at the top of
`run_nest.py` (the CLI only selects run mode). For a mid-size validation run that
crosses the hard-contact and friction activation times in a few minutes on CPU,
set `num_rods = 50` and `final_time = 1.0` there, then run:

```bash
uv run --no-sync python examples/NestCase/run_nest.py
```

Render plots and an mp4 (either inline via `--render`, or standalone on saved
data):

```bash
uv run --no-sync python examples/NestCase/run_nest.py --smoke --render
uv run --no-sync python examples/NestCase/post_processing.py \
  --input examples/NestCase/data/nest_diagnostics.npz
```

## Deviations from the C++ case

- **Dissipation**: the C++ force-based viscous damping (`f = -nu * v`, `nu = 0.2`)
  is replaced by the analytical linear damper. The default `damping_constant`
  (`5 s^-1`) equals the C++ effective decay rate `nu / (density * area)`.
- **Top lid**: the stress-controlled cyclic compression platen (active only at
  `t > 1 s`) is not yet implemented, so this case covers gravitational settling
  and packing.
- **Static friction**: only isotropic kinetic Coulomb friction is implemented;
  the C++ anisotropic static friction is deferred with the top-lid work.
- **Contact geometry**: rod-rod contact uses per-element capsules (finer than the
  C++ coarse 1-2 segment near-list model) and distributes each element contact
  force to its two nodes summing to the total (the C++ node convention applies
  the full force to each endpoint).
- **Shear**: the shear correction factor is overridden to `4/3` to match the C++
  value (PyElastica defaults to `27/28`).
