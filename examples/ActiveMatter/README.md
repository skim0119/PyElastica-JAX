# Active Matter (Snake Pit)

GPU JAX reproduction of the enclosed snake-pit active-matter case: randomly
packed active snakes inside a floor-and-four-wall pit, driven by a traveling-wave
internal torque and resolved with capsule-capsule rod-rod contact and
capsule-half-space wall contact.

## Layout

```
ActiveMatter/
├── run_snake_pit.py      # entry point: config + rollout + HDF5 frame dump
├── environment.py        # SnakePitSimulator mixin + build_simulation + packing
├── post_processing.py    # MPI-parallel renderer (data/ -> render/)
├── frame_io.py           # HDF5 frame I/O and output-path layout
├── operators/            # custom JAX operators (traveling-wave forcing)
├── data/                 # HDF5 frames + metadata (generated)
└── render/               # PNG frames + output.mp4 (generated)
```

Configuration lives in `SnakePitParameters` at the top of `run_snake_pit.py`.
Edit it (or its defaults) to change scale or physics; the CLI only selects run
mode and I/O options.

## Run

```bash
uv run --no-sync python examples/ActiveMatter/run_snake_pit.py            # CPU
uv run --no-sync python examples/ActiveMatter/run_snake_pit.py --gpu      # CUDA
uv run --no-sync python examples/ActiveMatter/run_snake_pit.py --smoke    # short
```

`--smoke` downscales the run while still exercising every operator. Use
`-N/--n-snakes` and `-T/--final-time` for scaling studies.

Frames are streamed to `data/` (or `data_<run-name>/` when `--run-name` is set):

- `metadata.h5` — case parameters, fps, wall geometry
- `frame_<idx>.h5` — rod node positions `(n_snakes, n_nodes, 3)`

## Render

Render top/side planar views to `render/png/` and assemble `render/output.mp4`:

```bash
uv run --no-sync python examples/ActiveMatter/post_processing.py
uv run --no-sync python examples/ActiveMatter/post_processing.py --run-name pit01
```

Pass `--render` to `run_snake_pit.py` to render immediately after the run. The
renderer splits frames across MPI ranks and can be launched with `mpiexec`:

```bash
mpiexec -n 4 uv run --no-sync python examples/ActiveMatter/post_processing.py --run-name pit01
```

## Active Matter Reference Physics

Reference C++ physics spec for exact reproduction of the snake-pit case. For
layout, configuration, run commands, and rendering, see [README.md](README.md).

Section 16 records where the current JAX code (`environment.py`,
`operators/active_matter_forcing.py`, `elastica_jax.contact.*`) deviates.

> Per time step: find nearby pairs (spatial hash) -> compute contact points -> spring-damper response -> apply loads to nodes/elements.

> The contact orientation is built from the tangent, **not** from the rod directors. Using directors for contact orientation caused a shear-wave instability under discrete contact events.

### Materials and effective contact parameters

One material is shared by all rods, floor, and walls:

- restitution `e = 0.24`
- stiffness `k_mat = contact_stiffness`
- damping `c_mat = contact_damping` (same for normal and tangential)
- friction `mu_s = mu_d = 1e-10` (effectively frictionless)
- threshold `1e-8` (contact proximity, penetration, or parallel-check)

When two bodies contact, stiffness and damping combine **in series**. Because
both sides share the same material, the effective pair values are halved:
`k_eff = k_mat / 2`, `c_eff = c_mat / 2`. This matters for exact reproduction.

### Hierarchical contact detection phases:
#### Finding nearby pairs (broad phase)

Each element gets an axis-aligned bounding box. Elements are hashed into a
spatial grid; only pairs in neighboring cells are tested for box overlap. Walls
are infinite planes — either modeled with a very large bounding box or handled
by a dedicated wall-contact rule.

#### Contact point geometry (fine phase)

Each element is a line segment from `c - (l/2) a` to `c + (l/2) a`. Find the
closest points on the two segments. Let `d_axis` = distance between them.

- If `d_axis < threshold`, skip (degenerate case, including adjacent elements on
  the same rod).
- Surface gap `d = d_axis - r1 - r2`; in contact if `d < threshold`.
- Contact normal points from element 2 toward element 1.
- If two elements are nearly parallel and overlap along their length: two contact
  points; otherwise one contact point at the surface midpoint.

### Arena description

The simulation arena is a rectangular box, centered at the origin, with each side of length `L_wall = wall_distance_ratio * base_length`. The box is enclosed by six walls; each wall is modeled as an infinite plane, defined by an anchor point and an inward-facing normal vector. Opposite walls are positioned at `±L_wall/2` along the two in-plane axes, with normals pointing inward (from the wall into the allowed region).

**Element-wall contact detection:**
At most one contact is possible per element-wall pair. The wall serves as an infinite plane, and the element (rod segment) is checked for proximity to it.

- If the element axis is nearly parallel to the wall normal, use the element center as the closest point.
- Otherwise, use the end cap of the element closest to the wall.
- The contact gap is `d = n · (closest_point - wall_anchor) - r`, where `n` is the wall normal, `r` is the element radius, and `closest_point` is chosen as above. Contact is detected if `d < threshold`.

> Note: The `0.5 * wall_distance_ratio * base_length` offset (see above) determines wall placement, but does not enter the gap calculation.

### Contact Description

Each contact instance records the two interacting bodies, the contact point, the contact normal, and the signed gap `d` (penetration occurs when `d < -threshold`). The relative velocity at the contact point incorporates both the translational and rotational motion of each element.

When contact occurs (i.e., penetration detected), a force is generated at the contact point:

- Penetration depth: `delta = -d`.
- **Normal force:** spring-damper model:
  `f_n = max(0, k_eff * delta + c_eff * v_n)`
  where `k_eff` and `c_eff` are the effective spring and damping coefficients, and `v_n` is the normal component of relative velocity.
- **Tangential force:** damped slip, limited by friction (`mu` is negligible here, so friction effects are minimal).
- The force `F` acts equally and oppositely on the two elements: element 1 receives `+F`, element 2 receives `-F`, both applied at the contact point.

During each simulation time step, contact resolution proceeds as follows:
1. Build element geometry using the current rod states.
2. Optionally update the list of potentially colliding pairs (set by `steps_between_detection`, with `0` meaning every step).
3. Compute precise contact points on cached pairs.
4. Apply spring-damper contact forces according to the penetration and relative velocities.

### Initialization protocols

Supported: `lattice`, `random-cone`, `random-cylinder`, `random-cube`,
`random-cube-v2`. With `R = radial_span_ratio * base_length` and
`H = vertical_span_ratio * base_length`:

- **`random-cylinder`**: rod start uniform in a vertical cylinder (radius `R`,
  at least `base_radius` above floor); random azimuth and small out-of-plane tilt;
  resample until the far end is also inside the cylinder.
- **`random-cone`**: same, but the far end must lie inside a cone.
- **`random-cube`**: layered flat packing in a square footprint.
- **`random-cube-v2`**: full 3D random positions and orientations in a cube.
- **`lattice`**: structured two-layer packing with alternating directions.
