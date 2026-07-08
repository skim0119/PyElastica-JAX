# Active Matter (Snake Pit)

GPU JAX reproduction of the enclosed snake-pit active-matter case: randomly
packed active snakes inside a floor-and-four-wall pit, driven by a traveling-wave
internal torque and resolved with capsule-capsule rod-rod contact and
capsule-half-space wall contact. The reference physics are specified in
[`CASE_DESCRIPTION.md`](CASE_DESCRIPTION.md).

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

`--smoke` downscales the run while still exercising every operator. Use `-N/--n-snakes`,
`--n-elements`, and `-T/--final-time` for scaling studies. `--mesh auto` uses one
shard per local device; `--mesh unified` keeps a single shard.

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
