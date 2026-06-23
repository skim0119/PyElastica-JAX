# Active Matter GPU JAX Examples

Two independent entry points reproduce the cases in
[`CASE_DESCRIPTION.md`](CASE_DESCRIPTION.md) as device-owned JAX rollouts:

- `run_snake_pit.py` — enclosed random packing with pit walls
- `run_snake_on_plane.py` — four snakes on a floor with gravity

Shared utilities (`contact_kernels.py`, `simulation_runtime.py`, `frame_io.py`)
provide contact math, device selection, and HDF5 frame I/O only. Neither case
imports the other.

Run the snake pit on CUDA:

```bash
uv run --no-sync python examples/ActiveMatter/run_snake_pit.py --backend cuda
```

Run snakes on a plane:

```bash
uv run --no-sync python examples/ActiveMatter/run_snake_on_plane.py --backend cuda
```

`--backend auto` selects CUDA, MPS, or CPU in that order. Use `--final-time`,
`--time-step`, `--n-elements`, and `--n-snakes` for shorter experiments or
scaling studies.

## Frame output and rendering

Simulations write chunked HDF5 frames into `output/` (or `output_<run-name>/` when
`--run-name` is set):

- `metadata.h5` — case parameters, fps, walls
- `frame_<idx>.h5` — rod positions `(n_snakes, n_nodes, 3)`

Render PNGs to `<output>/png/` and assemble `output.mp4` (or `output_<run-name>.mp4`).
FPS and other timing details are read from `metadata.h5`.

```bash
uv run --no-sync python examples/ActiveMatter/run_snake_on_plane.py --backend cpu
uv run --no-sync python examples/ActiveMatter/render.py
```

Named run:

```bash
uv run --no-sync python examples/ActiveMatter/run_snake_pit.py --run-name pit01
uv run --no-sync python examples/ActiveMatter/render.py --run-name pit01
```

MPI rendering:

```bash
mpiexec -n 4 uv run --no-sync python examples/ActiveMatter/render.py --run-name pit01
```
