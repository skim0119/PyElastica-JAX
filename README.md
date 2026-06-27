<div align='center'>
<h1> PyElastica-JAX </h1>
</div>

JAX integration to [`PyElastica`](https://github.com/GazzolaLab/PyElastica) framework for simulating assemblies of Cosserat Rods.
This framework extension supports more directly handling of the `block` concept.

> Currently under development. All features are experimental, and interface are subject to change without detailed discussion. Please leave issues if you have any suggestions or recommendations.
> Due to the style difference between `JAX` and `numba`, the implementation in original PyElastica package would be mostly not compatible. The style of setting up the simulation is kept same.

## Before you start

`JAX`'s default floating point precision is `float32`, but the Cosserat rod numerics is not very stable with `float32`. User should enable `float64` for more stable numerics. It is not decided yet whether to make this the default behavior within `PyElastica-JAX`.

```py
jax.config.update('jax_enable_x64', True)
```
or
```bash
JAX_ENABLE_X64=1 python ...
```

## Basic Tutorial

In `PyElastica`, a system, like `CosseratRod`, is collected and stored in a blocked memory space. This operation occurs during `finalize()`. Using `simulator.enable_block_supports(...)`, the type of block can be specified and configured.

```py
import jax
import jax.numpy as jnp
import elastica as ea
import elastica_jax as eaj

class JAXSimulator(
    ea.BaseSystemCollection,  # From PyElastica
    eaj.JAXOps
):
    pass

simulator = JAXSimulator()
rod_block = eaj.configure_rod_block()
simulator.enable_block_supports(  # <-- assign the JAX block for CosseratRod
    ea.CosseratRod,
    rod_block,
)

# Append rods
n_rods = 10
rods = []  # Serves as a rod view
for _ in range(n_rods):
    rod = ea.CosseratRod.straight_rod(...)
    rods.append(rod)
    simulator.append(rod)
simulator.finalize()
```

The syntax is almost similar to `PyElastica`. To use position-verlet scheme, load `eaj.PositionVerletJAX`.
Unlike `PyElastica`, stepper does not provide single `step` method, instead it provides `integrate` method. This is due to the architecture design of `JAX` that is not same as `numba`: the amortized optimization of `jit` compilation yield a lot more efficient runtime with `fori_loop`, which is ineffective to run single-step at a time.

```py
timestepper = eaj.PositionVerletJAX()
time = 0.0
final_time = 10.0
dt = 0.001
time = timestepper.integrate(
    simulator,
    time=time,
    final_time=final_time,
    dt=dt,
)
```

### How to collect data?

In `PyElastica`, data is collected by attaching `ea.Callback` to the system. In `PyElastica-JAX`, similar callback syntax is _not available_, as retrieving data from the block state is not supported.

> **_NOTE:_** This is key differnce between original PyElastica and PyElastica-JAX. The block rod (handled within the simulator during) is not always synchronized with the reference rods (created by user). This is to support multi-device and heterogeneous (GPU, CPU+GPU, etc.) execution more smoothly within JAX syntax pattern.
> In detail, callback in PyElastica interrupt the integrate loop and collect data for I/O. Most of the time, due to large number of simulation steps, branch like `step % step_skip == 0` is used to collect data intermittently. This could be achieved using `jax.pure_callback` (probably) but it defeats the point of using JIT. Hence, collection is taken cared outside, closer to user's control.

Here is the recommended way to collect data. The integration could be broken into chunks, at desirable frame rate.
User can fetch the data from the block state, and copy. For example, the following snippet could be used to collect the position to create animation.

```py
current_time = 0.0
final_time = 10.0
dt = 0.001
fps = 25.0
frame_dt = 1.0 / fps
steps_per_frame = int(round(frame_dt / dt))
n_frames = int(round(final_time / frame_dt)) + 1
for frame_idx in tqdm(range(n_frames)):
    jax.block_until_ready(rod_block)

    # Not recommended: access block memory directly. It may
    # includes ghost elements that may not be intuitive unless
    # user is aware of the block implementation details.
    # Although it could be useful for more advanced use cases.
    # shape would be (3, n_block_length)
    position_collection = rod_block.position_collection.copy()

    # Recommended: sync rod views from block to reference rods.
    rod_block.from_device()
    positions = []
    for rod_view in rods:
        positions.append(rod_view.position_collection.copy())
    # shape would be (n_rods, 3, n_nodes)
    position_collection = np.concatenate(positions, axis=0)

    # I/O
    save(
        f"results/frame_{frame_idx:06d}.h5",
        position_collection,
        time=current_time,
        frame_idx=frame_idx,
        n_workers=4,  # For very large case, it is recommended to parallelize I/O
    )

    # Integrate
    chunk_final_time = current_time + frame_dt
    stepper.integrate(
        simulator,
        time=current_time,
        final_time=chunk_final_time,
        dt=dt,
    )
    current_time = chunk_final_time
```

> This part is actively under development for more flexible and better API. Leave issues if you have any suggestions or recommendations.

### Defining Operation

In `PyElastica`, operations could be inserted using modules such as `ea.Forcing`, `ea.Damping`, `ea.Constraints`, etc. Depending on modules, user can add custom operation at the end of kinematic steps (`constrain_values`), end of dynamic steps (`constrain_rates`), or add loads before the dynamic steps (`synchronize`). This gives great flexibility and customizability to the simulation to define wide range of physics and behaviors for problems.

`PyElastica-JAX` follows the same concepts and philosophy, but with some simplification and changes according to `JAX`'s pure function pattern. This gives better optimization and amortized compilation for entire JIT timestepping loop.

In `PyElastica-JAX`, adding rod-wise operation is done with mixin module `eaj.JAXOps` into simulator class. This gives ability to run `operate` method to add operation to a specific system.

```py
class JAXSimulator(ea.BaseSystemCollection, eaj.JAXOps):
    pass

simulator = JAXSimulator()
rod_block = eaj.configure_rod_block()
simulator.enable_block_supports(ea.CosseratRod, rod_block)

rod = ea.CosseratRod.straight_rod(...)
simulator.append(rod)

simulator.operate(rod).using(
    MyOperation,
    ...  # Arguments for MyOperation
)

simulator.finalize()
```

Similar to `PyElastica`, module has associated template class `eaj.NoOpsJax` that user can derive from to implement the operation. Deriving this class is compulsory to be properly registered into the operation group.

By overriding following hooks, user can implement the oepration.

- `jax_operate_synchronize` — Executed before the dynamic step.
- `jax_operate_constrain_values` — Executed after the kinematic step.
- `jax_operate_constrain_rates` — Executed after the dynamic step.

> Similar to `PyElastica`, this flexibility gives user greater ability to tweak and customize the simulation; hence, it is user's responsiblity to ensure the consistency of the modeling and physics.

Each hook receives a `rod_view` (JAX arrays for one rod inside the block) and returns the updated view. This function should be _pure_ JAX function. See [`src/elastica_jax/operations.py`](https://github.com/skim0119/PyElastica-JAX/blob/main/src/elastica_jax/operations.py) for built-in operations.

Here is the example script of implementing gravity on the rod with one-end-fixed.

```py
type Vector = jax.Array
class CantileverOperation(eaj.NoOpsJax):
    """Apply uniform gravitational acceleration to nodal masses."""

    def __init__(
        self,
        acc_gravity: Vector,
        *,
        _system=None,
    ) -> None:
        self.acc_gravity = np.asarray(acc_gravity, dtype=np.float64)
        self.fixed_position_collection = np.asarray(
            _system.position_collection[..., 0].copy()
        )
        self.fixed_directors_collection = np.asarray(
            _system.director_collection[..., 0].copy()
        )

    def jax_operate_constrain_values(self, rod_view, time):
        rod_view.position_collection = rod_view.position_collection.at[:, 0].set(
            self.fixed_position_collection
        )
        rod_view.director_collection = rod_view.director_collection.at[:, :, 0].set(
            self.fixed_directors_collection
        )
        return rod_view

    def jax_operate_constrain_rates(self, rod_view, time):
        rod_view.velocity_collection = rod_view.velocity_collection.at[:, 0].set(
            jnp.zeros(3, dtype=rod_view.velocity_collection.dtype)
        )
        rod_view.omega_collection = rod_view.omega_collection.at[:, 0].set(
            jnp.zeros(3, dtype=rod_view.omega_collection.dtype)
        )
        return rod_view

    def jax_operate_synchronize(self, rod_view, time):
        rod_view.external_forces = (
            rod_view.external_forces
            + self.acc_gravity[:, None] * rod_view.mass[None, :]
        )
        return rod_view

# Register on a specific rod; `_system` is injected by the framework at finalize().
simulator.operate(rod).using(
    CantileverOperation,
    acc_gravity=np.array([0.0, 0.0, -9.80665]),
)

simulator.finalize()
```

### Defining Block Operation

`PyElastica-JAX` supports block-wise operation. This is partially tested on `PyElastica` have shown great performance boost, but not fully included due to difficulties in customizing every operation to be block-wise and JIT compatible. This enables faster implementation-execution of large-scale (many rods) simulations, especially when operation needs to be applied to all rods at each step.

This operation module provides two style of operation: __block__ and __per-rod__.
Block-rod is essentially a single-rod view of all rods in contiguous memory space, with ghost elements/memory padding to safely handle spanwise operations (differential and quadratures).
__Block__ operation treats entire rod at once. It is useful to define all-element operations, such as gravity, dissipation, field forcing, etc. __Per-rod__ operation treats each rod at once. Underneat, it uses `jax.vmap` to batch the operation across all rods.

> When using block operation with spanwise equation, carefully treat the ghost ghost elements and their behavior. Ghost padding is only to separate the numerics to be isolated within the rod, but does not have any safety to keep those values zeros. <TODO: add more details in documentation>

To use block operation, add the `eaj.JAXOpsBlock` mixin and register with `operate_block`:

```py
class JAXSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    pass

simulator = JAXSimulator()
rod_block = eaj.configure_rod_block()
simulator.enable_block_supports(ea.CosseratRod, rod_block)

for _ in range(n_rods):
    simulator.append(ea.CosseratRod.straight_rod(...))

simulator.operate_block(eaj.MemoryBlockCosseratRodJax).using(
    MyBlockOperation,
    ...  # Arguments for MyBlockOperation
)

simulator.finalize()
```

To define the operator, derive from `eaj.NoBlockOpJax` and override block-stage hooks. Similar to regular oepration, following functions could be overridden:

- `jax_block_operate_synchronize` — Executed before the dynamic step.
- `jax_block_operate_constrain_values` — Executed after the kinematic step.
- `jax_block_operate_constrain_rates` — Executed after the dynamic step.
- `jax_per_rod_operate_synchronize` — Executed before the dynamic step.
- `jax_per_rod_operate_constrain_values` — Executed after the kinematic step.
- `jax_per_rod_operate_constrain_rates` — Executed after the dynamic step.

`jax_block_operate_*` functions receive the entire block state, and return the updated state.
`jax_per_rod_operate_*` functions receive the single rod view and return the updated state.

Block-wide gravity on all rods and one-end-fixed on each rods:

```py
type Vector = jax.Array
type Director = jax.Array

class GravityBlockOp(eaj.NoBlockOpJax):
    def __init__(self, acc_gravity: Vector, tip_position: Vector, tip_director: Director) -> None:
        self.acc_gravity = acc_gravity

    def jax_per_rod_operate_constrain_values(self, rod_view, time):
        rod_view.position_collection = rod_view.position_collection.at[:, 0].set(
            self.fixed_position_collection
        )
        rod_view.director_collection = rod_view.director_collection.at[:, :, 0].set(
            self.fixed_directors_collection
        )
        return rod_view

    def jax_per_rod_operate_constrain_rates(self, rod_view, time):
        rod_view.velocity_collection = rod_view.velocity_collection.at[:, 0].set(
            jnp.zeros(3, dtype=rod_view.velocity_collection.dtype)
        )
        rod_view.omega_collection = rod_view.omega_collection.at[:, 0].set(
            jnp.zeros(3, dtype=rod_view.omega_collection.dtype)
        )
        return rod_view

    def jax_block_operate_synchronize(self, block_state, time):
        block_state.external_forces = block_state.external_forces + self.acc_gravity[:, None] * block_state.mass[None, :]
        return updated


simulator.operate_block(eaj.MemoryBlockCosseratRodJax).using(
    GravityBlockOp,
    acc_gravity=jnp.array([0.0, 0.0, -9.80665]),
    tip_position=jnp.array([0.0, 0.0, 0.0]),
    tip_director=jnp.eye(3),
)
```

## Advanced Usage

### GPU Execution

Pass an explicit backend to `configure_rod_block` so rod state is allocated on that
device before `finalize()`:

```py
rod_block = eaj.configure_rod_block(device="cuda", device_dtype=np.float64)
simulator.enable_block_supports(ea.CosseratRod, rod_block)
...
simulator.finalize()
```

### Sharded Block — multi-device execution

For many rods spread across multiple GPUs (or TPUs), configure a sharded block with an execution mesh:

```py
devices = jax.devices("cuda")  # list of multiple devices
rod_block = eaj.configure_rod_block_sharded(
    devices=devices,
    device_dtype=np.float64,
)
simulator.enable_block_supports(ea.CosseratRod, rod_block)
```

To emulate multi-device scenario on CPU, set the number of devices to be used for execution.

```py
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
```

### Multi-core Multi-host CPU Execution

TODO: `mpi4jax` integration for multi-core MPI execution.

## Terminology

This repository expand the usage of `block` concept directly, along with `JAX`'s concept of mapping memory and function.

- `memory_block` is a block that is a collection of rods (systems).
- `sharded_block` is a block that is sharded into multiple blocks.

## Features that are extended from PyElastica

- `configure_rod_block_sharded(..., block_checkpoint=path)`: pass the checkpoint path when configuring the block. During `finalize()` → `construct_memory_block_structures` → block `__init__`, an existing checkpoint skips packing rod data into the block and loads saved state instead; a missing file triggers a save after block construction.
