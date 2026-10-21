<div align='center'>
<h1> PyElastica-JAX (& MPI) </h1>
</div>

JAX integration to [`PyElastica`](https://github.com/GazzolaLab/PyElastica) framework for simulating assemblies of Cosserat Rods.
This framework extension supports more directly handling of the `block` concept.

The goal is to scale. If your problem is below an order of 100~ elements, this framework might not give any speed-up, but you are welcome to still explore the features.

> Currently under development. All features are experimental, and interface are subject to change without detailed discussion. Please leave issues if you have any suggestions or recommendations.
> Due to the style difference between `JAX` and `numba`, the implementation in original PyElastica package would be mostly not compatible. The style of setting up the simulation is kept same.

## Benchmark

### Single-node

![Snake throughput scaling](https://raw.githubusercontent.com/skim0119/PyElastica-JAX/assets/snake_throughput_scaling_combined.png)

### Multi-node, weak scaling

<img src="https://raw.githubusercontent.com/skim0119/PyElastica-JAX/assets/snake_mpi_throughput_scaling_combined.png" width="60%" alt="description">

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
import numpy as np
from tqdm import tqdm

import elastica as ea
import elastica_jax as eaj

jax.config.update("jax_enable_x64", True)

simulator = eaj.Simulator()  # <-- create the system collection. Unified API for all operations.
rod_block = eaj.configure_rod_block()
simulator.enable_block_supports(  # <-- assign the JAX block for CosseratRod
    ea.CosseratRod,
    rod_block,
)

# Append rods
n_rods = 2
rods = []
for rod_idx in range(n_rods):
    rod = ea.CosseratRod.straight_rod(
        n_elements=8,
        start=np.array([0.0, 0.05 * rod_idx, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=0.2,
        base_radius=0.005,
        density=1000.0,
        youngs_modulus=1.0e5,
    )
    rods.append(rod)
    simulator.append(rod)
    simulator.operate(rod).using(eaj.OneEndFixedJax)
    simulator.operate(rod).using(eaj.GravityForcesJax)
simulator.finalize()
```

The syntax is almost similar to `PyElastica`. To use position-verlet scheme, load `eaj.PositionVerletJAX`.
Unlike `PyElastica`, stepper does not provide single `step` method, instead it provides `integrate` method. This is due to the architecture design of `JAX` that is not same as `numba`: the amortized optimization of `jit` compilation yield a lot more efficient runtime with `fori_loop`, which is ineffective to run single-step at a time.

```py
timestepper = eaj.PositionVerletJAX()
time = 0.0
final_time = 0.01
dt = 0.0001
time = timestepper.integrate(
    simulator,
    time=time,
    final_time=final_time,
    dt=dt,
)
rod_block.from_device()
print(rods[0].position_collection[:, -1])  # Current tip position
```

### Choose your customization level

Start with `ea.CosseratRod` and the default `eaj.configure_rod_block()`.
Set geometry and linear elastic material parameters on the rod, register loads,
boundary conditions, and damping with `simulator.operate(rod).using(...)`, then
finalize and integrate. Rods in the default horizontal Block may have different
element counts. You do not need to handle packed indices for this workflow.

The host Rod supplies initial arrays and constructor data. After `finalize()`,
the Block owns the running state **and the runtime physics implementation**.
Overriding `compute_internal_forces_and_torques` on a host Rod does not transfer
that method to its JAX Block. A new constitutive law or rod model requires a
matching JAX Block implementation; see [custom rod physics](#custom-rod-physics).

| Your change | Extension point |
| --- | --- |
| Geometry, elastic moduli, or rest strains | Configure the host Rod before `finalize()` |
| Boundary conditions, forcing, or damping | `operate(rod)` with a `NoOpsJax` subclass |
| Rod-to-rod interactions | `pairwise_interaction(first, second)` |
| Batch the same rod-local operation | `operate_block(block)` with equal discretization |
| New constitutive law or rod state | A custom JAX Block with explicit packing and stepping behavior |

The advanced sections explain Block layout, synchronization, and device
placement. Keep the default setup until your model or measured performance
requires those details.

### How to collect data?

In `PyElastica`, data is collected by attaching `ea.Callback` to the system. In `PyElastica-JAX`, similar callback syntax is _not available_, as retrieving data from the block state is not supported.

> **_NOTE:_** This is key differnce between original PyElastica and PyElastica-JAX. The block rod (handled within the simulator during) is not always synchronized with the reference rods (created by user). This is to support multi-device and heterogeneous (GPU, CPU+GPU, etc.) execution more smoothly within JAX syntax pattern.
> In detail, callback in PyElastica interrupt the integrate loop and collect data for I/O. Most of the time, due to large number of simulation steps, branch like `step % step_skip == 0` is used to collect data intermittently. This could be achieved using `jax.pure_callback` (probably) but it defeats the point of using JIT. Hence, collection is taken cared outside, closer to user's control.

Here is the recommended way to collect data. The integration can be broken into chunks of an integer number of steps. Choose `steps_per_frame` to set the sampling interval; the final chunk may be shorter. Use `eaj.save` to write HDF5 frames. **Save/Load never call `from_device` / `to_device`**: a block Save reads device state; a rod Save reads host arrays as-is. Sync only when you intentionally want host rods updated.

```py
# Continue the simulation built above, starting at its returned time.
current_time = time
final_time = current_time + 0.01
dt = 0.0001
steps_per_frame = 25
n_steps = int(round((final_time - current_time) / dt))
assert np.isclose(current_time + n_steps * dt, final_time), (
    "The sampling duration must be an integer multiple of dt."
)
frame_steps = range(0, n_steps + steps_per_frame, steps_per_frame)
for frame_idx, step in enumerate(tqdm(frame_steps, desc="Saving simulation")):
    target_time = time + min(step, n_steps) * dt
    current_time = timestepper.integrate(
        simulator,
        time=current_time,
        final_time=target_time,
        dt=dt,
    )
    eaj.save(
        rod_block,
        f"results/frame_{frame_idx:06d}.h5",
        time=current_time,
        frame_idx=frame_idx,
        verbose=0,
    )

# Sync only when inspecting user-held host rods.
rod_block.from_device()
print(rods[0].position_collection[:, -1])
```

Schema levels (`verbose`): `0` geometry/pose, `1` adds rates and strains, `10` full state (block/simulator files at level 10 are Load-safe for resume). For large block/simulator Saves, pass `n_workers>1` for chunked parallel writes. With an MPI `comm`, ranks barrier around Save/Load and only rank 0 writes a shared path.

> This part is actively under development for more flexible and better API. Leave issues if you have any suggestions or recommendations.

### Defining Operation

In `PyElastica`, operations could be inserted using modules such as `ea.Forcing`, `ea.Damping`, `ea.Constraints`, etc. Depending on modules, user can add custom operation at the end of kinematic steps (`constrain_values`), end of dynamic steps (`constrain_rates`), or add loads before the dynamic steps (`synchronize`). This gives great flexibility and customizability to the simulation to define wide range of physics and behaviors for problems.

`PyElastica-JAX` follows the same concepts and philosophy, but with some simplification and changes according to `JAX`'s pure function pattern. This gives better optimization and amortized compilation for entire JIT timestepping loop.

Use `eaj.Simulator` for the system collection. An operation to a designated rod could be included using `operate` method.

```py
simulator = eaj.Simulator()
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

Use `operate(rod)` by default. Block operations are an advanced optimization or
an extension for physics that needs several rods' state together.

For `jax_per_rod_operate_*`, or a `NoOpsJax` reused through `operate_block`, all
rods in that Block must have equal element counts and the same open/ring
topology. Unsupported combinations fail during `finalize()`. With unequal rods,
register the same class separately on each rod instead:

```py
for rod in rods:
    simulator.operate(rod).using(eaj.OneEndFixedJax)
```

True `jax_block_operate_*` hooks receive packed state directly and do not have
this batching restriction; their implementation must handle the layout and
its ghost padding correctly.


`PyElastica-JAX` supports block-wise operation. This is partially tested on `PyElastica` and have shown great performance boost, but not fully included due to difficulties in customizing every operation to be block-wise and JIT compatible. This enables faster implementation-execution of large-scale (many rods) simulations, especially when operation needs to be applied to all rods at each step.

This operation module provides two styles of operation: __block__ and __per-rod__. The default memory block packs rods **horizontally** into contiguous arrays with ghost padding so spanwise kernels stay isolated per rod.

- __Block__ operation (`jax_block_operate_*`) receives the full authoritative
  Block state (packed or stacked) as-is. Use this for Block-native logic such
  as contact across rods or transforms that index packed ghosts.
- __Per-rod__ operation (`jax_per_rod_operate_*`) receives one Rod-local view.
  The Block batches across rods with ``map_rods`` (gather/scatter when packed;
  leading-rod axis when stacked). Shared identical callables use ``jax.vmap``.

> When using block-native spanwise equations, carefully treat ghost elements.
> Ghost padding isolates numerics within each rod, but does not keep those
> values zero. <TODO: add more details in documentation>

Register block operations with `operate_block` on `eaj.Simulator`:

```py
simulator = eaj.Simulator()
rod_block = eaj.configure_rod_block()
simulator.enable_block_supports(ea.CosseratRod, rod_block)

for _ in range(n_rods):
    simulator.append(ea.CosseratRod.straight_rod(...))

simulator.operate_block(rod_block).using(
    MyBlockOperation,
    ...  # Arguments for MyBlockOperation
)

simulator.finalize()
```

To define the operator, derive from `eaj.NoBlockOpJax` and override block-stage hooks. Similar to regular operation, following functions could be overridden:

- `jax_block_operate_synchronize` — Executed before the dynamic step.
- `jax_block_operate_constrain_values` — Executed after the kinematic step.
- `jax_block_operate_constrain_rates` — Executed after the dynamic step.
- `jax_per_rod_operate_synchronize` — Executed before the dynamic step.
- `jax_per_rod_operate_constrain_values` — Executed after the kinematic step.
- `jax_per_rod_operate_constrain_rates` — Executed after the dynamic step.

`jax_block_operate_*` functions take the full Block state dictionary and return
an updated dictionary. There is no Rod-local projection.

`jax_per_rod_operate_*` functions receive a single rod view and return the
updated view (or committed state). Prefer this style for rod-shaped logic that
should run once per rod; do not register that logic as `jax_block_operate_*`
expecting a secret gather/`vmap`.

> **Reusing single-rod operators on blocks:** Classes derived from ``eaj.NoOpsJax`` (the same ``jax_operate_*`` API used with ``operate(rod)``) can also be registered with ``operate_block(rod_block).using(...)``. This is intentional: one operator instance is created per rod in the block (so ``__init__`` can read that rod's ``_system``), and the existing rod-local kernels are batched through ``Block.map_rods`` like ``jax_per_rod_operate_*``. For example, ``eaj.OneEndFixedJax`` works with both ``operate(rod)`` and ``operate_block(rod_block)`` without duplicating the implementation under ``jax_per_rod_operate_*``.

Apply block-wide gravity and fix one end of each rod:

```py
class GravityBlockOp(eaj.NoBlockOpJax):
    """Add gravity to a horizontal or stacked Block's nodal forces."""

    def __init__(self, *, acc_gravity: jax.Array, **kwargs: object) -> None:
        self.acc_gravity = acc_gravity

    def jax_block_operate_synchronize(
        self, block_state: dict[str, jax.Array], time: eaj.JAXTime
    ) -> dict[str, jax.Array]:
        """Return Block state with gravitational nodal forces added."""
        updated = dict(block_state)
        updated["external_forces"] = (
            block_state["external_forces"]
            + self.acc_gravity[:, None] * block_state["mass"][..., None, :]
        )
        return updated


# Register these before simulator.finalize(), in place of per-rod operators.
simulator.operate_block(rod_block).using(
    GravityBlockOp,
    acc_gravity=jnp.array([0.0, -9.80665, 0.0]),
)
simulator.operate_block(rod_block).using(eaj.OneEndFixedJax)
```

### Rod-Rod Interaction

> Work-in-progress.

Register pair operators with ``pairwise_interaction`` on ``eaj.Simulator``.
Each operator receives rod-local views through ``jax_operation`` and may update
one or both rods:

```py
simulator = eaj.Simulator()


class Repell(eaj.NoRodRodBlockOpJax):
    def jax_operation(self, rod_one_view, rod_two_view, time):
        ... # Operation
        return rod_one_view, rod_two_view


simulator.pairwise_interaction(rod_one, rod_two).using(
    Repell,
    ...  # Arguments for Repell
)
```

Rods may live in horizontal or stacked blocks, in the same block or in
separate blocks on the **same device**. On a single device, ``PositionVerletJAX`` compiles one coupled
``fori_loop`` over all block states and applies rod-to-rod operators during the
global synchronize stage.

See the [Muscular Snake example](examples/MuscularSnake/) for a full case with a
body block, a muscle block, and eight cross-block surface joints.

Cross-device rod-to-rod coupling between independent blocks is not yet supported.

## Advanced Usage

### Custom rod physics

A host Rod subclass alone does not install new runtime physics. The default
Block evaluates the built-in linear Cosserat stress/couple laws from the packed
material arrays. Changing `youngs_modulus` or `shear_modulus` needs no new Block.
Changing the constitutive law itself requires implementing it in a JAX Block.
There is currently no separate constitutive-function registration API.

Use the following responsibilities when extending a Block:

- Pack initial arrays and model parameters during `finalize()`.
- Implement `jax_compute_internal_forces_and_torques(state, time)` as a pure
  JAX transform. Use the supplied state and time; do not read mutable host rod
  arrays inside the kernel or convert traced values to NumPy.
- Handle ghost and periodic entries for the chosen layout. A horizontal Block
  extension does not automatically extend the stacked implementation.
- If adding evolving state, include it in the Block's state transforms and
  explicitly implement its rod projection, synchronization, and Save/Load
  behavior. The default field lists do not discover custom arrays.

Here is a minimal extension adding axial viscous tension to the elastic law:
`N_viscous = c * dilatation_rate`, where `c` has units of N s. It uses the
horizontal Block's existing difference kernel to keep rods' loads isolated.
This example supports open rods with the standard state fields. It adds a
force contribution; `internal_stress` and `internal_couple` remain the elastic
fields. A new field or periodic constitutive law needs its own packing and
boundary treatment.

```py
from elastica_jax.memory_block.memory_block_rod_jax import (
    _jax_two_point_difference_for_block_structure,
)


class AxialViscousBlock(eaj._CosseratRodMemoryBlock):
    """Add axial viscous tension to the default elastic open-rod model."""

    AXIAL_DAMPING = 0.1  # N s

    def jax_compute_internal_forces_and_torques(
        self, state: dict[str, jax.Array], time: eaj.JAXTime
    ) -> dict[str, jax.Array]:
        """Return elastic and axial viscous loads for the supplied state.

        Parameters
        ----------
        state
            Packed standard Cosserat collections on the device.
        time
            Current integration stage time.

        Returns
        -------
        dict[str, jax.Array]
            Updated geometry, strains, and internal loads.

        Raises
        ------
        AssertionError
            If the Block contains a ring rod.
        """
        assert not self.ring_rod_flag, "AxialViscousBlock supports only open rods."
        updated = super().jax_compute_internal_forces_and_torques(state, time)
        tension = self.AXIAL_DAMPING * updated["dilatation_rate"]
        viscous_force = updated["tangents"] * tension[None, :]
        updated["internal_forces"] = (
            updated["internal_forces"]
            + _jax_two_point_difference_for_block_structure(
                viscous_force, self._device_metadata["ghost_elems_idx"]
            )
        )
        # Tension follows the tangent, so it adds no material-frame torque.
        return updated


# Use this Block in place of configure_rod_block() in the basic tutorial.
rod_block = eaj.configure_rod_block(inner_block_cls=AxialViscousBlock)
simulator.enable_block_supports(ea.CosseratRod, rod_block)
```

Custom HDF5 serializers are supported for standalone Blocks through serial
`save(block, ..., n_workers=1)` and `load(block, ...)`. Simulator Save/Load and
parallel Save currently use the standard Block schema and reject overridden
serializers before writing their payload. Save custom Blocks individually until
composite and parallel I/O support their serialization contract.

The `_`-prefixed class and kernel are implementation interfaces and may change.
This example intentionally exposes that dependency. For a larger extension,
see the [muscle rod](examples/MuscleRod/muscle_rod.py) and its
[matching JAX Block](examples/MuscleRod/memory_block_muscle_rods_jax.py), which
show the separate host-model and runtime implementations.

> The following advanced usage methods may require special care. They are not guaranteed to be compatible with each other unless explicitly specified, and should generally be used independently for correct behavior.
> Scaling is not yet guaranteed to be optimal, and may depends on hardwares.

### GPU Execution

Pass an explicit backend to `configure_rod_block` so rod state is allocated on that
device before `finalize()`:

```py
rod_block = eaj.configure_rod_block(device="cuda")
simulator.enable_block_supports(ea.CosseratRod, rod_block)
...
simulator.finalize()
```

### Vertical (stacked) rod block

By default, `configure_rod_block` packs rods **horizontally**: arrays are
concatenated along the spatial axis with ghost separators, e.g. position
``(3, N_total)``.

For __equal-sized__ rods, you can instead pack rods **vertically** by
stacking on a leading batch axis, sized ``(N_rods, 3, N_elements)``.

| field kind | shape |
| --- | --- |
| vector | ``(n_rods, 3, N)`` |
| tensor | ``(n_rods, 3, 3, N)`` |
| scalar | ``(n_rods, N)`` |

where ``N`` is ``n_nodes``, ``n_elems``, or ``n_voronoi`` depending on the
variable. Timestep kernels run under ``jax.vmap`` over the rod axis.
``jax_per_rod_operate_*`` operators are batched the same way.
This gives greater parallelism in GPU or multi-device.

```py
rod_block = eaj.configure_rod_block(
    device="cpu",
    inner_block_cls=eaj._CosseratRodVerticalMemoryBlock,  # <--
)
simulator.enable_block_supports(ea.CosseratRod, rod_block)
```

Constraints:

- all rods in the block must share the same ``n_elems`` (otherwise finalize
  raises an assertion)
- ring rods are not supported

Author rod-shaped logic with `jax_per_rod_operate_*` (one rod view). Author
Block-native logic with `jax_block_operate_*` on the full stacked state. Do not
special-case the stacked layout in user operators.
### Multi-device Block Execution (User control)

> Work-in-progress
> Not exactly decided to keep both mpi and jax.distributed support. Maybe they are complementary?

Use separate blocks when each rod group is independent. Each block is assigned to
one device and compiled as its own JIT `fori_loop` rollout:

```py
devices = eaj.resolve_backend_devices("cuda")
block_0 = eaj.configure_rod_block(device=devices[0])
block_1 = eaj.configure_rod_block(device=devices[1])

# Distinct rod types select which configured block owns each rod.
simulator.enable_block_supports(RodTypeOnGPU0, block_0)
simulator.enable_block_supports(RodTypeOnGPU1, block_1)
simulator.operate_block(block_0).using(MyBlockOp)
simulator.operate_block(block_1).using(MyBlockOp)
```

See the [explicit multi-block tutorial](tutorial/01-block-multi-device.py) for a
complete example.

### Multi-device Block Execution (Distributed shard)

> TODO: Old implementation is removed due to new vertical block design. With
> jax.distributed modules, this could enable cross-node execution easily.
> Not yet sure how to handle operations

### Multi-core Multi-host CPU Execution

> Work-in-progress

`PyElastica-JAX` includes `mpi` block layout support, to ease the simulation setup in `mpi` manner.

> Implementation of cross-rank operators is possible with rod-rod operation, but front API is not yet designed. Leave issues if you have any suggestions or recommendations.

Following example outlines a cantilever simulation under uniform gravity using the
built-in `GravityAnalyticalDamperJax` operator across MPI ranks:

```py
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Expose local CPU cores to JAX on each MPI rank (set before importing jax).
n_local_devices = 1
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={n_local_devices}"
)

import numpy as np
import jax

import elastica as ea
import elastica_jax as eaj

jax.config.update("jax_enable_x64", True)


simulator = eaj.Simulator()
rod_block = eaj.configure_rod_block_mpi(comm=comm)
simulator.enable_block_supports(ea.CosseratRod, rod_block)

# Each MPI rank owns a disjoint rod subset.
n_rods_total = 64
for rod_index in range(n_rods_total):
    if rod_block.owns_rod(rod_index):  # REVIEW: This pattern seems not ideal
        rod = ea.CosseratRod.straight_rod(
            n_elements=20,
            start=np.array([0.1 * rod_index, 0.0, 0.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            base_length=0.35,
            base_radius=0.01,
            density=1_000.0,
            youngs_modulus=5.0e6,
        )
        simulator.append(rod)

dt = 1.0e-4
simulator.operate_block(rod_block).using(eaj.OneEndFixedJax)
simulator.operate_block(rod_block).using(
    eaj.GravityAnalyticalDamperJax,
    time_step=dt,
    uniform_damping_constant=0.5,
)
simulator.finalize()

stepper = eaj.PositionVerletJAX()
stepper.integrate(
    simulator,
    time=0.0,
    final_time=10.0,
    dt=dt,
)
comm.Barrier()
```

Launch with `mpirun -n <world_size> python script.py`. Cross-rank block operators
that require halo exchange will use `mpi4jax`; gravity and one-end-fixed constraints
are local to each rank's block.

## Terminology

This repository expand the usage of `block` concept directly, along with `JAX`'s concept of mapping memory and function.

- `memory_block` is a block that is a collection of rods (systems).
- `vertical_block` / stacked block packs equal-length rods on a leading batch
  axis ``(n_rods, ...)`` and batches kernels with ``vmap`` (no ghost separators).
- `mpi_block` is a rank-local memory block: rods are partitioned across MPI ranks.

## Features that are extended from PyElastica

- `eaj.Simulator` includes the full registration surface, unlike previous `PyElastica` collections that requires to compose mixins or classic modules onto a blank collection.
- `configure_rod_block(..., inner_block_cls=eaj._CosseratRodVerticalMemoryBlock)`:
  use the stacked-axis rod block for equal-length straight rods.
