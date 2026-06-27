# JAX Guide

This page documents how the JAX path differs from the CPU path in PyElastica and
covers implementation details that are not repeated in the [README](../README.md)
tutorial.

> For setup, time stepping, operator registration, and data collection examples, start with the README.

## Package Layout

Import PyElastica rod/system types from `elastica` and JAX extensions from
`elastica_jax`:

```python
import jax
jax.config.update("jax_enable_x64", True)

import elastica as ea
import elastica_jax as eaj  # Probably safe to load after jax config update.
```

JAX-owned pieces live in `elastica_jax`, including:

- `configure_rod_block` / `configure_rod_block_sharded` — block factories for `enable_block_supports`
- `PositionVerletJAX` — JIT-compiled Position Verlet integrator
- `JAXOps` / `JAXOpsBlock` — operator registration mixins
- `NoOpsJax` / `NoBlockOpJax` — operator base templates

## Execution Model

The intended workflow is:

1. Create rods with `ea.CosseratRod` as usual.
2. Build a simulator from `ea.BaseSystemCollection` plus the JAX mixins you need.
3. Configure the block with `configure_rod_block`.
4. Register operators before `finalize()`:
   - rod-local: `simulator.operate(rod).using(OpClass, ...)`
   - block-wide: `simulator.operate_block(ea.CosseratRod).using(OpClass, ...)`
5. Call `finalize()`, which packs rods into a block and lowers registered operators.
6. Run `eaj.PositionVerletJAX().integrate(...)`.

Unlike the CPU stepper, `PositionVerletJAX` exposes `integrate(...)` rather than a
single-step API. The rollout is compiled as one JAX loop, which amortizes JIT cost
across many steps.

## Block Configuration and Access

PyElastica hides block construction behind `finalize()`. PyElastica-JAX keeps that
entry point, but also lets you configure the block up front and read the built block
back without scanning `final_systems()`.

Configure with ``configure_rod_block`` or ``configure_rod_block_sharded``, pass the
returned block instance to ``enable_block_supports``, and use that same instance after
``finalize()``:

```python
class JAXSimulator(ea.BaseSystemCollection):
    pass

simulator = JAXSimulator()

rod_block = eaj.configure_rod_block()
simulator.enable_block_supports(ea.CosseratRod, rod_block)

for _ in range(n_rods):
    simulator.append(ea.CosseratRod.straight_rod(...))

simulator.finalize()
# rod_block is now the built block
assert rod_block.n_rods == n_rods
```

For sharded or checkpointed runs:

```python
execution_mesh = eaj.ExecutionMesh.from_devices(devices, n_rods=n_rods)
rod_block = eaj.configure_rod_block_sharded(
    mesh=execution_mesh,
    block_checkpoint=checkpoint_path,
)
simulator.enable_block_supports(ea.CosseratRod, rod_block)
simulator.finalize()
```


Once the block is built, it owns contiguous device memory for rod state. Original rod
values are no longer synchronized with that memory automatically.

> **Note:** This differs from PyElastica behavior, where block memory stays aliased to rod objects after `finalize()`.

During rollout, operators receive a `rod_view` into block memory rather than the
original rod object.

### Fetch data

To read the data back, use `block.from_device(...)` to sync the data back to the original
rod objects. Alternatively, you can use `block.to_device(...)` to load the data to the block.
(`.to_device(...)` is useful for resetting the block state to the initial values.)

Parameter could be the desire rod, iterable (tuple or list) of rods, or 'all'. By default, `all` is used: synchronize values between block and all related rods.
You can pass the keyword argument ``variables=Iterable[str]`` to synchronize specific fields only.

> If variable does not exist in the block, it will raise KeyError.

```python

# Initialize rod, block, and simulator
...

print(rod.position_collection)  # Original initialized values.
simulator.append(rod)
simulator.finalize()

timestepper = eaj.PositionVerletJAX()
timestepper.integrate(simulator, time=0.0, final_time=1.0, dt=0.001)

print(rod.position_collection)  # Before sync. Initialized values.
block.from_device(rod, variables=("position_collection",))
print(rod.position_collection)  # After sync. Updated values.
np.savez("rod_positions.npz", position_collection=rod.position_collection)
```

Example tutorial script is in [tutorial/00-block-device-communication.py](../tutorial/00-block-device-communication.py).

## Differences From The CPU Path

The CPU path is based on mutable live objects. Existing modules such as forcing,
damping, and constraints typically receive a `system` object and mutate arrays like
`system.position_collection` and `system.external_forces` in place.

The JAX path does not work that way. JAX requires:

- explicit state passed through the time-stepping loop
- pure staged transforms
- no hidden Python-side mutation during traced execution

Because of that, the JAX path does **not** directly reuse existing host module
implementations at runtime.

### Block State vs User Rod References

After `finalize()`, simulation state lives in the block created by the simulator. The
original rod objects you appended are not automatically kept in sync with device
state. This supports multi-device and heterogeneous execution without forcing host
readback every step.

Read block data explicitly when needed — for example through block attributes or
`block.from_device(update_rods=True)` — and treat chunked integration plus external
I/O as the recommended collection pattern. See the README data-collection section.

## Limitations

### Not All Mixins Are Supported

The JAX path is structured around:

- `ea.BaseSystemCollection`
- `eaj.JAXOps`
- `eaj.JAXOpsBlock`

Existing CPU mixins such as `Forcing`, `Damping`, `Constraints`, `Connections`,
`Contact`, and `CallBacks` are not automatically reusable inside pure JAX rollout.

This is a deliberate restriction of the current implementation, not a bug.

### Load Classes Must Be Re-implemented For JAX

Host-side load classes remain valid for CPU simulations, but they are not lowered into
JAX automatically. To participate in device rollout, implement a JAX operator by
subclassing `eaj.NoOpsJax` (rod-local) or `eaj.NoBlockOpJax` (block-level).

Built-in examples live in [`src/elastica_jax/operations.py`](../src/elastica_jax/operations.py).

## The Rod View Contract

JAX operators do not receive the original rod object during rollout. At `finalize()`,
each operator is lowered against a rod-local view into packed block state:

```python
rod_view.position_collection
rod_view.velocity_collection
rod_view.external_forces
rod_view.mass
```

Assign whole fields back on the view and return it:

```python
rod_view.external_forces = new_external_forces
return rod_view
```

:::{important}
Prefer JAX-style functional updates over NumPy-style in-place mutation of indexed
subviews:

```python
forces = rod_view.external_forces
forces = forces.at[..., -1].add(tip_force)
rod_view.external_forces = forces
```
:::

## Mixed Rod / Rigid-Body Operators

For coupled rod and rigid-body systems, add `eaj.JAXOpsRodRigidBody` and register with:

```python
simulator.using_on(rod, sphere).operate(MyCouplingOp, ...)
```

This mixin follows the same staged hook model as `JAXOps`, but operators receive both
rod and rigid-body views.

## Advanced Usage

### Sharded Blocks

For multi-device execution, configure the block factory with an execution mesh:

```python
execution_mesh = eaj.ExecutionMesh.from_devices(devices, n_rods=n_rods)
rod_block = eaj.configure_rod_block_sharded(
    mesh=execution_mesh,
    device_dtype=np.float64,
    block_checkpoint=checkpoint_path,
)
simulator.enable_block_supports(ea.CosseratRod, rod_block)
simulator.finalize()
```

On multi-core CPU, you can expose multiple host devices with:

```python
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
```

### Block Checkpoints

Pass `block_checkpoint=path` to `configure_rod_block_sharded(...)`.
During `finalize()`, an existing checkpoint skips repacking rod data and loads saved
state instead; a missing file triggers a save after block construction.

## Terminology

- `memory_block` — one block collecting many rods in contiguous device memory.
- `sharded_block` — a logical block split across multiple device-local blocks.

## Mental Model

Treat the JAX path as a parallel execution path with its own operator classes, not as
drop-in acceleration for every existing host-side module.

| Path | Use |
| --- | --- |
| CPU | existing mixins and host load classes |
| JAX rod-local | `JAXOps` + `NoOpsJax` |
| JAX many-rod / field ops | `JAXOpsBlock` + `NoBlockOpJax` |

This keeps the PyElastica setup style intact while allowing pure device-side rollout
where supported.
