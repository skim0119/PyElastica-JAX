"""Tutorial: assign rods to separate JAX devices with distinct memory blocks.

A single ``configure_rod_block`` keeps every rod on one device. To place rods on
different devices explicitly, create one configured block per device and register
each with ``enable_block_supports`` using distinct rod types:

    rod_block_1 = eaj.configure_rod_block(device=jax.devices()[0], ...)
    rod_block_2 = eaj.configure_rod_block(device=jax.devices()[1], ...)
    simulator.enable_block_supports(cr_type_1, rod_block_1)
    simulator.enable_block_supports(cr_type_2, rod_block_2)

The blocks are independent execution units. ``PositionVerletJAX`` compiles one
``jax.lax.fori_loop`` rollout per block and launches each rollout on the device
that owns that block. Operators that couple different blocks cannot use this
path and are rejected instead of falling back to a Python timestep loop.

This script emulates multiple CPU devices with ``XLA_FLAGS`` so it runs
without CUDA hardware. On a multi-GPU machine, use
``eaj.resolve_backend_devices("cuda")`` instead of ``jax.devices("cpu")``.

For rods split across devices inside one logical block, see
``02-block-multi-device-sharded.py`` and the ``gpu2x_sharded`` benchmark backend.
"""

from __future__ import annotations

import os
import time

N_DEVICES = 2
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_DEVICES}"

import numpy as np  # noqa: E402

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


class CantileverSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    pass


def main(
    n_rods: int = N_DEVICES,
    final_time: float = 100.0,
    time_step: float = 1.0e-4,
    show: bool = False,
) -> None:
    simulator = CantileverSimulator()
    rod_block_1 = eaj.configure_rod_block(
        device=jax.devices()[0],
        device_dtype=np.float64,
    )
    rod_block_2 = eaj.configure_rod_block(
        device=jax.devices()[1],
        device_dtype=np.float64,
    )

    # FIXME: This is hack to create same Cosserat rod but make it register differently.
    # Might be useful to have easier way to separate same rods on different block.
    skip_attrs = {
        "__dict__",
        "__weakref__",
        "__module__",
        "__annotations__",
        "__doc__",
        "__qualname__",
    }
    cr_type_1 = type(
        "CR_ON_DEVICE_1",
        ea.CosseratRod.__bases__,
        {k: v for k, v in ea.CosseratRod.__dict__.items() if k not in skip_attrs},
    )
    cr_type_2 = type(
        "CR_ON_DEVICE_2",
        ea.CosseratRod.__bases__,
        {k: v for k, v in ea.CosseratRod.__dict__.items() if k not in skip_attrs},
    )
    simulator.enable_block_supports(cr_type_1, rod_block_1)
    simulator.enable_block_supports(cr_type_2, rod_block_2)

    initial_positions: list[np.ndarray] = []
    rods = []
    for rod_index in range(n_rods):
        # Just to give offset in the visualization:
        rod_type = cr_type_1 if rod_index % 2 == 0 else cr_type_2
        start = np.array([0.1 * rod_index, 0.1 * rod_index, 0.0], dtype=np.float64)
        rod = rod_type.straight_rod(
            n_elements=20,
            start=start,
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            base_length=0.35,
            base_radius=0.01,
            density=1_000.0,
            youngs_modulus=5.0e6,
        )
        initial_positions.append(rod.position_collection.copy())
        rods.append(rod)
        simulator.append(rod)

    simulator.operate_block(rod_block_1).using(eaj.OneEndFixedJax)
    simulator.operate_block(rod_block_2).using(eaj.OneEndFixedJax)
    simulator.operate_block(rod_block_1).using(
        eaj.GravityAnalyticalDamperJax,
        time_step=time_step,
        uniform_damping_constant=0.5,
    )
    simulator.operate_block(rod_block_2).using(
        eaj.GravityAnalyticalDamperJax,
        time_step=time_step,
        uniform_damping_constant=0.5,
    )
    simulator.finalize()

    stepper = eaj.PositionVerletJAX()
    integrate_start = time.perf_counter()
    stepper.integrate(
        simulator,
        time=0.0,
        final_time=final_time,
        dt=time_step,
    )
    jax.block_until_ready(rod_block_1)
    jax.block_until_ready(rod_block_2)
    integrate_walltime = time.perf_counter() - integrate_start

    # Fetch all rods data
    rod_block_1.from_device()
    rod_block_2.from_device()
    final_positions = [rod.position_collection for rod in rods]

    print("\nIntegrated cantilever tips (z displacement):")
    for rod_index, (initial, final) in enumerate(
        zip(initial_positions, final_positions, strict=True)
    ):
        tip_delta = final[:, -1] - initial[:, -1]
        print(f"  rod {rod_index}: dz = {tip_delta[2]:.6f}")

    print(f"integrate walltime: {integrate_walltime:.4f} s")
    if show:
        import matplotlib.pyplot as plt

        for idx, final in enumerate(final_positions):
            plt.plot(final[2, :], final[1, :], label=f"zy {idx}")
            plt.plot(final[2, :], final[0, :], label=f"zx {idx}")
        plt.xlabel("z")
        plt.ylabel("x or y")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Cantilevers on separate devices")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main(show=True)
