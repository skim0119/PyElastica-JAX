"""Tutorial: shard rod memory across multiple JAX devices.

A single ``configure_rod_block`` keeps every rod on one device. A **sharded
memory block** splits rods evenly across the requested JAX devices:

    rod_block = eaj.configure_rod_block_sharded(
        devices=jax.devices(),
        device_dtype=np.float64,
    )

During ``finalize()``, PyElastica builds one inner block per device. Block
state is then stored as a dictionary with a ``"shards"`` entry instead of
one flat device buffer.

This script emulates multiple CPU devices with ``XLA_FLAGS`` so it runs
without CUDA hardware. On a multi-GPU machine, build the mesh from
``eaj.resolve_backend_devices("cuda")`` instead of ``jax.devices("cpu")``.

For a timed multi-GPU rollout, see
``benchmark/snake-self-activate-single-node/`` and the ``gpu2x`` /
``gpu2x_sharded`` backends.
"""

from __future__ import annotations

import os
import time

N_DEVICES = 4
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_DEVICES}"

import numpy as np

import elastica as ea
import elastica_jax as eaj
import jax

jax.config.update("jax_enable_x64", True)


class CantileverSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    pass


def main(
    n_rods: int = N_DEVICES,
    final_time: float = 0.1,
    time_step: float = 1.0e-4,
    show: bool = False,
) -> None:
    simulator = CantileverSimulator()
    rod_block = eaj.configure_rod_block_sharded(
        devices=jax.devices(),
        device_dtype=np.float64,
    )
    simulator.enable_block_supports(ea.CosseratRod, rod_block)

    initial_positions: list[np.ndarray] = []
    rods = []
    for rod_index in range(n_rods):
        # Just to give offset in the visualization:
        start = np.array([0.1 * rod_index, 0.1 * rod_index, 0.0], dtype=np.float64)
        rod = ea.CosseratRod.straight_rod(
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

    simulator.operate_block(ea.CosseratRod).using(eaj.OneEndFixedJax)
    simulator.operate_block(ea.CosseratRod).using(
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
    jax.block_until_ready(rod_block)
    integrate_walltime = time.perf_counter() - integrate_start

    # Fetch all rods data
    rod_block.from_device()
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
        plt.title("Rod (all shards identical)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main(show=True)
