"""Tutorial: synchronize rod host memory with a JAX memory block."""

from __future__ import annotations

import time

import numpy as np

import elastica as ea
import elastica_jax as eaj
import jax

jax.config.update("jax_enable_x64", True)


class CantileverSimulator(eaj.Simulator):
    pass


def main(
    final_time: float = 100.0, time_step: float = 1.0e-4, show: bool = False
) -> None:
    rod = ea.CosseratRod.straight_rod(
        n_elements=20,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=0.35,
        base_radius=0.01,
        density=1_000.0,
        youngs_modulus=5.0e6,
    )
    initial_position = rod.position_collection.copy()

    simulator = CantileverSimulator()
    rod_block = eaj.configure_rod_block()
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    simulator.append(rod)

    simulator.operate(rod).using(eaj.OneEndFixedJax)
    simulator.operate(rod).using(
        eaj.GravityAnalyticalDamperJax,
        time_step=time_step,
        uniform_damping_constant=0.5,
    )

    simulator.finalize()

    stepper = eaj.PositionVerletJAX()
    integrate_start = time.perf_counter()
    stepper.integrate(simulator, time=0.0, final_time=final_time, dt=time_step)
    jax.block_until_ready(rod_block)
    integrate_walltime = time.perf_counter() - integrate_start

    print("Rod before sync still shows initial host values:")
    print(np.allclose(rod.position_collection, initial_position))

    from_device_start = time.perf_counter()
    rod_block.from_device(rod, variables=("position_collection",))
    from_device_walltime = time.perf_counter() - from_device_start

    print("Rod after block.from_device(...) shows integrated values:")
    print(not np.allclose(rod.position_collection, initial_position))
    print("Tip displacement:", rod.position_collection[:, -1] - initial_position[:, -1])

    # # rod.position_collection[:] = initial_position
    # # to_device_start = time.perf_counter()
    # # rod_block.to_device(rod, variables=("position_collection",))
    # # jax.block_until_ready(rod_block)
    # # to_device_walltime = time.perf_counter() - to_device_start
    # # print("Block reset from rod host memory via block.to_device(...)")

    print(f"integrate walltime:    {integrate_walltime:.4f} s")
    print(f"from_device walltime:  {from_device_walltime * 1e3:.3f} ms")
    # print(f"to_device walltime:    {to_device_walltime * 1e3:.3f} ms")

    if show:
        import matplotlib.pyplot as plt

        print(rod.position_collection.shape)

        plt.plot(
            rod.position_collection[2, :], rod.position_collection[1, :], label="zy"
        )
        plt.plot(
            rod.position_collection[2, :], rod.position_collection[0, :], label="zx"
        )
        plt.xlabel("z")
        plt.ylabel("x or y")
        plt.gca().set_aspect("equal", adjustable="box")

        plt.title("Rod")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main(show=True)
