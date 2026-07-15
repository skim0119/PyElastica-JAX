"""Shared helpers for multi-snake Numba vs JAX benchmarks."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import numpy as np  # noqa: E402

import elastica as ea  # noqa: E402
import elastica_jax as eaj  # noqa: E402

import jax  # noqa: E402

from snake_operation import (  # noqa: E402
    GravityPlaneContactBlockJax,
    SnakeMuscleTorquesBlockJax,
)

BenchmarkTiming: TypeAlias = tuple[float, float]

jax.config.update("jax_enable_x64", True)

DEFAULT_PERIOD = 2.0
DEFAULT_BASE_LENGTH = 0.35
DEFAULT_BASE_RADIUS = DEFAULT_BASE_LENGTH * 0.011
DEFAULT_DENSITY = 1000.0
DEFAULT_YOUNGS_MODULUS = 1.0e6
DEFAULT_POISSON_RATIO = 0.5
DEFAULT_GRAVITY = -9.80665
DEFAULT_DAMPING = 2.0e-3
DEFAULT_FROUDE = 0.1
DEFAULT_N_ELEM = 50
DEFAULT_DT = 1.0e-4


def default_b_coeff() -> np.ndarray:
    return np.array(
        [3.4e-3, 3.3e-3, 4.2e-3, 2.6e-3, 3.6e-3, 3.5e-3, 1.0],
        dtype=np.float64,
    )


def build_rod(
    rod_type: type[ea.CosseratRod] = ea.CosseratRod,
) -> ea.CosseratRod:
    """Build one benchmark rod using the requested concrete rod type."""
    return rod_type.straight_rod(
        n_elements=DEFAULT_N_ELEM,
        start=np.zeros(3),
        direction=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=DEFAULT_BASE_LENGTH,
        base_radius=DEFAULT_BASE_RADIUS,
        density=DEFAULT_DENSITY,
        youngs_modulus=DEFAULT_YOUNGS_MODULUS,
    )


class PyElasticaSimulator(ea.BaseSystemCollection, ea.Forcing, ea.Damping, ea.Contact):
    pass


class JAXSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    pass


JAXRodBlock: TypeAlias = (
    eaj._CosseratRodMemoryBlock
    | eaj._CosseratRodVerticalMemoryBlock
    | eaj._MpiCosseratRodBlock
)


def _distinct_cosserat_rod_types() -> tuple[type[ea.CosseratRod], type[ea.CosseratRod]]:
    excluded_attributes = {
        "__dict__",
        "__weakref__",
        "__module__",
        "__annotations__",
        "__doc__",
        "__qualname__",
    }
    rod_attributes = {
        name: value
        for name, value in ea.CosseratRod.__dict__.items()
        if name not in excluded_attributes
    }
    first_type = type(
        "SnakeRodOnDevice0",
        ea.CosseratRod.__bases__,
        dict(rod_attributes),
    )
    second_type = type(
        "SnakeRodOnDevice1",
        ea.CosseratRod.__bases__,
        dict(rod_attributes),
    )
    return first_type, second_type


def _configure_jax_block_operators(
    simulator: JAXSimulator,
    rod_blocks: Sequence[JAXRodBlock],
) -> None:
    b_coeff = default_b_coeff()
    period = DEFAULT_PERIOD
    mu = DEFAULT_BASE_LENGTH / (
        period * period * np.abs(DEFAULT_GRAVITY) * DEFAULT_FROUDE
    )
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)

    for rod_block in rod_blocks:
        simulator.operate_block(rod_block).using(
            SnakeMuscleTorquesBlockJax,
            b_coeff=b_coeff,
            period=period,
            base_length=DEFAULT_BASE_LENGTH,
        )
        simulator.operate_block(rod_block).using(
            GravityPlaneContactBlockJax,
            plane_origin=np.array(
                [0.0, -DEFAULT_BASE_LENGTH * 0.011, 0.0], dtype=np.float64
            ),
            plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            slip_velocity_tol=1.0e-8,
            k=1.0,
            nu=1.0e-6,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
        simulator.operate_block(rod_block).using(
            eaj.AnalyticalLinearDamperJax,
            time_step=np.float64(DEFAULT_DT),
            damping_constant=DEFAULT_DAMPING,
        )


def _block_until_ready(rod_blocks: Sequence[JAXRodBlock]) -> None:
    for rod_block in rod_blocks:
        for leaf in jax.tree_util.tree_leaves(rod_block.jax_get_state()):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()


def build_cpu_sim(
    *,
    n_snakes: int,
    n_elem: int = DEFAULT_N_ELEM,
    period: float = DEFAULT_PERIOD,
    base_length: float = DEFAULT_BASE_LENGTH,
    density: float = DEFAULT_DENSITY,
    youngs_modulus: float = DEFAULT_YOUNGS_MODULUS,
    poisson_ratio: float = DEFAULT_POISSON_RATIO,
    time_step: float = DEFAULT_DT,
) -> tuple[PyElasticaSimulator, list[ea.CosseratRod]]:
    b_coeff = default_b_coeff()
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    wave_length = float(b_coeff[-1])
    mu = base_length / (period * period * np.abs(DEFAULT_GRAVITY) * DEFAULT_FROUDE)
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)

    sim = PyElasticaSimulator()
    rods: list[ea.CosseratRod] = []
    ground_plane = ea.Plane(
        plane_origin=np.array([0.0, -base_length * 0.011, 0.0], dtype=np.float64),
        plane_normal=normal,
    )
    sim.append(ground_plane)

    for idx in range(n_snakes):
        rod = build_rod()
        sim.append(rod)
        sim.add_forcing_to(rod).using(
            ea.GravityForces,
            acc_gravity=np.array([0.0, DEFAULT_GRAVITY, 0.0], dtype=np.float64),
        )
        sim.add_forcing_to(rod).using(
            ea.MuscleTorques,
            base_length=base_length,
            b_coeff=b_coeff[:-1],
            period=period,
            wave_number=2.0 * np.pi / wave_length,
            phase_shift=0.0,
            rest_lengths=rod.rest_lengths,
            ramp_up_time=period,
            direction=normal,
            with_spline=True,
        )
        sim.detect_contact_between(rod, ground_plane).using(
            ea.RodPlaneContactWithAnisotropicFriction,
            k=1.0,
            nu=1.0e-6,
            slip_velocity_tol=1.0e-8,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
        sim.dampen(rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=DEFAULT_DAMPING,
            time_step=time_step,
        )
        rods.append(rod)

    sim.finalize()
    return sim, rods


def build_jax_sim(
    *,
    device: jax.Device | Sequence[jax.Device],
    device_dtype: np.dtype,
    n_snakes: int,
    sharded: bool = False,
    inner_block_cls: type | None = None,
) -> tuple[JAXSimulator, JAXRodBlock]:
    rod_block: JAXRodBlock
    if sharded:
        raise NotImplementedError
    else:
        assert isinstance(device, jax.Device), (
            "A non-sharded JAX block requires exactly one device."
        )
        configure_kwargs: dict[str, Any] = {
            "device": device,
            "device_dtype": np.dtype(device_dtype),
        }
        if inner_block_cls is not None:
            configure_kwargs["inner_block_cls"] = inner_block_cls
        rod_block = eaj.configure_rod_block(**configure_kwargs)

    sim = JAXSimulator()
    sim.enable_block_supports(ea.CosseratRod, rod_block)
    for _ in range(n_snakes):
        rod = build_rod()
        sim.append(rod)

    _configure_jax_block_operators(sim, (rod_block,))
    sim.finalize()

    return sim, rod_block


def build_jax_sim_gpu2x(
    *,
    devices: Sequence[jax.Device],
    device_dtype: np.dtype,
    n_snakes: int,
) -> tuple[JAXSimulator, tuple[JAXRodBlock, JAXRodBlock]]:
    """Build two explicitly assigned rod blocks on separate devices."""
    assert len(devices) >= 2, "gpu2x requires at least two devices."
    assert n_snakes >= 2, "gpu2x requires at least two snakes."

    first_rod_type, second_rod_type = _distinct_cosserat_rod_types()
    rod_blocks = (
        eaj.configure_rod_block(
            device=devices[0],
            device_dtype=np.dtype(device_dtype),
        ),
        eaj.configure_rod_block(
            device=devices[1],
            device_dtype=np.dtype(device_dtype),
        ),
    )
    simulator = JAXSimulator()
    simulator.enable_block_supports(first_rod_type, rod_blocks[0])
    simulator.enable_block_supports(second_rod_type, rod_blocks[1])

    first_block_count = (n_snakes + 1) // 2
    for snake_index in range(n_snakes):
        rod_type = (
            first_rod_type if snake_index < first_block_count else second_rod_type
        )
        simulator.append(build_rod(rod_type))

    _configure_jax_block_operators(simulator, rod_blocks)
    simulator.finalize()
    return simulator, rod_blocks


def time_average(n_iter: int, fn) -> float:  # type: ignore[no-untyped-def]
    assert n_iter > 0, "n_iter must be positive."
    start = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - start) / n_iter


def snapshot_jax_state_to_host(
    state: dict[str, jax.Array],
) -> dict[str, np.ndarray]:
    host_state = jax.device_get(state)
    return {key: np.asarray(value).copy() for key, value in host_state.items()}


def restore_jax_state_from_host(
    host_state: dict[str, np.ndarray],
    device: jax.Device,
) -> dict[str, jax.Array]:
    return {
        key: jax.device_put(np.asarray(value), device=device)
        for key, value in host_state.items()
    }


def save_jax_state_npz(path: Path, state: dict[str, np.ndarray]) -> None:
    np.savez(path, **state)


def load_jax_state_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(value).copy() for key, value in data.items()}


def emit_report(lines: list[str], log_path: Path | None) -> None:
    report = "\n".join(lines)
    print(report)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(report + "\n", encoding="utf-8")


def run_pyelastica_rollout(
    *,
    n_snakes: int,
    steps: int,
    warmup_runs: int,
    n_elem: int = DEFAULT_N_ELEM,
    dt: float = DEFAULT_DT,
) -> BenchmarkTiming:
    """Build a PyElastica simulator and time a fixed-length Position Verlet rollout."""
    dt_value = np.float64(dt)

    instantiate_start = time.perf_counter()
    cpu_sim, _cpu_rods = build_cpu_sim(
        n_snakes=n_snakes,
        n_elem=n_elem,
        time_step=dt,
    )
    instantiate_seconds = time.perf_counter() - instantiate_start

    time_value = np.float64(0.0)
    stepper = ea.PositionVerlet()
    for _ in range(warmup_runs):
        time_value = stepper.step(cpu_sim, time_value, dt_value)

    rollout_start = time.perf_counter()
    for _ in range(steps):
        time_value = stepper.step(cpu_sim, time_value, dt_value)
    rollout_seconds = time.perf_counter() - rollout_start
    return instantiate_seconds, rollout_seconds


def run_pyelastica_rollout_mpi(
    *,
    comm: Any,
    snakes_per_rank: int,
    steps: int,
    warmup_runs: int,
) -> tuple[float, list[float] | None]:
    """
    Time one MPI-local PyElastica rollout and gather per-rank walltimes.

    Each rank independently builds and integrates ``snakes_per_rank`` snakes
    with the same operators as ``run_pyelastica_rollout`` (muscle torques,
    gravity, anisotropic plane friction, analytical damping, PositionVerlet).
    There is no inter-rank physics coupling; this matches the weak-scaling
    work per JAX MPI rank.

    Parameters
    ----------
    comm
        MPI communicator used to gather rollout timing samples.
    snakes_per_rank
        Number of snakes owned and integrated by this rank.
    steps
        Number of timed integration steps.
    warmup_runs
        Number of warmup steps before timing.

    Returns
    -------
    tuple[float, list[float] | None]
        Maximum instantiation time across ranks and gathered per-rank rollout
        times on rank 0. Non-root ranks receive ``None`` for the gathered
        rollout times.
    """
    from mpi4py import MPI

    assert snakes_per_rank > 0, "snakes_per_rank must be positive."
    instantiate_seconds, rollout_seconds = run_pyelastica_rollout(
        n_snakes=snakes_per_rank,
        steps=steps,
        warmup_runs=warmup_runs,
    )
    rollout_seconds_all_ranks = comm.gather(rollout_seconds, root=0)
    instantiate_seconds = comm.allreduce(instantiate_seconds, op=MPI.MAX)
    return instantiate_seconds, rollout_seconds_all_ranks


def integrate_jax_block_rollout(
    jax_sim: JAXSimulator,
    jax_blocks: Sequence[JAXRodBlock],
    *,
    steps: int,
    warmup_runs: int,
) -> float:
    dt_value = DEFAULT_DT
    stepper = eaj.PositionVerletJAX()
    time_value = np.float64(0.0)
    for _ in range(warmup_runs):
        time_value = stepper.integrate(
            jax_sim,
            time=time_value,
            final_time=time_value + steps * dt_value,
            dt=dt_value,
        )
        _block_until_ready(jax_blocks)
    rollout_start = time.perf_counter()
    time_value = stepper.integrate(
        jax_sim,
        time=time_value,
        final_time=time_value + steps * dt_value,
        dt=dt_value,
    )
    _block_until_ready(jax_blocks)
    return time.perf_counter() - rollout_start


def run_jax_rollout(
    *,
    backend: str,
    n_snakes: int,
    steps: int,
    warmup_runs: int,
) -> BenchmarkTiming:
    """Build a single-device JAX block simulator and time a Position Verlet rollout."""
    dtype = np.dtype(np.float64)
    device = eaj.resolve_backend_devices(backend)[0]
    # CPU sweep uses the stacked-axis vertical block; CUDA keeps horizontal packing.
    inner_block_cls = (
        eaj._CosseratRodVerticalMemoryBlock
        if backend == "cpu"
        else eaj._CosseratRodMemoryBlock
    )

    with jax.default_device(device):
        instantiate_start = time.perf_counter()
        jax_sim, jax_block = build_jax_sim(
            device=device,
            device_dtype=dtype,
            n_snakes=n_snakes,
            inner_block_cls=inner_block_cls,
        )
        _block_until_ready((jax_block,))
        instantiate_seconds = time.perf_counter() - instantiate_start
        rollout_seconds = integrate_jax_block_rollout(
            jax_sim,
            (jax_block,),
            steps=steps,
            warmup_runs=warmup_runs,
        )

    return instantiate_seconds, rollout_seconds


def run_jax_rollout_gpu2x_sharded(
    *,
    n_snakes: int,
    steps: int,
    warmup_runs: int,
) -> BenchmarkTiming:
    raise NotImplementedError

    """Build a 2-GPU sharded JAX block simulator and time a Position Verlet rollout."""
    assert n_snakes % 2 == 0, "gpu2x_sharded requires a snake count divisible by two."
    dtype = np.dtype(np.float64)
    devices = two_gpu_sharded_devices()

    instantiate_start = time.perf_counter()
    jax_sim, jax_block = build_jax_sim(
        device=devices,
        device_dtype=dtype,
        n_snakes=n_snakes,
        sharded=True,
    )
    _block_until_ready((jax_block,))
    instantiate_seconds = time.perf_counter() - instantiate_start
    rollout_seconds = integrate_jax_block_rollout(
        jax_sim,
        (jax_block,),
        steps=steps,
        warmup_runs=warmup_runs,
    )

    return instantiate_seconds, rollout_seconds


def run_jax_rollout_gpu2x(
    *,
    n_snakes: int,
    steps: int,
    warmup_runs: int,
) -> BenchmarkTiming:
    """Build two explicitly assigned GPU blocks and time one JAX rollout."""
    instantiate_start = time.perf_counter()
    dtype = np.dtype(np.float64)
    devices = eaj.resolve_backend_devices("cuda")
    jax_sim, jax_blocks = build_jax_sim_gpu2x(
        devices=devices,
        device_dtype=dtype,
        n_snakes=n_snakes,
    )
    _block_until_ready(jax_blocks)
    instantiate_seconds = time.perf_counter() - instantiate_start
    rollout_seconds = integrate_jax_block_rollout(
        jax_sim,
        jax_blocks,
        steps=steps,
        warmup_runs=warmup_runs,
    )

    return instantiate_seconds, rollout_seconds


def mpi_snakes_per_rank(
    *,
    snakes_per_rank_exp: int,
    snakes_per_rank_multiplier: int = 1,
) -> int:
    """
    Return the number of snakes owned by one MPI rank.

    Parameters
    ----------
    snakes_per_rank_exp
        Exponent for the base per-rank snake count ``2 ** exp``.
    snakes_per_rank_multiplier
        Scalar applied before weak scaling across MPI ranks. Use this to make
        one GPU rank match the aggregate work of several CPU ranks on a node.

    Returns
    -------
    int
        ``snakes_per_rank_multiplier * (2 ** snakes_per_rank_exp)``.
    """
    assert snakes_per_rank_multiplier > 0, (
        "snakes_per_rank_multiplier must be positive."
    )
    return snakes_per_rank_multiplier * (2**snakes_per_rank_exp)


def mpi_global_snake_count(
    *,
    snakes_per_rank_exp: int,
    comm_size: int,
    snakes_per_rank_multiplier: int = 1,
) -> int:
    """Return the weak-scaling global snake count for an MPI world size."""
    return (
        mpi_snakes_per_rank(
            snakes_per_rank_exp=snakes_per_rank_exp,
            snakes_per_rank_multiplier=snakes_per_rank_multiplier,
        )
        * comm_size
    )


def resolve_mpi_block_device(*, backend: str, comm: Any) -> jax.Device:
    """
    Return the JAX device for one MPI rank's local rod block.

    Parameters
    ----------
    backend
        JAX backend name such as ``"cpu"`` or ``"cuda"``.
    comm
        MPI communicator used to resolve a fallback local rank.

    Returns
    -------
    jax.Device
        Device passed to ``configure_rod_block_mpi``.
    """
    devices = eaj.resolve_backend_devices(backend)
    if backend == "cpu":
        return devices[0]

    local_rank = int(os.environ.get("SLURM_LOCALID", comm.Get_rank()))
    return devices[local_rank % len(devices)]


def build_jax_sim_mpi(
    *,
    comm: Any,
    device_dtype: np.dtype,
    n_snakes_total: int,
    backend: str = "cpu",
    vertical: bool = False,
) -> tuple[JAXSimulator, eaj._MpiCosseratRodBlock]:
    """Build a JAX simulator with rods distributed across MPI ranks."""
    device = resolve_mpi_block_device(backend=backend, comm=comm)
    inner_block_cls = (
        eaj._CosseratRodVerticalMemoryBlock if vertical else eaj._CosseratRodMemoryBlock
    )
    rod_block = eaj.configure_rod_block_mpi(
        comm=comm,
        device_dtype=np.dtype(device_dtype),
        device=device,
        inner_block_cls=inner_block_cls,
    )
    simulator = JAXSimulator()
    simulator.enable_block_supports(ea.CosseratRod, rod_block)
    for snake_index in range(n_snakes_total):
        if rod_block.owns_rod(snake_index):
            simulator.append(build_rod())
    _configure_jax_block_operators(simulator, (rod_block,))
    simulator.finalize()
    return simulator, rod_block


def run_jax_rollout_mpi(
    *,
    comm: Any,
    n_snakes_total: int,
    steps: int,
    warmup_runs: int,
    backend: str = "cpu",
    vertical: bool = False,
) -> tuple[float, list[float] | None]:
    """
    Build an MPI-local JAX block simulator and time a Position Verlet rollout.

    Parameters
    ----------
    comm
        MPI communicator used to gather rollout timing samples.
    n_snakes_total
        Total number of snakes across all MPI ranks.
    steps
        Number of timed integration steps.
    warmup_runs
        Number of warmup integration chunks before timing.
    backend
        JAX backend for the MPI-local rod block, such as ``"cpu"`` or ``"cuda"``.
    vertical
        If True, pack rods with ``_CosseratRodVerticalMemoryBlock``.

    Returns
    -------
    tuple[float, list[float] | None]
        Maximum instantiation time across ranks and gathered per-rank rollout
        times on rank 0. Non-root ranks receive ``None`` for the gathered
        rollout times.
    """
    from mpi4py import MPI

    dtype = np.dtype(np.float64)
    device = resolve_mpi_block_device(backend=backend, comm=comm)
    instantiate_start = time.perf_counter()
    with jax.default_device(device):
        jax_sim, jax_block = build_jax_sim_mpi(
            comm=comm,
            device_dtype=dtype,
            n_snakes_total=n_snakes_total,
            backend=backend,
            vertical=vertical,
        )
        _block_until_ready((jax_block,))
        instantiate_seconds = time.perf_counter() - instantiate_start
        rollout_seconds = integrate_jax_block_rollout(
            jax_sim,
            (jax_block,),
            steps=steps,
            warmup_runs=warmup_runs,
        )
    rollout_seconds_all_ranks = comm.gather(rollout_seconds, root=0)
    instantiate_seconds = comm.allreduce(instantiate_seconds, op=MPI.MAX)
    return instantiate_seconds, rollout_seconds_all_ranks
