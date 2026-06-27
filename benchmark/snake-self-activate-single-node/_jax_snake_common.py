"""Shared helpers for multi-snake Numba vs JAX benchmarks."""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import numpy as np

import elastica as ea
import elastica_jax as eaj

import jax
import jax.numpy as jnp

from elastica_jax._linalg import _jax_batch_cross, _jax_batch_matvec

type BenchmarkTiming = tuple[float, float]

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


def build_rod() -> ea.CosseratRod:
    return ea.CosseratRod.straight_rod(
        DEFAULT_N_ELEM,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        DEFAULT_BASE_LENGTH,
        DEFAULT_BASE_RADIUS,
        DEFAULT_DENSITY,
        DEFAULT_YOUNGS_MODULUS,
    )


class PyElasticaSimulator(ea.BaseSystemCollection, ea.Forcing, ea.Damping, ea.Contact):
    pass


class JAXSimulator(ea.BaseSystemCollection, eaj.JAXOpsBlock):
    pass


def _uniform_index_matrix(
    start_idx: np.ndarray,
    end_idx: np.ndarray,
) -> np.ndarray:
    widths = end_idx - start_idx
    assert np.all(widths == widths[0]), "All rods must share the same discretization."
    offsets = np.arange(int(widths[0]), dtype=np.int32)
    return start_idx[:, None].astype(np.int32) + offsets[None, :]


def _gather_vector_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.moveaxis(jnp.take(array, indices, axis=-1), 1, 0)


def _gather_tensor_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.moveaxis(jnp.take(array, indices, axis=-1), 2, 0)


def _gather_scalar_batch(array: jax.Array, indices: jax.Array) -> jax.Array:
    return jnp.take(array, indices, axis=-1)


def _scatter_set_vector_batch(
    array: jax.Array, indices: jax.Array, values: jax.Array
) -> jax.Array:
    return array.at[:, indices].set(jnp.moveaxis(values, 0, 1))


def _batch_matvec_over_rods(
    matrix_collection: jax.Array,
    vector_collection: jax.Array,
) -> jax.Array:
    return jax.vmap(_jax_batch_matvec, in_axes=(0, 0))(
        matrix_collection, vector_collection
    )


def _batch_cross_over_rods(
    first_vector_collection: jax.Array,
    second_vector_collection: jax.Array,
) -> jax.Array:
    return jax.vmap(_jax_batch_cross, in_axes=(0, 0))(
        first_vector_collection,
        second_vector_collection,
    )


def _node_to_element_position_batch(position_collection: jax.Array) -> jax.Array:
    return 0.5 * (position_collection[:, :, 1:] + position_collection[:, :, :-1])


def _node_to_element_velocity_batch(
    mass: jax.Array, velocity_collection: jax.Array
) -> jax.Array:
    numerator = (
        mass[:, None, 1:] * velocity_collection[:, :, 1:]
        + mass[:, None, :-1] * velocity_collection[:, :, :-1]
    )
    denominator = mass[:, None, 1:] + mass[:, None, :-1]
    return numerator / denominator


def _node_to_element_mass_or_force_batch(nodal_collection: jax.Array) -> jax.Array:
    elemental_collection = 0.5 * (
        nodal_collection[:, :, :-1] + nodal_collection[:, :, 1:]
    )
    elemental_collection = elemental_collection.at[:, :, 0].add(
        0.5 * nodal_collection[:, :, 0]
    )
    elemental_collection = elemental_collection.at[:, :, -1].add(
        0.5 * nodal_collection[:, :, -1]
    )
    return elemental_collection


def _elements_to_nodes_batch(element_collection: jax.Array) -> jax.Array:
    node_collection = jnp.zeros(
        (
            element_collection.shape[0],
            element_collection.shape[1],
            element_collection.shape[2] + 1,
        ),
        dtype=element_collection.dtype,
    )
    node_collection = node_collection.at[:, :, :-1].add(0.5 * element_collection)
    node_collection = node_collection.at[:, :, 1:].add(0.5 * element_collection)
    return node_collection


def _find_slipping_elements_batch(
    velocity_slip: jax.Array, velocity_threshold: jax.Array
) -> jax.Array:
    abs_velocity_slip = jnp.linalg.norm(velocity_slip, axis=1)
    normalized = abs_velocity_slip / velocity_threshold - 1.0
    slipped = jnp.minimum(1.0, normalized)
    slip_function = jnp.ones_like(abs_velocity_slip)
    slip_values = jnp.abs(1.0 - slipped)
    return jnp.where(abs_velocity_slip > velocity_threshold, slip_values, slip_function)


from snake_operation import GravityPlaneContactBlockJax, SnakeMuscleTorquesBlockJax


def two_gpu_half_split_mesh(n_snakes: int) -> eaj.ExecutionMesh:
    """Map the first half of snakes to GPU 0 and the remainder to GPU 1."""
    assert n_snakes >= 2, "gpu2x requires at least two snakes."
    devices = eaj.resolve_backend_devices("cuda")
    assert len(devices) >= 2, "gpu2x requires at least two CUDA devices."
    split = n_snakes // 2
    rod_to_shard = np.array(
        [0] * split + [1] * (n_snakes - split),
        dtype=np.int32,
    )
    return eaj.ExecutionMesh(devices=devices[:2], rod_to_shard=rod_to_shard)


def _block_until_ready_rod_block(
    rod_block: eaj._CosseratRodMemoryBlock | eaj._ShardedCosseratRodBlock,
) -> None:
    shard_blocks = getattr(rod_block, "_shard_blocks", None)
    mesh = getattr(rod_block, "mesh", None)
    if shard_blocks is not None and mesh is not None and mesh.is_sharded:
        for shard_block in shard_blocks:
            jax.block_until_ready(shard_block)
        return
    jax.block_until_ready(rod_block)


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
    device: jax.Device | eaj.ExecutionMesh,
    device_dtype: np.dtype,
    n_snakes: int,
    sharded: bool = False,
) -> tuple[JAXSimulator, eaj._CosseratRodMemoryBlock | eaj._ShardedCosseratRodBlock]:
    if sharded:
        rod_block = eaj.configure_rod_block_sharded(
            mesh=device,
            device_dtype=np.dtype(device_dtype),
        )
    else:
        rod_block = eaj.configure_rod_block(
            device=device,
            device_dtype=np.dtype(device_dtype),
        )

    b_coeff = default_b_coeff()
    period = DEFAULT_PERIOD
    mu = DEFAULT_BASE_LENGTH / (
        period * period * np.abs(DEFAULT_GRAVITY) * DEFAULT_FROUDE
    )
    kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu], dtype=np.float64)
    static_mu_array = np.zeros(kinetic_mu_array.shape, dtype=np.float64)

    sim = JAXSimulator()
    sim.enable_block_supports(ea.CosseratRod, rod_block)
    for idx in range(n_snakes):
        rod = build_rod()
        sim.append(rod)

    sim.operate_block(ea.CosseratRod).using(
        SnakeMuscleTorquesBlockJax,
        b_coeff=b_coeff,
        period=period,
        base_length=DEFAULT_BASE_LENGTH,
    )
    sim.operate_block(ea.CosseratRod).using(
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
    sim.operate_block(ea.CosseratRod).using(
        eaj.AnalyticalLinearDamperJax,
        time_step=np.float64(DEFAULT_DT),
        damping_constant=DEFAULT_DAMPING,
    )
    sim.finalize()

    return sim, rod_block


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


def integrate_jax_block_rollout(
    jax_sim: JAXSimulator,
    jax_block: eaj._CosseratRodMemoryBlock | eaj._ShardedCosseratRodBlock,
    *,
    steps: int,
    warmup_runs: int,
) -> float:
    dt_value = DEFAULT_DT
    stepper = eaj.PositionVerletJAX()
    time_value = np.float64(0.0)
    time_value = stepper.integrate(
        jax_sim,
        time=time_value,
        final_time=time_value + warmup_runs * dt_value,
        dt=dt_value,
    )
    _block_until_ready_rod_block(jax_block)
    rollout_start = time.perf_counter()
    stepper.integrate(
        jax_sim,
        time=time_value,
        final_time=time_value + steps * dt_value,
        dt=dt_value,
    )
    _block_until_ready_rod_block(jax_block)
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

    with jax.default_device(device):
        instantiate_start = time.perf_counter()
        jax_sim, jax_block = build_jax_sim(
            device=device,
            device_dtype=dtype,
            n_snakes=n_snakes,
        )
        _block_until_ready_rod_block(jax_block)
        instantiate_seconds = time.perf_counter() - instantiate_start
        rollout_seconds = integrate_jax_block_rollout(
            jax_sim,
            jax_block,
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
    """Build a 2-GPU sharded JAX block simulator and time a Position Verlet rollout."""
    dtype = np.dtype(np.float64)
    mesh = two_gpu_half_split_mesh(n_snakes)

    instantiate_start = time.perf_counter()
    jax_sim, jax_block = build_jax_sim(
        device=mesh,
        device_dtype=dtype,
        n_snakes=n_snakes,
        sharded=True,
    )
    _block_until_ready_rod_block(jax_block)
    instantiate_seconds = time.perf_counter() - instantiate_start
    rollout_seconds = integrate_jax_block_rollout(
        jax_sim,
        jax_block,
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
    """Run two single-GPU block rollouts in parallel, one per CUDA device."""
    devices = eaj.resolve_backend_devices("cuda")
    split = n_snakes // 2
    remainder = n_snakes - split
    jobs = (
        (devices[0], split),
        (devices[1], remainder),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_jax_rollout,
                device=device,
                n_snakes=count,
                steps=steps,
                warmup_runs=warmup_runs,
            )
            for device, count in jobs
        ]
        timings = [future.result() for future in futures]

    instantiate_seconds = max(timing[0] for timing in timings)
    rollout_seconds = max(timing[1] for timing in timings)
    return instantiate_seconds, rollout_seconds
