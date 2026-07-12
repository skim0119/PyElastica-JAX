"""Stacked-axis JAX memory block for equal-length Cosserat rods.

Horizontal blocks pack rods into one contiguous ``(3, N_total)`` matrix with
ghost separators between rods. This block instead stacks rods on a leading
batch axis:

- vector fields: ``(n_rods, 3, N)``
- tensor fields: ``(n_rods, 3, 3, N)``
- scalar fields: ``(n_rods, N)``

where ``N`` is ``n_nodes``, ``n_elems``, or ``n_voronoi`` depending on the
variable. Per-rod kernels run under :func:`jax.vmap`.

Only equal-length straight rods are supported.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np

from elastica.rod.cosserat_rod import (
    _compute_bending_twist_strains,
    _compute_shear_stretch_strains,
)
from elastica.rod.data_structures import _RodSymplecticStepperMixin
from elastica.rod.rod_base import RodBase
from elastica.typing import RodType, SystemIdxType

from elastica_jax._calculus import (
    _jax_average as _jax_position_average,
    _jax_difference as _jax_position_difference,
    _jax_trapezoidal,
    _jax_two_point_difference,
)
from elastica_jax._linalg import (
    _jax_batch_cross,
    _jax_batch_dot,
    _jax_batch_matmul,
    _jax_batch_matvec,
    _jax_batch_transpose_matvec,
)
from elastica_jax._rotations import _jax_get_rotation_matrix, _jax_inv_rotate
from elastica_jax.memory_block.memory_block_rod_jax import (
    RodSyncTarget,
    _SYNCABLE_ATTRS,
)

import jax
import jax.numpy as jnp


class JAXRodStackedView:
    """Rod-local facade over stacked ``(n_rods, ...)`` device state."""

    def __init__(
        self,
        state: dict[str, Any],
        rod_idx: int,
        *,
        updates: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_rod_idx", rod_idx)
        object.__setattr__(self, "_updates", {} if updates is None else updates)

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            raise AttributeError(attr)
        source = self._updates.get(attr, self._state[attr])
        return source[self._rod_idx]

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr.startswith("_"):
            object.__setattr__(self, attr, value)
            return
        base = self._updates.get(attr, self._state[attr])
        self._updates[attr] = base.at[self._rod_idx].set(value)

    def commit(self) -> dict[str, Any]:
        updated = dict(self._state)
        updated.update(self._updates)
        return updated


@jax.jit
def _jax_compute_internal_forces_and_torques_one_rod(
    position_collection: jax.Array,
    velocity_collection: jax.Array,
    volume: jax.Array,
    rest_lengths: jax.Array,
    rest_voronoi_lengths: jax.Array,
    director_collection: jax.Array,
    rest_sigma: jax.Array,
    shear_matrix: jax.Array,
    mass_second_moment_of_inertia: jax.Array,
    omega_collection: jax.Array,
    bend_matrix: jax.Array,
    rest_kappa: jax.Array,
) -> tuple[jax.Array, ...]:
    """Single straight-rod force/torque kernel (no ghosts, no periodic BC)."""
    position_diff = _jax_position_difference(position_collection)
    lengths = jnp.sqrt(jnp.sum(position_diff * position_diff, axis=0)) + 1.0e-14
    tangents = position_diff / lengths[jnp.newaxis, :]
    radius = jnp.sqrt(volume / lengths / jnp.pi)
    dilatation = lengths / rest_lengths
    voronoi_dilatation = _jax_position_average(lengths) / rest_voronoi_lengths

    sigma = dilatation[jnp.newaxis, :] * _jax_batch_matvec(
        director_collection, tangents
    ) - jnp.array([[0.0], [0.0], [1.0]], dtype=position_collection.dtype)
    internal_stress = _jax_batch_matvec(shear_matrix, sigma - rest_sigma)
    cosserat_internal_stress = (
        _jax_batch_transpose_matvec(director_collection, internal_stress)
        / dilatation[jnp.newaxis, :]
    )
    internal_forces = _jax_two_point_difference(cosserat_internal_stress)

    kappa = _jax_inv_rotate(director_collection) / rest_voronoi_lengths[jnp.newaxis, :]
    internal_couple = _jax_batch_matvec(bend_matrix, kappa - rest_kappa)

    r_dot_v = _jax_batch_dot(position_collection, velocity_collection)
    r_plus_one_dot_v = _jax_batch_dot(
        position_collection[..., 1:], velocity_collection[..., :-1]
    )
    r_dot_v_plus_one = _jax_batch_dot(
        position_collection[..., :-1], velocity_collection[..., 1:]
    )
    dilatation_rate = (
        (r_dot_v[:-1] + r_dot_v[1:] - r_dot_v_plus_one - r_plus_one_dot_v)
        / lengths
        / rest_lengths
    )

    voronoi_dilatation_inv_cube = 1.0 / (voronoi_dilatation**3)
    bend_twist_couple_2d = _jax_two_point_difference(
        internal_couple * voronoi_dilatation_inv_cube[jnp.newaxis, :]
    )
    bend_twist_couple_3d = _jax_trapezoidal(
        _jax_batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths[jnp.newaxis, :]
        * voronoi_dilatation_inv_cube[jnp.newaxis, :]
    )
    shear_stretch_couple = (
        _jax_batch_cross(
            _jax_batch_matvec(director_collection, tangents), internal_stress
        )
        * rest_lengths[jnp.newaxis, :]
    )
    j_omega_upon_e = (
        _jax_batch_matvec(mass_second_moment_of_inertia, omega_collection)
        / dilatation[jnp.newaxis, :]
    )
    lagrangian_transport = _jax_batch_cross(j_omega_upon_e, omega_collection)
    unsteady_dilatation = (
        j_omega_upon_e * dilatation_rate[jnp.newaxis, :] / dilatation[jnp.newaxis, :]
    )
    internal_torques = (
        bend_twist_couple_2d
        + bend_twist_couple_3d
        + shear_stretch_couple
        + lagrangian_transport
        + unsteady_dilatation
    )

    return (
        lengths,
        tangents,
        radius,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        sigma,
        kappa,
        internal_stress,
        internal_couple,
        internal_forces,
        internal_torques,
    )


@jax.jit
def _jax_update_kinematics_one_rod(
    position_collection: jax.Array,
    director_collection: jax.Array,
    velocity_collection: jax.Array,
    omega_collection: jax.Array,
    prefac: float,
) -> tuple[jax.Array, jax.Array]:
    position_collection = position_collection + prefac * velocity_collection
    rotation_matrix = _jax_get_rotation_matrix(prefac, omega_collection)
    director_collection = _jax_batch_matmul(rotation_matrix, director_collection)
    return position_collection, director_collection


@jax.jit
def _jax_update_dynamics_one_rod(
    velocity_collection: jax.Array,
    omega_collection: jax.Array,
    acceleration_collection: jax.Array,
    alpha_collection: jax.Array,
    prefac: float,
) -> tuple[jax.Array, jax.Array]:
    velocity_collection = velocity_collection + prefac * acceleration_collection
    omega_collection = omega_collection + prefac * alpha_collection
    return velocity_collection, omega_collection


@jax.jit
def _jax_update_accelerations_one_rod(
    internal_forces: jax.Array,
    external_forces: jax.Array,
    mass: jax.Array,
    inv_mass_second_moment_of_inertia: jax.Array,
    internal_torques: jax.Array,
    external_torques: jax.Array,
    dilatation: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    acceleration_collection = (internal_forces + external_forces) / mass[jnp.newaxis, :]
    alpha_collection = (
        _jax_batch_matvec(
            inv_mass_second_moment_of_inertia,
            internal_torques + external_torques,
        )
        * dilatation[jnp.newaxis, :]
    )
    return acceleration_collection, alpha_collection


@jax.jit
def _jax_zero_external_loads_one_rod(
    external_forces: jax.Array, external_torques: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jnp.zeros_like(external_forces), jnp.zeros_like(external_torques)


_jax_compute_internal_forces_and_torques_vmap = jax.vmap(
    _jax_compute_internal_forces_and_torques_one_rod
)
_jax_update_kinematics_vmap = jax.vmap(
    _jax_update_kinematics_one_rod, in_axes=(0, 0, 0, 0, None)
)
_jax_update_dynamics_vmap = jax.vmap(
    _jax_update_dynamics_one_rod, in_axes=(0, 0, 0, 0, None)
)
_jax_update_accelerations_vmap = jax.vmap(_jax_update_accelerations_one_rod)
_jax_zero_external_loads_vmap = jax.vmap(_jax_zero_external_loads_one_rod)


@jax.tree_util.register_pytree_node_class
class _CosseratRodVerticalMemoryBlock(RodBase, _RodSymplecticStepperMixin):
    """
    Stack equal-length Cosserat rods as ``(n_rods, 3, N)`` (vectors).

    Parameters
    ----------
    device
        JAX device that owns the packed state.
    device_dtype
        ``float32`` or ``float64`` for host/device arrays.

    Notes
    -----
    All rods in the block must have the same ``n_elems``. Ring rods are
    rejected. Timestep methods apply single-rod kernels with ``jax.vmap``.
    """

    allow_cpu_fallback: bool = False

    def __init__(
        self,
        *,
        device: jax.Device,
        device_dtype: np.dtype,
    ) -> None:
        self._device_dtype = device_dtype
        self._initial_device = device

    def __call__(
        self,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> _CosseratRodVerticalMemoryBlock:
        assert len(systems) > 0, "Vertical rod block requires at least one rod."
        assert not any(system.ring_rod_flag for system in systems), (
            "Vertical rod blocks do not support ring rods."
        )
        n_elems_values = {int(system.n_elems) for system in systems}
        assert len(n_elems_values) == 1, (
            "Vertical rod blocks require equal-length rods; "
            f"got n_elems={sorted(n_elems_values)}."
        )

        self._systems = tuple(systems)
        self.n_systems = len(systems)
        self.n_rods = len(systems)
        self.ring_rod_flag = False
        self.system_idx_list = np.asarray(system_idx_list, dtype=np.int32)

        self.n_elems = int(systems[0].n_elems)
        self.n_nodes = self.n_elems + 1
        self.n_voronoi = self.n_elems - 1
        self.n_elems_in_rods = np.full(self.n_rods, self.n_elems, dtype=np.int32)

        # Per-rod spans within each stacked rod (always the full domain).
        self.start_idx_in_rod_nodes = np.zeros(self.n_rods, dtype=np.int32)
        self.end_idx_in_rod_nodes = np.full(self.n_rods, self.n_nodes, dtype=np.int32)
        self.start_idx_in_rod_elems = np.zeros(self.n_rods, dtype=np.int32)
        self.end_idx_in_rod_elems = np.full(self.n_rods, self.n_elems, dtype=np.int32)
        self.start_idx_in_rod_voronoi = np.zeros(self.n_rods, dtype=np.int32)
        self.end_idx_in_rod_voronoi = np.full(
            self.n_rods, self.n_voronoi, dtype=np.int32
        )

        self._allocate_stacked_attributes(systems)
        self._recompute_sigma_kappa()

        # Symplectic mixin placeholders; JAX path does not use these buffers.
        self.v_w_collection = np.zeros((2, 0), dtype=self._device_dtype)
        self.dvdt_dwdt_collection = np.zeros((2, 0), dtype=self._device_dtype)

        self._device_state: dict[str, jax.Array] = {}
        device = self._initial_device
        self._device_platform = device.platform
        self._device_dirty = False
        self._initialize_device_state(device=device)
        return self

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def device_platform(self) -> str:
        return self._device_platform

    @property
    def device_dtype(self) -> np.dtype:
        return self._device_dtype

    def _domain_shape(self, domain_type: Literal["node", "element", "voronoi"]) -> int:
        if domain_type == "node":
            return self.n_nodes
        if domain_type == "element":
            return self.n_elems
        return self.n_voronoi

    def _empty_stacked(
        self,
        *,
        domain_type: Literal["node", "element", "voronoi"],
        value_type: Literal["scalar", "vector", "tensor"],
    ) -> np.ndarray:
        domain = self._domain_shape(domain_type)
        if value_type == "scalar":
            shape: tuple[int, ...] = (self.n_rods, domain)
        elif value_type == "vector":
            shape = (self.n_rods, 3, domain)
        else:
            shape = (self.n_rods, 3, 3, domain)
        return np.zeros(shape, dtype=self._device_dtype)

    def _stack_from_systems(self, systems: list[RodType], attr_name: str) -> np.ndarray:
        return np.stack(
            [
                np.asarray(system.__dict__[attr_name], dtype=self._device_dtype)
                for system in systems
            ],
            axis=0,
        )

    def _allocate_stacked_attributes(self, systems: list[RodType]) -> None:
        attr_specs: tuple[
            tuple[
                str,
                Literal["node", "element", "voronoi"],
                Literal["scalar", "vector", "tensor"],
            ],
            ...,
        ] = (
            ("mass", "node", "scalar"),
            ("position_collection", "node", "vector"),
            ("internal_forces", "node", "vector"),
            ("external_forces", "node", "vector"),
            ("velocity_collection", "node", "vector"),
            ("acceleration_collection", "node", "vector"),
            ("radius", "element", "scalar"),
            ("volume", "element", "scalar"),
            ("density", "element", "scalar"),
            ("lengths", "element", "scalar"),
            ("rest_lengths", "element", "scalar"),
            ("dilatation", "element", "scalar"),
            ("dilatation_rate", "element", "scalar"),
            ("tangents", "element", "vector"),
            ("sigma", "element", "vector"),
            ("rest_sigma", "element", "vector"),
            ("internal_torques", "element", "vector"),
            ("external_torques", "element", "vector"),
            ("internal_stress", "element", "vector"),
            ("omega_collection", "element", "vector"),
            ("alpha_collection", "element", "vector"),
            ("director_collection", "element", "tensor"),
            ("mass_second_moment_of_inertia", "element", "tensor"),
            ("inv_mass_second_moment_of_inertia", "element", "tensor"),
            ("shear_matrix", "element", "tensor"),
            ("voronoi_dilatation", "voronoi", "scalar"),
            ("rest_voronoi_lengths", "voronoi", "scalar"),
            ("kappa", "voronoi", "vector"),
            ("rest_kappa", "voronoi", "vector"),
            ("internal_couple", "voronoi", "vector"),
            ("bend_matrix", "voronoi", "tensor"),
        )
        for attr_name, domain_type, value_type in attr_specs:
            stacked = self._stack_from_systems(systems, attr_name)
            expected = self._empty_stacked(
                domain_type=domain_type, value_type=value_type
            ).shape
            assert stacked.shape == expected, (
                f"{attr_name} stacked shape {stacked.shape} != expected {expected}."
            )
            self.__dict__[attr_name] = stacked

    def _recompute_sigma_kappa(self) -> None:
        for rod_idx in range(self.n_rods):
            _compute_shear_stretch_strains(
                self.position_collection[rod_idx],
                self.volume[rod_idx],
                self.lengths[rod_idx],
                self.tangents[rod_idx],
                self.radius[rod_idx],
                self.rest_lengths[rod_idx],
                self.rest_voronoi_lengths[rod_idx],
                self.dilatation[rod_idx],
                self.voronoi_dilatation[rod_idx],
                self.director_collection[rod_idx],
                self.sigma[rod_idx],
            )
            _compute_bending_twist_strains(
                self.director_collection[rod_idx],
                self.rest_voronoi_lengths[rod_idx],
                self.kappa[rod_idx],
            )

    def _normalize_attr_names(
        self, variables: Iterable[str] | None = None
    ) -> tuple[str, ...]:
        if variables is None:
            return _SYNCABLE_ATTRS
        return tuple(dict.fromkeys(variables))

    def _validate_sync_variables(self, variables: Sequence[str]) -> None:
        for variable in variables:
            assert variable in _SYNCABLE_ATTRS, (
                f"Unsupported sync attribute {variable!r}"
            )
            assert hasattr(self, variable), f"Missing host attribute {variable!r}"
            assert variable in self._device_state, (
                f"Missing device attribute {variable!r}"
            )

    def _normalize_rod_sync_target(self, rods: RodSyncTarget) -> tuple[RodType, ...]:
        assert hasattr(self, "_systems"), (
            "Block must be built before synchronizing with rods."
        )
        if rods == "all":
            return self._systems
        if isinstance(rods, (list, tuple)):
            return tuple(rods)
        single_rod: RodType = rods  # type: ignore[assignment]
        return (single_rod,)

    def _resolve_system_indices(self, rods: Sequence[RodType]) -> list[int]:
        indices: list[int] = []
        for rod in rods:
            try:
                indices.append(self._systems.index(rod))
            except ValueError as exc:
                raise ValueError(
                    f"Rod {rod!r} was not packed into this block."
                ) from exc
        return indices

    def _pull_rod_state_to_block(
        self,
        variables: Sequence[str],
        system_indices: Sequence[int],
    ) -> None:
        for system_idx in system_indices:
            system = self._systems[system_idx]
            for variable in variables:
                np.copyto(
                    getattr(self, variable)[system_idx],
                    system.__dict__[variable],
                )

    def _push_block_state_to_rods(
        self,
        variables: Sequence[str],
        system_indices: Sequence[int],
    ) -> None:
        for system_idx in system_indices:
            system = self._systems[system_idx]
            for variable in variables:
                np.copyto(
                    system.__dict__[variable],
                    getattr(self, variable)[system_idx],
                )

    def _initialize_device_state(
        self,
        attrs: Iterable[str] | None = None,
        *,
        device: jax.Device | None = None,
    ) -> None:
        target_device = device if device is not None else self._initial_device
        for attr in self._normalize_attr_names(attrs):
            host_array = np.asarray(getattr(self, attr), dtype=self._device_dtype)
            self._device_state[attr] = jax.device_put(host_array, device=target_device)
        self._device_platform = target_device.platform
        self._refresh_device_views()
        self._device_dirty = False

    def to_device(
        self,
        rods: RodSyncTarget = "all",
        *,
        variables: Iterable[str] | None = None,
    ) -> None:
        """Copy host rod state into the block and upload selected fields."""
        sync_variables = self._normalize_attr_names(variables)
        self._validate_sync_variables(sync_variables)
        system_indices = self._resolve_system_indices(
            self._normalize_rod_sync_target(rods)
        )
        self._pull_rod_state_to_block(sync_variables, system_indices)
        for variable in sync_variables:
            host_array = np.asarray(getattr(self, variable), dtype=self._device_dtype)
            self._device_state[variable] = jax.device_put(
                host_array,
                device=self._initial_device,
            )
        self._refresh_device_views()
        self._device_dirty = False

    def from_device(
        self,
        rods: RodSyncTarget = "all",
        *,
        variables: Iterable[str] | None = None,
        update_rods: bool = True,
    ) -> None:
        """Copy selected fields from device to host block memory and rods."""
        sync_variables = self._normalize_attr_names(variables)
        self._validate_sync_variables(sync_variables)
        for variable in sync_variables:
            np.copyto(
                getattr(self, variable),
                np.asarray(self._device_state[variable]),
            )
        if update_rods:
            system_indices = self._resolve_system_indices(
                self._normalize_rod_sync_target(rods)
            )
            self._push_block_state_to_rods(sync_variables, system_indices)
        self._device_dirty = False

    def _refresh_device_views(self) -> None:
        self.position_collection_device = self._device_state["position_collection"]
        self.director_collection_device = self._device_state["director_collection"]
        self.velocity_collection_device = self._device_state["velocity_collection"]
        self.omega_collection_device = self._device_state["omega_collection"]
        self.acceleration_collection_device = self._device_state[
            "acceleration_collection"
        ]
        self.alpha_collection_device = self._device_state["alpha_collection"]

    def jax_get_state(self) -> dict[str, jax.Array]:
        return dict(self._device_state)

    def jax_set_state(self, state: dict[str, jax.Array]) -> None:
        self._device_state = dict(state)
        self._refresh_device_views()
        self._device_dirty = True

    def iterate_rods(self) -> Iterator[JAXRodStackedView]:
        """Yield a rod-local view of each stacked rod's device state."""
        state = self._device_state
        for rod_idx in range(self.n_rods):
            yield JAXRodStackedView(state, rod_idx)

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jax.Array, ...],
        tuple[_CosseratRodVerticalMemoryBlock, tuple[str, ...]],
    ]:
        device_state = getattr(self, "_device_state", None)
        if not device_state:
            return (), (self, ())
        keys = tuple(sorted(device_state.keys()))
        return tuple(device_state[key] for key in keys), (self, keys)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[_CosseratRodVerticalMemoryBlock, tuple[str, ...]],
        children: tuple[jax.Array, ...],
    ) -> _CosseratRodVerticalMemoryBlock:
        block, keys = aux_data
        if keys:
            block.jax_set_state(dict(zip(keys, children, strict=True)))
        return block

    def jax_kinematic_step(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
        prefac: np.float64,
    ) -> dict[str, jax.Array]:
        position_collection, director_collection = _jax_update_kinematics_vmap(
            state["position_collection"],
            state["director_collection"],
            state["velocity_collection"],
            state["omega_collection"],
            self._device_dtype.type(prefac),
        )
        updated = dict(state)
        updated["position_collection"] = position_collection
        updated["director_collection"] = director_collection
        return updated

    def jax_compute_internal_forces_and_torques(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        (
            lengths,
            tangents,
            radius,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            sigma,
            kappa,
            internal_stress,
            internal_couple,
            internal_forces,
            internal_torques,
        ) = _jax_compute_internal_forces_and_torques_vmap(
            state["position_collection"],
            state["velocity_collection"],
            state["volume"],
            state["rest_lengths"],
            state["rest_voronoi_lengths"],
            state["director_collection"],
            state["rest_sigma"],
            state["shear_matrix"],
            state["mass_second_moment_of_inertia"],
            state["omega_collection"],
            state["bend_matrix"],
            state["rest_kappa"],
        )
        updated = dict(state)
        updated["lengths"] = lengths
        updated["tangents"] = tangents
        updated["radius"] = radius
        updated["dilatation"] = dilatation
        updated["dilatation_rate"] = dilatation_rate
        updated["voronoi_dilatation"] = voronoi_dilatation
        updated["sigma"] = sigma
        updated["kappa"] = kappa
        updated["internal_stress"] = internal_stress
        updated["internal_couple"] = internal_couple
        updated["internal_forces"] = internal_forces
        updated["internal_torques"] = internal_torques
        return updated

    def jax_dynamic_step(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
        dt: np.float64,
    ) -> dict[str, jax.Array]:
        acceleration_collection, alpha_collection = _jax_update_accelerations_vmap(
            state["internal_forces"],
            state["external_forces"],
            state["mass"],
            state["inv_mass_second_moment_of_inertia"],
            state["internal_torques"],
            state["external_torques"],
            state["dilatation"],
        )
        velocity_collection, omega_collection = _jax_update_dynamics_vmap(
            state["velocity_collection"],
            state["omega_collection"],
            acceleration_collection,
            alpha_collection,
            self._device_dtype.type(dt),
        )
        updated = dict(state)
        updated["acceleration_collection"] = acceleration_collection
        updated["alpha_collection"] = alpha_collection
        updated["velocity_collection"] = velocity_collection
        updated["omega_collection"] = omega_collection
        return updated

    def jax_zero_external_loads(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        external_forces, external_torques = _jax_zero_external_loads_vmap(
            state["external_forces"],
            state["external_torques"],
        )
        updated = dict(state)
        updated["external_forces"] = external_forces
        updated["external_torques"] = external_torques
        return updated
