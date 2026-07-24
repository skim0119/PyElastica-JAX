"""Protocols for Cosserat rod memory-block factories."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Iterator, Protocol, runtime_checkable

import jax
import numpy as np

from elastica.typing import RodType, SystemIdxType

type JAXBlockState = dict[str, jax.Array]


class RodViewMetadata(Protocol):
    def slice_for_attr(self, attr: str) -> slice: ...


class RodView(Protocol):
    """Rod-shaped projection passed to ``map_rods`` operators."""

    def __getattr__(self, attr: str) -> jax.Array: ...
    def __setattr__(self, attr: str, value: jax.Array) -> None: ...
    def commit(self) -> JAXBlockState: ...


type RodLocalOp = Callable[[RodView, np.float64 | float], RodView | JAXBlockState]


class PackedRodIndexLayout(Protocol):
    """Per-rod index spans for packed (horizontal) Block gather/scatter."""

    start_idx_in_rod_nodes: np.ndarray
    end_idx_in_rod_nodes: np.ndarray
    start_idx_in_rod_elems: np.ndarray
    end_idx_in_rod_elems: np.ndarray
    start_idx_in_rod_voronoi: np.ndarray
    end_idx_in_rod_voronoi: np.ndarray


@runtime_checkable
class RodBlockProtocol(Protocol):
    """
    Wide Block face: pack, sync, stepper state, and Rod-local projection.

    Returned by ``configure_rod_block*``. PyElastica builds the block by
    calling the instance as ``block(systems, system_idx_list)`` during
    ``finalize()``. The same object then appears in ``final_systems()``.
    Position Verlet only invokes the pure ``jax_*`` methods on this face.
    """

    def __init__(
        self,
        *,
        device: jax.Device | Sequence[jax.Device],
        device_dtype: np.dtype,
    ) -> None:
        """Configure device placement before systems are packed."""
        ...

    def __call__(
        self,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> RodBlockProtocol:
        """Pack ``systems`` into this block and return ``self``."""
        ...

    def iterate_rods(self) -> Iterator[RodView]:
        """Iterate over Rod-local views of device Block state."""
        ...

    def rods(self) -> Sequence[RodType]:
        """Return host rods packed into this Block, in block order.

        Returns
        -------
        Sequence[RodType]
            Packed host rods in block order.
        """
        ...

    def to_device(
        self,
        rods: object = "all",
        *,
        variables: Iterable[str] | None = None,
    ) -> None:
        """Copy host rod state into the Block and upload fields to device."""
        ...

    def from_device(
        self,
        rods: object = "all",
        *,
        variables: Iterable[str] | None = None,
        update_rods: bool = True,
    ) -> None:
        """Copy selected fields from device to host Block memory and rods."""
        ...

    def map_rods(
        self,
        state: JAXBlockState,
        op: RodLocalOp | Sequence[RodLocalOp],
        time: np.float64 | float,
    ) -> JAXBlockState:
        """Project to Rod-local state, apply ``op``, write back Block state.

        Parameters
        ----------
        state :
            Authoritative Block state.
        op :
            One shared Rod-local operator, or one operator per rod.
        time :
            Simulation time passed through to ``op``.

        Returns
        -------
        JAXBlockState
            Updated Block state after rod-local projection.
        """
        ...

    def jax_get_state(self) -> JAXBlockState:
        """Return authoritative device Block state for Position Verlet."""
        ...

    def jax_set_state(self, state: JAXBlockState) -> None:
        """Replace authoritative device Block state after rollout."""
        ...

    def device_put(self, value: object) -> object:
        """Place a value on this Block's execution device."""
        ...

    def jax_kinematic_step(
        self,
        state: JAXBlockState,
        time: object,
        prefac: object,
    ) -> JAXBlockState:
        """Advance kinematics for one Position Verlet half-step."""
        ...

    def jax_dynamic_step(
        self,
        state: JAXBlockState,
        time: object,
        dt: object,
    ) -> JAXBlockState:
        """Advance dynamics for one Position Verlet step."""
        ...

    def jax_compute_internal_forces_and_torques(
        self,
        state: JAXBlockState,
        time: object,
    ) -> JAXBlockState:
        """Compute internal forces and torques on Block state."""
        ...

    def jax_zero_external_loads(
        self,
        state: JAXBlockState,
        time: object,
    ) -> JAXBlockState:
        """Clear external forces and torques on Block state."""
        ...
