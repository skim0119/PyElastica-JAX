from __future__ import annotations
from typing import Any, Protocol


JAXPyTree = Any


class JAXBlock(Protocol):
    """
    Minimal block interface required for a JAX-owned timestep loop.

    Notes
    -----
    All ``jax_*`` methods below must behave as pure transforms on PyTree state.
    They must not mutate Python object fields when used inside a JAX loop.
    The only host-side mutation point is ``jax_set_state(...)`` after rollout.
    """

    def jax_get_state(self) -> JAXPyTree: ...

    def jax_set_state(self, state: JAXPyTree) -> None: ...

    def jax_kinematic_step(
        self,
        state: JAXPyTree,
        time: np.float64,
        prefac: np.float64,
    ) -> JAXPyTree: ...

    def jax_dynamic_step(
        self,
        state: JAXPyTree,
        time: np.float64,
        dt: np.float64,
    ) -> JAXPyTree: ...

    def jax_compute_internal_forces_and_torques(
        self,
        state: JAXPyTree,
        time: np.float64,
    ) -> JAXPyTree: ...

    def jax_zero_external_loads(
        self,
        state: JAXPyTree,
        time: np.float64,
    ) -> JAXPyTree: ...


class JAXSystems(Protocol):
    """
    Minimal collection interface required for a JAX-owned timestep loop.

    Notes
    -----
    Constraint/forcing/contact/damping phases must also be exposed as pure state
    transforms. Host callbacks are intentionally excluded from the loop contract.
    """

    def final_systems(self) -> Iterable[JAXBlock]: ...

    def jax_constrain_values(
        self,
        states: tuple[JAXPyTree, ...],
        time: np.float64,
    ) -> tuple[JAXPyTree, ...]: ...

    def jax_synchronize(
        self,
        states: tuple[JAXPyTree, ...],
        time: np.float64,
    ) -> tuple[JAXPyTree, ...]: ...

    def jax_constrain_rates(
        self,
        states: tuple[JAXPyTree, ...],
        time: np.float64,
    ) -> tuple[JAXPyTree, ...]: ...
