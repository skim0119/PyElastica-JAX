"""PyElastica-style rod-rod contact via ``pairwise_interaction``."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from elastica_jax.contact.rod_rod_kernels import apply_rod_rod_contact_forces
from elastica_jax.memory_block.memory_block_rod_jax import JAXRodView
from elastica_jax.operations import JAXTime
from elastica_jax.rod_rod_operation import NoRodRodBlockOpJax


class RodRodContactJax(NoRodRodBlockOpJax):
    """JAX port of ``elastica.RodRodContact`` for one registered rod pair.

    Register one instance per unordered rod pair, matching PyElastica's
    ``detect_contact_between(first_rod, second_rod).using(RodRodContact, ...)``.

    Parameters
    ----------
    k : float
        Contact spring constant.
    nu : float
        Contact damping constant.
    """

    def __init__(
        self,
        k: float,
        nu: float,
        *,
        _first_system: Any = None,
        _second_system: Any = None,
    ) -> None:
        self.contact_k = k
        self.contact_nu = nu

    def jax_operation(
        self,
        rod_one_view: JAXRodView,
        rod_two_view: JAXRodView,
        time: JAXTime,
    ) -> tuple[JAXRodView, JAXRodView]:
        del time
        updated_one, updated_two = apply_rod_rod_contact_forces(
            x_one=rod_one_view.position_collection[:, :-1],
            radius_one=rod_one_view.radius,
            length_one=rod_one_view.lengths,
            tangent_one=rod_one_view.tangents,
            velocity_one=rod_one_view.velocity_collection,
            internal_forces_one=rod_one_view.internal_forces,
            external_forces_one=rod_one_view.external_forces,
            x_two=rod_two_view.position_collection[:, :-1],
            radius_two=rod_two_view.radius,
            length_two=rod_two_view.lengths,
            tangent_two=rod_two_view.tangents,
            velocity_two=rod_two_view.velocity_collection,
            internal_forces_two=rod_two_view.internal_forces,
            external_forces_two=rod_two_view.external_forces,
            contact_k=jnp.asarray(self.contact_k),
            contact_nu=jnp.asarray(self.contact_nu),
        )
        rod_one_view.external_forces = updated_one
        rod_two_view.external_forces = updated_two
        return rod_one_view, rod_two_view
