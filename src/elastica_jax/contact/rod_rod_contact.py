"""PyElastica-style rod-rod contact via ``pairwise_interaction``."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from elastica_jax.contact.rod_rod_kernels import apply_rod_rod_contact_forces
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
        _first_system=None,
        _second_system=None,
    ) -> None:
        del _first_system, _second_system
        self.contact_k = jnp.asarray(k, dtype=jnp.float64)
        self.contact_nu = jnp.asarray(nu, dtype=jnp.float64)

    def jax_operation(self, rod_one_view, rod_two_view, time):
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
            contact_k=self.contact_k,
            contact_nu=self.contact_nu,
        )
        rod_one_view.external_forces = updated_one
        rod_two_view.external_forces = updated_two
        return rod_one_view, rod_two_view
