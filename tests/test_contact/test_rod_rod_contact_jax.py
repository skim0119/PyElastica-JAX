from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import elastica as ea
import elastica_jax as eaj
from elastica_jax.contact.rod_rod_kernels import apply_rod_rod_contact_forces


def _straight_rod(*, start: np.ndarray) -> ea.CosseratRod:
    return ea.CosseratRod.straight_rod(
        4,
        start,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        0.35,
        0.01,
        1000.0,
        youngs_modulus=1.0e5,
    )


class _PairwiseContactSimulator(ea.BaseSystemCollection, eaj.JAXInteraction):
    pass


def test_rod_rod_contact_jax_updates_external_forces() -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        simulator = _PairwiseContactSimulator()
        rod_block = eaj.configure_rod_block()
        simulator.enable_block_supports(ea.CosseratRod, rod_block)

        rod_one = _straight_rod(start=np.zeros(3))
        rod_two = _straight_rod(start=np.array([0.02, 0.0, 0.0]))
        simulator.append(rod_one)
        simulator.append(rod_two)
        simulator.pairwise_interaction(rod_one, rod_two).using(
            eaj.RodRodContactJax,
            k=1.0e3,
            nu=1.0e-3,
        )
        simulator.finalize()

        block = tuple(simulator.final_systems())[0]
        state = block.jax_get_state()
        state = block.jax_compute_internal_forces_and_torques(state, np.float64(0.0))
        updated_states = simulator.jax_synchronize((state,), np.float64(0.0))
        updated_state = updated_states[0]

        first_node = int(block.start_idx_in_rod_nodes[0])
        second_node = int(block.start_idx_in_rod_nodes[1])
        force_norm = float(
            np.linalg.norm(updated_state["external_forces"][:, first_node])
            + np.linalg.norm(updated_state["external_forces"][:, second_node])
        )
        assert force_norm > 0.0, "Pairwise rod-rod contact did not apply forces."


def test_apply_rod_rod_contact_forces_matches_pyelastica_kernel() -> None:
    rod_one = _straight_rod(start=np.zeros(3))
    rod_two = _straight_rod(start=np.array([0.02, 0.0, 0.0]))

    ext_one = rod_one.external_forces.copy()
    ext_two = rod_two.external_forces.copy()
    ea.RodRodContact(k=1.0e3, nu=1.0e-3).apply_contact(rod_one, rod_two)

    jax_ext_one, jax_ext_two = apply_rod_rod_contact_forces(
        x_one=jnp.asarray(rod_one.position_collection[:, :-1]),
        radius_one=jnp.asarray(rod_one.radius),
        length_one=jnp.asarray(rod_one.lengths),
        tangent_one=jnp.asarray(rod_one.tangents),
        velocity_one=jnp.asarray(rod_one.velocity_collection),
        internal_forces_one=jnp.asarray(rod_one.internal_forces),
        external_forces_one=jnp.asarray(ext_one),
        x_two=jnp.asarray(rod_two.position_collection[:, :-1]),
        radius_two=jnp.asarray(rod_two.radius),
        length_two=jnp.asarray(rod_two.lengths),
        tangent_two=jnp.asarray(rod_two.tangents),
        velocity_two=jnp.asarray(rod_two.velocity_collection),
        internal_forces_two=jnp.asarray(rod_two.internal_forces),
        external_forces_two=jnp.asarray(ext_two),
        contact_k=jnp.asarray(1.0e3),
        contact_nu=jnp.asarray(1.0e-3),
    )

    assert np.allclose(
        np.asarray(jax_ext_one),
        rod_one.external_forces,
        rtol=1.0e-10,
        atol=1.0e-10,
    ), "JAX rod-rod contact forces on rod one differ from PyElastica."
    assert np.allclose(
        np.asarray(jax_ext_two),
        rod_two.external_forces,
        rtol=1.0e-10,
        atol=1.0e-10,
    ), "JAX rod-rod contact forces on rod two differ from PyElastica."
