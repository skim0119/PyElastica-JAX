"""Unit tests for the Hertzian contact law and Coulomb friction kernel."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from elastica_jax.contact.kernels import contact_force, slip_ramp

_STIFFNESS = 1.0e6
_DAMPING = 1.0e3


def test_slip_ramp_regimes() -> None:
    threshold = 1.0e-3
    speeds = jnp.array([0.0, 0.5e-3, 1.0e-3, 1.25e-3, 1.5e-3, 2.0e-3])
    values = np.asarray(slip_ramp(speeds, threshold))
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.0)
    assert values[-2] == pytest.approx(0.0)
    assert np.all(np.diff(values) <= 1.0e-12)


def test_no_force_without_overlap() -> None:
    distance = jnp.array([0.05])
    normal = jnp.array([[1.0, 0.0, 0.0]])
    velocity = jnp.zeros((1, 3))
    force = contact_force(
        distance,
        normal,
        velocity,
        contact_stiffness=_STIFFNESS,
        contact_damping=_DAMPING,
        hertzian=True,
    )
    np.testing.assert_allclose(np.asarray(force), 0.0)


def test_hertzian_elastic_magnitude() -> None:
    penetration = 0.1
    distance = jnp.array([-penetration])
    normal = jnp.array([[1.0, 0.0, 0.0]])
    velocity = jnp.zeros((1, 3))
    force = contact_force(
        distance,
        normal,
        velocity,
        contact_stiffness=_STIFFNESS,
        contact_damping=_DAMPING,
        hertzian=True,
    )
    expected = _STIFFNESS * penetration**1.5
    np.testing.assert_allclose(np.asarray(force)[0], [expected, 0.0, 0.0], rtol=1e-10)


def test_linear_default_matches_half_scaled_law() -> None:
    penetration = 0.1
    distance = jnp.array([-penetration])
    normal = jnp.array([[1.0, 0.0, 0.0]])
    velocity = jnp.zeros((1, 3))
    force = contact_force(
        distance,
        normal,
        velocity,
        contact_stiffness=_STIFFNESS,
        contact_damping=_DAMPING,
    )
    expected = 0.5 * _STIFFNESS * penetration
    np.testing.assert_allclose(np.asarray(force)[0], [expected, 0.0, 0.0], rtol=1e-10)


def test_kinetic_friction_opposes_slip_and_gates() -> None:
    penetration = 0.1
    distance = jnp.array([-penetration])
    normal = jnp.array([[1.0, 0.0, 0.0]])
    # Tangential slip of the first body along +y (no normal closing speed).
    velocity = jnp.array([[0.0, 1.0, 0.0]])
    friction_coefficient = 0.4
    normal_magnitude = _STIFFNESS * penetration**1.5

    active = contact_force(
        distance,
        normal,
        velocity,
        contact_stiffness=_STIFFNESS,
        contact_damping=_DAMPING,
        hertzian=True,
        friction_coefficient=friction_coefficient,
        static_velocity_threshold=1.0e-6,
        friction_gate=1.0,
    )
    result = np.asarray(active)[0]
    np.testing.assert_allclose(result[0], normal_magnitude, rtol=1e-10)
    np.testing.assert_allclose(
        result[1], -friction_coefficient * normal_magnitude, rtol=1e-10
    )

    gated = contact_force(
        distance,
        normal,
        velocity,
        contact_stiffness=_STIFFNESS,
        contact_damping=_DAMPING,
        hertzian=True,
        friction_coefficient=friction_coefficient,
        static_velocity_threshold=1.0e-6,
        friction_gate=0.0,
    )
    np.testing.assert_allclose(np.asarray(gated)[0, 1], 0.0)
