"""Geometry and response kernels for capsule contact."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

CONTACT_THRESHOLD = 1.0e-8


def closest_points_on_segments(
    p0: jax.Array,
    p1: jax.Array,
    q0: jax.Array,
    q1: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Vectorized closest points for equally shaped ``(..., 3)`` segments."""
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = jnp.sum(u * u, axis=-1)
    b = jnp.sum(u * v, axis=-1)
    c = jnp.sum(v * v, axis=-1)
    d = jnp.sum(u * w, axis=-1)
    e = jnp.sum(v * w, axis=-1)
    denominator = a * c - b * b
    safe = jnp.maximum(denominator, CONTACT_THRESHOLD)
    s = jnp.where(denominator > CONTACT_THRESHOLD, (b * e - c * d) / safe, 0.0)
    s = jnp.clip(s, 0.0, 1.0)
    t = jnp.clip((b * s + e) / jnp.maximum(c, CONTACT_THRESHOLD), 0.0, 1.0)
    s = jnp.clip((b * t - d) / jnp.maximum(a, CONTACT_THRESHOLD), 0.0, 1.0)
    return p0 + s[..., None] * u, q0 + t[..., None] * v


def scatter_element_loads(
    external_forces: jax.Array,
    external_torques: jax.Array,
    element_indices: jax.Array,
    force: jax.Array,
    torque_world: jax.Array,
    directors: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    nodal_force = 0.5 * force
    external_forces = external_forces.at[:, element_indices].add(nodal_force.T)
    external_forces = external_forces.at[:, element_indices + 1].add(nodal_force.T)
    torque_material = jnp.einsum("nij,nj->ni", directors, torque_world)
    external_torques = external_torques.at[:, element_indices].add(torque_material.T)
    return external_forces, external_torques


def slip_ramp(speed: jax.Array, threshold: float | jax.Array) -> jax.Array:
    """Static-to-kinetic blend factor (the C++ ``_linear`` function).

    Returns ``1`` for ``speed <= threshold`` (fully static, no kinetic
    friction), ramps linearly to ``0`` at ``1.5 * threshold``, and stays ``0``
    for faster slip (fully kinetic).

    Parameters
    ----------
    speed : jax.Array
        Tangential slip speed magnitude.
    threshold : float | jax.Array
        Static velocity threshold ``v_static``.

    Returns
    -------
    jax.Array
        Blend factor in ``[0, 1]`` with the same shape as ``speed``.
    """
    speed = jnp.abs(speed)
    threshold = jnp.abs(threshold)
    width = 0.5 * threshold
    ramp = jnp.abs(1.0 - jnp.minimum(1.0, (speed - threshold) / width))
    factor = jnp.where(speed > threshold, ramp, 1.0)
    return jnp.where(speed > threshold + width, 0.0, factor)


def contact_force(
    distance: jax.Array,
    normal: jax.Array,
    relative_velocity: jax.Array,
    *,
    contact_stiffness: float | jax.Array,
    contact_damping: float | jax.Array,
    hertzian: bool = False,
    friction_coefficient: float = 0.0,
    static_velocity_threshold: float = 1.0,
    friction_gate: float | jax.Array = 1.0,
) -> jax.Array:
    """Normal contact response with optional Hertzian law and Coulomb friction.

    Parameters
    ----------
    distance : jax.Array
        Signed surface gap; negative under overlap (penetration ``= -distance``).
    normal : jax.Array
        Unit contact normal pointing toward the first body, shape ``(..., 3)``.
    relative_velocity : jax.Array
        Velocity of the first body relative to the second, shape ``(..., 3)``.
    contact_stiffness, contact_damping : float | jax.Array
        Normal elastic and viscous coefficients.
    hertzian : bool, optional
        If ``True`` use the ``gamma^1.5`` elastic / ``gamma^0.5`` damping law and
        the full (unscaled, unclamped) coefficients, matching the C++ nest
        contact. If ``False`` (default) use the linear ``0.5`` scaled and
        non-negative-clamped law, preserving the original behaviour.
    friction_coefficient : float, optional
        Isotropic kinetic Coulomb coefficient. ``0`` (default) disables friction.
    static_velocity_threshold : float, optional
        Slip speed below which kinetic friction is suppressed (see ``slip_ramp``).
    friction_gate : float | jax.Array, optional
        Multiplicative gate (``0`` or ``1``) used to activate friction after a
        given simulation time.

    Returns
    -------
    jax.Array
        Contact force on the first body, shape ``(..., 3)``.
    """
    normal_speed = -jnp.sum(relative_velocity * normal, axis=-1)
    penetration = -distance
    active = distance < -CONTACT_THRESHOLD
    if hertzian:
        gamma_sqrt = jnp.sqrt(jnp.maximum(penetration, 0.0))
        magnitude = (
            contact_stiffness * penetration + contact_damping * normal_speed
        ) * gamma_sqrt
    else:
        magnitude = jnp.maximum(
            0.0,
            0.5 * contact_stiffness * penetration
            + 0.5 * contact_damping * normal_speed,
        )
    normal_force = magnitude[..., None] * normal

    tangent_velocity = -relative_velocity - normal_speed[..., None] * normal
    tangent_speed = jnp.linalg.norm(tangent_velocity, axis=-1)
    tangent = tangent_velocity / jnp.maximum(
        tangent_speed[..., None], CONTACT_THRESHOLD
    )
    if friction_coefficient:
        blend = slip_ramp(tangent_speed, static_velocity_threshold)
        friction_magnitude = (
            friction_coefficient * jnp.abs(magnitude) * (1.0 - blend) * friction_gate
        )
        tangential = friction_magnitude[..., None] * tangent
    else:
        tangent_magnitude = jnp.minimum(
            0.5 * contact_damping * tangent_speed, 1.0e-10 * magnitude
        )
        tangential = tangent_magnitude[..., None] * tangent
    return jnp.where(active[..., None], normal_force + tangential, 0.0)


def apply_capsule_pair_forces(
    *,
    pair_first: jax.Array,
    pair_second: jax.Array,
    pair_active: jax.Array,
    centers: jax.Array,
    velocities: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    omega: jax.Array,
    directors: jax.Array,
    block_element_indices: jax.Array,
    external_forces: jax.Array,
    external_torques: jax.Array,
    contact_stiffness: float | jax.Array,
    contact_damping: float | jax.Array,
    cached_candidates: jax.Array,
    last_detection_time: jax.Array,
    time: float | jax.Array,
    steps_between_detection: int | jax.Array,
    time_step: float | jax.Array,
    hertzian: bool = False,
    friction_coefficient: float = 0.0,
    static_velocity_threshold: float = 1.0,
    friction_gate: float | jax.Array = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fine-phase capsule contact on a bounded active pair list."""
    first = pair_first
    second = pair_second
    active_pair = pair_active & (first >= 0) & (second >= 0)
    first = jnp.where(active_pair, first, 0)
    second = jnp.where(active_pair, second, 0)

    c1, c2 = centers[first], centers[second]
    a1, a2 = axes[first], axes[second]
    half1, half2 = 0.5 * lengths[first], 0.5 * lengths[second]
    p1, p2 = closest_points_on_segments(
        c1 - half1[:, None] * a1,
        c1 + half1[:, None] * a1,
        c2 - half2[:, None] * a2,
        c2 + half2[:, None] * a2,
    )
    delta = p1 - p2
    axis_distance = jnp.linalg.norm(delta, axis=-1)
    normal = delta / jnp.maximum(axis_distance[:, None], CONTACT_THRESHOLD)
    distance = axis_distance - radii[first] - radii[second]

    detection_interval = steps_between_detection * time_step
    detection_due = (detection_interval == 0.0) | (
        time - last_detection_time >= detection_interval
    )
    broad_phase = jnp.where(
        detection_due,
        active_pair,
        cached_candidates & active_pair,
    )
    last_detection_time = jnp.where(detection_due, time, last_detection_time)
    distance = jnp.where(broad_phase, distance, 1.0)
    contact = p2 + (radii[second] + 0.5 * distance)[:, None] * normal
    arm1, arm2 = contact - c1, contact - c2
    v1 = velocities[first] + jnp.cross(omega[first], arm1)
    v2 = velocities[second] + jnp.cross(omega[second], arm2)
    force = contact_force(
        distance,
        normal,
        v1 - v2,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
        hertzian=hertzian,
        friction_coefficient=friction_coefficient,
        static_velocity_threshold=static_velocity_threshold,
        friction_gate=friction_gate,
    )
    force = jnp.where((axis_distance > CONTACT_THRESHOLD)[:, None], force, 0.0)
    parallel_overlap = (
        jnp.abs(jnp.sum(a1 * a2, axis=-1)) > 1.0 - CONTACT_THRESHOLD
    ) & (jnp.abs(jnp.sum((c2 - c1) * a1, axis=-1)) < lengths[first])
    force *= jnp.where(parallel_overlap, 2.0, 1.0)[:, None]
    force = jnp.where(broad_phase[:, None], force, 0.0)

    f1 = jnp.zeros_like(centers).at[first].add(force)
    f2 = jnp.zeros_like(centers).at[second].add(-force)
    t1 = jnp.zeros_like(centers).at[first].add(jnp.cross(arm1, force))
    t2 = jnp.zeros_like(centers).at[second].add(jnp.cross(arm2, -force))
    total_force, total_torque = f1 + f2, t1 + t2
    return (
        *scatter_element_loads(
            external_forces,
            external_torques,
            block_element_indices,
            total_force,
            total_torque,
            directors,
        ),
        broad_phase,
        last_detection_time,
    )


def apply_wall_contacts(
    *,
    wall_origins: jax.Array | np.ndarray,
    wall_normals: jax.Array | np.ndarray,
    centers: jax.Array,
    velocities: jax.Array,
    axes: jax.Array,
    lengths: jax.Array,
    radii: jax.Array,
    omega: jax.Array,
    directors: jax.Array,
    block_element_indices: jax.Array,
    external_forces: jax.Array,
    external_torques: jax.Array,
    contact_stiffness: float | jax.Array,
    contact_damping: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    origins = jnp.asarray(wall_origins, dtype=centers.dtype)
    normals = jnp.asarray(wall_normals, dtype=centers.dtype)
    cosine = jnp.einsum("ni,wi->nw", axes, normals)
    sign = jnp.where(cosine > 0.0, -1.0, 1.0)
    closest = centers[:, None, :] + sign[..., None] * (
        0.5 * lengths[:, None, None] * axes[:, None, :]
    )
    parallel = jnp.abs(cosine) < CONTACT_THRESHOLD
    closest = jnp.where(parallel[..., None], centers[:, None, :], closest)
    distance = jnp.einsum("nwi,wi->nw", closest - origins[None, :, :], normals)
    distance -= radii[:, None]
    normal = jnp.broadcast_to(normals[None, :, :], closest.shape)
    contact = closest - (radii[:, None] + 0.5 * distance)[..., None] * normal
    arm = contact - centers[:, None, :]
    contact_velocity = velocities[:, None, :] + jnp.cross(omega[:, None, :], arm)
    force = contact_force(
        distance,
        normal,
        contact_velocity,
        contact_stiffness=contact_stiffness,
        contact_damping=contact_damping,
    )
    total_force = jnp.sum(force, axis=1)
    total_torque = jnp.sum(jnp.cross(arm, force), axis=1)
    return scatter_element_loads(
        external_forces,
        external_torques,
        block_element_indices,
        total_force,
        total_torque,
        directors,
    )
