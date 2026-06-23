"""Shared capsule contact kernels for active-matter GPU examples."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

CONTACT_THRESHOLD = 1.0e-8


def closest_points_on_segments(p0, p1, q0, q1):
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
    external_forces,
    external_torques,
    element_indices,
    force,
    torque_world,
    directors,
):
    nodal_force = 0.5 * force
    external_forces = external_forces.at[:, element_indices].add(nodal_force.T)
    external_forces = external_forces.at[:, element_indices + 1].add(nodal_force.T)
    torque_material = jnp.einsum("nij,nj->ni", directors, torque_world)
    external_torques = external_torques.at[:, element_indices].add(torque_material.T)
    return external_forces, external_torques


def contact_force(
    distance, normal, relative_velocity, *, contact_stiffness, contact_damping
):
    normal_speed = -jnp.sum(relative_velocity * normal, axis=-1)
    stiffness = 0.5 * contact_stiffness
    damping = 0.5 * contact_damping
    magnitude = jnp.maximum(0.0, stiffness * (-distance) + damping * normal_speed)
    tangent_velocity = -relative_velocity - normal_speed[..., None] * normal
    tangent_speed = jnp.linalg.norm(tangent_velocity, axis=-1)
    tangent_magnitude = jnp.minimum(damping * tangent_speed, 1.0e-10 * magnitude)
    tangent = tangent_velocity / jnp.maximum(
        tangent_speed[..., None], CONTACT_THRESHOLD
    )
    active = distance < -CONTACT_THRESHOLD
    return jnp.where(
        active[..., None],
        magnitude[..., None] * normal + tangent_magnitude[..., None] * tangent,
        0.0,
    )


def apply_rod_contacts(
    *,
    pair_first,
    pair_second,
    centers,
    velocities,
    axes,
    lengths,
    radii,
    omega,
    directors,
    elem,
    external_forces,
    external_torques,
    cached_candidates,
    last_detection_time,
    time,
    contact_stiffness,
    contact_damping,
    steps_between_detection,
    time_step,
):
    first = jnp.asarray(pair_first)
    second = jnp.asarray(pair_second)
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

    cell_size = 2.0 * jnp.max(radii) + jnp.max(lengths)
    cells1 = jnp.floor(c1 / cell_size).astype(jnp.int32)
    cells2 = jnp.floor(c2 / cell_size).astype(jnp.int32)
    grid_neighbors = jnp.all(jnp.abs(cells1 - cells2) <= 1, axis=-1)
    extent1 = half1[:, None] * jnp.abs(a1) + radii[first, None]
    extent2 = half2[:, None] * jnp.abs(a2) + radii[second, None]
    detected_candidates = grid_neighbors & jnp.all(
        jnp.abs(c1 - c2) <= extent1 + extent2, axis=-1
    )
    detection_interval = steps_between_detection * time_step
    detection_due = (detection_interval == 0.0) | (
        time - last_detection_time >= detection_interval
    )
    broad_phase = jnp.where(detection_due, detected_candidates, cached_candidates)
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
    )
    force = jnp.where((axis_distance > CONTACT_THRESHOLD)[:, None], force, 0.0)
    parallel_overlap = (
        jnp.abs(jnp.sum(a1 * a2, axis=-1)) > 1.0 - CONTACT_THRESHOLD
    ) & (jnp.abs(jnp.sum((c2 - c1) * a1, axis=-1)) < lengths[first])
    force *= jnp.where(parallel_overlap, 2.0, 1.0)[:, None]

    f1 = jnp.zeros_like(centers).at[first].add(force)
    f2 = jnp.zeros_like(centers).at[second].add(-force)
    t1 = jnp.zeros_like(centers).at[first].add(jnp.cross(arm1, force))
    t2 = jnp.zeros_like(centers).at[second].add(jnp.cross(arm2, -force))
    total_force, total_torque = f1 + f2, t1 + t2
    return (
        *scatter_element_loads(
            external_forces,
            external_torques,
            elem,
            total_force,
            total_torque,
            directors,
        ),
        broad_phase,
        last_detection_time,
    )


def apply_wall_contacts(
    *,
    wall_origins,
    wall_normals,
    centers,
    velocities,
    axes,
    lengths,
    radii,
    omega,
    directors,
    elem,
    external_forces,
    external_torques,
    contact_stiffness,
    contact_damping,
):
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
        elem,
        total_force,
        total_torque,
        directors,
    )


def spline_actuation_amplitude(n_elements: int) -> np.ndarray:
    spline_x = np.array(
        [0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.53, 0.61, 0.69, 0.77, 0.85, 0.93]
    )
    spline_y = np.array(
        [
            0.003394,
            0.003474,
            0.003604,
            0.003697,
            0.003673,
            0.003525,
            0.003330,
            0.003170,
            0.003126,
            0.003219,
            0.003371,
            0.003489,
        ]
    )
    centers = (np.arange(n_elements) + 0.5) / n_elements
    try:
        from scipy.interpolate import CubicSpline

        return CubicSpline(spline_x, spline_y)(centers)
    except ModuleNotFoundError:  # pragma: no cover - project depends on SciPy
        return np.interp(centers, spline_x, spline_y)
