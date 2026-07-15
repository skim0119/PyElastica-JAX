"""JAX kernels mirroring PyElastica's ``RodRodContact`` element loops."""

from __future__ import annotations

import jax
import jax.numpy as jnp

_GAMMA_EPS = 1.0e-5
_PARALLEL_EPS = 1.0e-6
_NORM_EPS = 1.0e-14


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.dot(a, b)


def _norm(vec: jax.Array) -> jax.Array:
    return jnp.linalg.norm(vec)


def find_min_dist_jax(
    x1: jax.Array,
    e1: jax.Array,
    x2: jax.Array,
    e2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Closest points between two centerline segments (PyElastica ``_find_min_dist``)."""
    e1e1 = _dot(e1, e1)
    e1e2 = _dot(e1, e2)
    e2e2 = _dot(e2, e2)
    x1e1 = _dot(x1, e1)
    x1e2 = _dot(x1, e2)
    x2e1 = _dot(e1, x2)
    x2e2 = _dot(x2, e2)

    parallel = (
        jnp.abs(1.0 - e1e2**2 / jnp.maximum(e1e1 * e2e2, _NORM_EPS)) < _PARALLEL_EPS
    )

    def parallel_branch(_: None) -> tuple[jax.Array, jax.Array]:
        t_val = (x2e1 - x1e1) / jnp.maximum(e1e1, _NORM_EPS)
        t_val = jnp.clip(t_val, 0.0, 1.0)
        s_val = (x1e2 + t_val * e1e2 - x2e2) / jnp.maximum(e2e2, _NORM_EPS)
        s_val = jnp.clip(s_val, 0.0, 1.0)
        return s_val, t_val

    def general_branch(_: None) -> tuple[jax.Array, jax.Array]:
        denom = jnp.maximum(e1e1 * e2e2 - e1e2**2, _NORM_EPS)
        s_val = (e1e1 * (x1e2 - x2e2) + e1e2 * (x2e1 - x1e1)) / denom
        t_val = (e1e2 * s_val + x2e1 - x1e1) / jnp.maximum(e1e1, _NORM_EPS)

        def endpoint_search(_: None) -> tuple[jax.Array, jax.Array]:
            potential_t = (x2e1 - x1e1) / jnp.maximum(e1e1, _NORM_EPS)
            t_best = jnp.clip(potential_t, 0.0, 1.0)
            s_best = jnp.zeros((), dtype=x1.dtype)
            dist_best = _norm(x1 + e1 * t_best - x2)

            potential_t = (x2e1 + e1e2 - x1e1) / jnp.maximum(e1e1, _NORM_EPS)
            potential_t = jnp.clip(potential_t, 0.0, 1.0)
            dist = _norm(x1 + e1 * potential_t - x2 - e2)
            s_best, t_best, dist_best = jax.lax.cond(
                dist < dist_best,
                lambda _: (jnp.ones((), dtype=x1.dtype), potential_t, dist),
                lambda args: args,
                (s_best, t_best, dist_best),
            )

            potential_s = (x1e2 - x2e2) / jnp.maximum(e2e2, _NORM_EPS)
            potential_s = jnp.clip(potential_s, 0.0, 1.0)
            dist = _norm(x2 + potential_s * e2 - x1)
            s_best, t_best, dist_best = jax.lax.cond(
                dist < dist_best,
                lambda _: (potential_s, jnp.zeros((), dtype=x1.dtype), dist),
                lambda args: args,
                (s_best, t_best, dist_best),
            )

            potential_s = (x1e2 + e1e2 - x2e2) / jnp.maximum(e2e2, _NORM_EPS)
            potential_s = jnp.clip(potential_s, 0.0, 1.0)
            dist = _norm(x2 + potential_s * e2 - x1 - e1)
            s_best, t_best, _ = jax.lax.cond(
                dist < dist_best,
                lambda _: (potential_s, jnp.ones((), dtype=x1.dtype), dist),
                lambda args: args,
                (s_best, t_best, dist_best),
            )
            return s_best, t_best

        def keep_general(_: None) -> tuple[jax.Array, jax.Array]:
            return s_val, t_val

        out_of_bounds = (s_val < 0.0) | (s_val > 1.0) | (t_val < 0.0) | (t_val > 1.0)
        return jax.lax.cond(out_of_bounds, endpoint_search, keep_general, operand=None)

    s, t = jax.lax.cond(parallel, parallel_branch, general_branch, operand=None)
    return x2 + s * e2 - x1 - t * e1, x2 + s * e2, x1 - t * e1


def rods_aabb_disjoint(
    position_one: jax.Array,
    radius_one: jax.Array,
    length_one: jax.Array,
    position_two: jax.Array,
    radius_two: jax.Array,
    length_two: jax.Array,
) -> jax.Array:
    """Return whether rod AABBs are disjoint (PyElastica ``_prune_using_aabbs_rod_rod``)."""
    max_dim_one = jnp.max(radius_one) + jnp.max(length_one)
    max_dim_two = jnp.max(radius_two) + jnp.max(length_two)
    aabb_one_lo = jnp.min(position_one, axis=1) - max_dim_one
    aabb_one_hi = jnp.max(position_one, axis=1) + max_dim_one
    aabb_two_lo = jnp.min(position_two, axis=1) - max_dim_two
    aabb_two_hi = jnp.max(position_two, axis=1) + max_dim_two
    separated = (
        (aabb_one_hi[0] < aabb_two_lo[0])
        | (aabb_one_lo[0] > aabb_two_hi[0])
        | (aabb_one_hi[1] < aabb_two_lo[1])
        | (aabb_one_lo[1] > aabb_two_hi[1])
        | (aabb_one_hi[2] < aabb_two_lo[2])
        | (aabb_one_lo[2] > aabb_two_hi[2])
    )
    return separated


def _scatter_pair_force(
    external_forces: jax.Array,
    node_index: int,
    n_elements: int,
    net_contact_force: jax.Array,
    *,
    is_first_rod: bool,
) -> jax.Array:
    sign = 1.0 if is_first_rod else -1.0
    force = sign * net_contact_force
    at_start = node_index == 0
    at_end = node_index == n_elements - 1

    def start_nodes(forces: jax.Array) -> jax.Array:
        forces = forces.at[:, node_index].add(force * (2.0 / 3.0))
        return forces.at[:, node_index + 1].add(force * (4.0 / 3.0))

    def end_nodes(forces: jax.Array) -> jax.Array:
        forces = forces.at[:, node_index].add(force * (4.0 / 3.0))
        return forces.at[:, node_index + 1].add(force * (2.0 / 3.0))

    def interior_nodes(forces: jax.Array) -> jax.Array:
        forces = forces.at[:, node_index].add(force)
        return forces.at[:, node_index + 1].add(force)

    return jax.lax.cond(
        at_start,
        start_nodes,
        lambda forces: jax.lax.cond(at_end, end_nodes, interior_nodes, forces),
        external_forces,
    )


def apply_rod_rod_contact_forces(
    *,
    x_one: jax.Array,
    radius_one: jax.Array,
    length_one: jax.Array,
    tangent_one: jax.Array,
    velocity_one: jax.Array,
    internal_forces_one: jax.Array,
    external_forces_one: jax.Array,
    x_two: jax.Array,
    radius_two: jax.Array,
    length_two: jax.Array,
    tangent_two: jax.Array,
    velocity_two: jax.Array,
    internal_forces_two: jax.Array,
    external_forces_two: jax.Array,
    contact_k: jax.Array,
    contact_nu: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply PyElastica-style rod-rod contact for one registered rod pair."""
    disjoint = rods_aabb_disjoint(
        x_one,
        radius_one,
        length_one,
        x_two,
        radius_two,
        length_two,
    )

    def skip_pair(
        forces: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        return forces

    def run_pair(
        forces: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        return _apply_rod_rod_contact_forces_loop(
            x_one=x_one,
            radius_one=radius_one,
            length_one=length_one,
            tangent_one=tangent_one,
            velocity_one=velocity_one,
            internal_forces_one=internal_forces_one,
            external_forces_one=forces[0],
            x_two=x_two,
            radius_two=radius_two,
            length_two=length_two,
            tangent_two=tangent_two,
            velocity_two=velocity_two,
            internal_forces_two=internal_forces_two,
            external_forces_two=forces[1],
            contact_k=contact_k,
            contact_nu=contact_nu,
        )

    return jax.lax.cond(
        disjoint,
        skip_pair,
        run_pair,
        (external_forces_one, external_forces_two),
    )


def _apply_rod_rod_contact_forces_loop(
    *,
    x_one: jax.Array,
    radius_one: jax.Array,
    length_one: jax.Array,
    tangent_one: jax.Array,
    velocity_one: jax.Array,
    internal_forces_one: jax.Array,
    external_forces_one: jax.Array,
    x_two: jax.Array,
    radius_two: jax.Array,
    length_two: jax.Array,
    tangent_two: jax.Array,
    velocity_two: jax.Array,
    internal_forces_two: jax.Array,
    external_forces_two: jax.Array,
    contact_k: jax.Array,
    contact_nu: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    n_elements_one = x_one.shape[1]
    n_elements_two = x_two.shape[1]
    edge_one = length_one[None, :] * tangent_one
    edge_two = length_two[None, :] * tangent_two

    def outer_step(
        i: int, carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        ext_one, ext_two = carry

        def inner_step(
            j: int, inner_carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            inner_ext_one, inner_ext_two = inner_carry
            radii_sum = radius_one[i] + radius_two[j]
            length_sum = length_one[i] + length_two[j]
            x_sel_one = x_one[:, i]
            x_sel_two = x_two[:, j]
            del_x = x_sel_one - x_sel_two
            norm_del_x = _norm(del_x)
            skip = (norm_del_x >= (radii_sum + length_sum)) | (
                radii_sum + length_sum <= 0.0
            )

            def apply_contact(
                forces: tuple[jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array]:
                f_one, f_two = forces
                distance_vector, _, _ = find_min_dist_jax(
                    x_sel_one,
                    edge_one[:, i],
                    x_sel_two,
                    edge_two[:, j],
                )
                distance_length = jnp.maximum(_norm(distance_vector), _NORM_EPS)
                unit_distance = distance_vector / distance_length
                gamma = radii_sum - distance_length
                skip_contact = gamma < -_GAMMA_EPS

                def add_forces(
                    state: tuple[jax.Array, jax.Array],
                ) -> tuple[jax.Array, jax.Array]:
                    cur_one, cur_two = state
                    rod_one_elemental_forces = 0.5 * (
                        cur_one[:, i]
                        + cur_one[:, i + 1]
                        + internal_forces_one[:, i]
                        + internal_forces_one[:, i + 1]
                    )
                    rod_two_elemental_forces = 0.5 * (
                        cur_two[:, j]
                        + cur_two[:, j + 1]
                        + internal_forces_two[:, j]
                        + internal_forces_two[:, j + 1]
                    )
                    equilibrium_forces = (
                        -rod_one_elemental_forces + rod_two_elemental_forces
                    )
                    normal_force = _dot(equilibrium_forces, unit_distance)
                    normal_force = jnp.abs(jnp.minimum(normal_force, 0.0))
                    mask = jnp.where(gamma > 0.0, 1.0, 0.0)
                    contact_force = contact_k * gamma
                    interpenetration_velocity = 0.5 * (
                        (velocity_one[:, i] + velocity_one[:, i + 1])
                        - (velocity_two[:, j] + velocity_two[:, j + 1])
                    )
                    contact_damping_force = contact_nu * _dot(
                        interpenetration_velocity, unit_distance
                    )
                    net_contact_force = (
                        normal_force
                        + 0.5 * mask * (contact_damping_force + contact_force)
                    ) * unit_distance
                    updated_one = _scatter_pair_force(
                        cur_one,
                        i,
                        n_elements_one,
                        net_contact_force,
                        is_first_rod=True,
                    )
                    updated_two = _scatter_pair_force(
                        cur_two,
                        j,
                        n_elements_two,
                        net_contact_force,
                        is_first_rod=False,
                    )
                    return updated_one, updated_two

                return jax.lax.cond(
                    skip_contact,
                    lambda state: state,
                    add_forces,
                    forces,
                )

            return jax.lax.cond(
                skip,
                lambda forces: forces,
                apply_contact,
                (inner_ext_one, inner_ext_two),
            )

        return jax.lax.fori_loop(0, n_elements_two, inner_step, (ext_one, ext_two))

    return jax.lax.fori_loop(
        0,
        n_elements_one,
        outer_step,
        (external_forces_one, external_forces_two),
    )
