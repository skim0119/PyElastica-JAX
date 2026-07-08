"""Ground-plane and cylindrical-wall interaction for the nest packing case.

Node-based reproduction of the C++ ``FrictionPlaneInteraction`` boundary
response (excluding the moving top lid, which is handled separately): a
Hertzian ``gamma^1.5`` elastic response with ``gamma^0.5`` normal damping on
both the flat ground and the cylindrical wall, plus isotropic kinetic Coulomb
friction that activates after a set time.
"""

from __future__ import annotations

import jax.numpy as jnp

import elastica_jax as eaj
from elastica_jax.contact import CONTACT_THRESHOLD, slip_ramp


class SubstrateInteractionJax(eaj.NoBlockOpJax):
    """Ground-plane and cylindrical-wall contact on packed rod nodes.

    Parameters
    ----------
    cylinder_radius : float
        Radius of the containing cylinder.
    rod_radius : float
        Rod cross-section radius (contact offset from surfaces).
    stiffness : float
        Elastic coefficient ``k`` in the ``k * gamma^1.5`` normal law.
    damping : float
        Viscous coefficient ``eta`` in the ``eta * gamma^0.5`` normal damping.
    kinetic_friction_coefficient : float
        Isotropic kinetic Coulomb coefficient applied to boundary contacts.
    static_velocity_threshold : float
        Slip speed below which kinetic friction is suppressed.
    friction_start_time : float
        Simulation time after which kinetic friction is active.
    """

    def __init__(
        self,
        *,
        cylinder_radius: float,
        rod_radius: float,
        stiffness: float,
        damping: float,
        kinetic_friction_coefficient: float = 0.0,
        static_velocity_threshold: float = 1.0,
        friction_start_time: float = float("inf"),
        _system=None,
    ) -> None:
        self.cylinder_radius = float(cylinder_radius)
        self.rod_radius = float(rod_radius)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.kinetic_friction_coefficient = float(kinetic_friction_coefficient)
        self.static_velocity_threshold = float(static_velocity_threshold)
        self.friction_start_time = float(friction_start_time)

    def jax_block_operate_synchronize(self, state, time):
        positions = state["position_collection"]
        velocities = state["velocity_collection"]
        friction_gate = jnp.where(time >= self.friction_start_time, 1.0, 0.0)

        ground_normal = jnp.zeros_like(positions)
        ground_normal = ground_normal.at[2].set(1.0)
        ground_penetration = self.rod_radius - positions[2]
        ground_force = self._half_space_force(
            ground_normal, ground_penetration, velocities, friction_gate
        )

        radial = positions.at[2].set(0.0)
        distance = jnp.sqrt(jnp.sum(radial * radial, axis=0) + 1.0e-30)
        wall_normal = -radial / distance[None, :]
        wall_penetration = distance - (self.cylinder_radius - self.rod_radius)
        wall_force = self._half_space_force(
            wall_normal, wall_penetration, velocities, friction_gate
        )

        external_forces = state["external_forces"] + ground_force + wall_force
        return {**state, "external_forces": external_forces}

    def _half_space_force(self, normal, penetration, velocities, friction_gate):
        """Return the ``(3, n_nodes)`` contact force for one half-space.

        Parameters
        ----------
        normal : jax.Array
            Unit restoring normals, shape ``(3, n_nodes)``.
        penetration : jax.Array
            Signed overlap per node (positive under contact), shape ``(n_nodes,)``.
        velocities : jax.Array
            Nodal velocities, shape ``(3, n_nodes)``.
        friction_gate : jax.Array
            Scalar ``0`` or ``1`` gate that activates kinetic friction.

        Returns
        -------
        jax.Array
            Nodal contact force, shape ``(3, n_nodes)``.
        """
        active = penetration > 0.0
        gamma_sqrt = jnp.sqrt(jnp.maximum(penetration, 0.0))
        normal_velocity = jnp.sum(velocities * normal, axis=0)
        normal_magnitude = (
            self.stiffness * penetration - self.damping * normal_velocity
        ) * gamma_sqrt
        normal_force = normal_magnitude[None, :] * normal

        slip = velocities - normal_velocity[None, :] * normal
        slip_speed = jnp.linalg.norm(slip, axis=0)
        slip_direction = slip / jnp.maximum(slip_speed[None, :], CONTACT_THRESHOLD)
        blend = slip_ramp(slip_speed, self.static_velocity_threshold)
        friction_magnitude = (
            self.kinetic_friction_coefficient
            * jnp.abs(normal_magnitude)
            * (1.0 - blend)
            * friction_gate
        )
        friction_force = -friction_magnitude[None, :] * slip_direction

        return jnp.where(active[None, :], normal_force + friction_force, 0.0)
