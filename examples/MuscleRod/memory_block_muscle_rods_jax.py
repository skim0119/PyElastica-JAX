"""JAX memory block for :class:`MuscleArm`."""

from __future__ import annotations

from typing import Any

import numpy as np

from elastica_jax.memory_block.memory_block_rod_jax import (
    _CosseratRodMemoryBlock,
    _jax_trapezoidal_for_block_structure,
    _jax_two_point_difference_for_block_structure,
)
from elastica.typing import SystemIdxType

from .muscle_rod import MuscleArm

import jax
import jax.numpy as jnp


def _reset_vector_ghosts(array: jax.Array, ghost_idx: jax.Array) -> jax.Array:
    if ghost_idx.size == 0:
        return array
    return array.at[..., ghost_idx].set(0.0)


@jax.jit
def _jax_compute_muscle_loads(
    time: jax.Array,
    radius: jax.Array,
    sigma: jax.Array,
    kappa: jax.Array,
    rest_lengths: jax.Array,
    rest_voronoi_lengths: jax.Array,
    dilatation: jax.Array,
    voronoi_dilatation: jax.Array,
    tangents: jax.Array,
    director_collection: jax.Array,
    ratio_muscle_position: jax.Array,
    rest_muscle_area: jax.Array,
    max_muscle_stress: jax.Array,
    transverse_muscle: jax.Array,
    muscle_rest_length: jax.Array,
    activation_offset: jax.Array,
    activation_amplitude: jax.Array,
    activation_frequency: jax.Array,
    activation_phase: jax.Array,
    force_length_coefficient: jax.Array,
    ghost_elems_idx: jax.Array,
    ghost_voronoi_idx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    muscle_position = ratio_muscle_position * radius[jnp.newaxis, jnp.newaxis, :]
    average_position = 0.5 * (muscle_position[..., 1:] + muscle_position[..., :-1])
    position_derivative = (muscle_position[..., 1:] - muscle_position[..., :-1]) / (
        rest_voronoi_lengths * voronoi_dilatation
    )[jnp.newaxis, jnp.newaxis, :]
    voronoi_strain = (
        jnp.cross(
            jnp.transpose(kappa)[jnp.newaxis, ...],
            jnp.transpose(average_position, (0, 2, 1)),
        ).transpose(0, 2, 1)
        + position_derivative
    )
    voronoi_strain = _reset_vector_ghosts(voronoi_strain, ghost_voronoi_idx)

    n_elems = sigma.shape[-1]
    muscle_strain = jnp.zeros(
        (ratio_muscle_position.shape[0], 3, n_elems), dtype=sigma.dtype
    )
    muscle_strain = muscle_strain.at[..., 0].set(0.5 * voronoi_strain[..., 0])
    muscle_strain = muscle_strain.at[..., -1].set(0.5 * voronoi_strain[..., -1])
    muscle_strain = muscle_strain.at[..., 1:-1].set(
        0.5 * (voronoi_strain[..., 1:] + voronoi_strain[..., :-1])
    )
    muscle_strain = muscle_strain + sigma[jnp.newaxis, ...]
    muscle_strain = muscle_strain.at[:, 2, :].add(1.0)

    strain_norm = jnp.linalg.norm(muscle_strain, axis=1)
    muscle_length = jnp.where(
        transverse_muscle, 1.0 / jnp.sqrt(strain_norm), strain_norm
    )
    muscle_tangent = muscle_strain / muscle_length[:, jnp.newaxis, :]
    normalized_length = muscle_length / muscle_rest_length
    force_length_weight = jnp.maximum(
        0.0,
        1.0 - force_length_coefficient * (normalized_length - 1.0) ** 2,
    )
    activation = jnp.clip(
        activation_offset
        + activation_amplitude
        * jnp.sin(2.0 * jnp.pi * activation_frequency * time + activation_phase),
        0.0,
        1.0,
    )
    force_magnitude = (
        activation
        * max_muscle_stress
        * force_length_weight
        * rest_muscle_area
        / dilatation[jnp.newaxis, :]
    )
    muscle_force = force_magnitude[:, jnp.newaxis, :] * muscle_tangent
    muscle_force = _reset_vector_ghosts(muscle_force, ghost_elems_idx)

    muscle_couple = 0.5 * (
        jnp.cross(
            jnp.transpose(muscle_position[..., :-1], (0, 2, 1)),
            jnp.transpose(muscle_force[..., :-1], (0, 2, 1)),
        )
        + jnp.cross(
            jnp.transpose(muscle_position[..., 1:], (0, 2, 1)),
            jnp.transpose(muscle_force[..., 1:], (0, 2, 1)),
        )
    ).transpose(0, 2, 1)
    muscle_couple = _reset_vector_ghosts(muscle_couple, ghost_voronoi_idx)

    total_force = jnp.sum(muscle_force, axis=0)
    force_lab = jnp.einsum("jik,jk->ik", director_collection, total_force)
    nodal_load = _jax_two_point_difference_for_block_structure(
        force_lab, ghost_elems_idx
    )

    total_couple = jnp.sum(muscle_couple, axis=0)
    torque_load = _jax_two_point_difference_for_block_structure(
        total_couple, ghost_voronoi_idx
    )
    torque_load += _jax_trapezoidal_for_block_structure(
        jnp.cross(kappa.T, total_couple.T).T * rest_voronoi_lengths[jnp.newaxis, :],
        ghost_voronoi_idx,
    )
    material_tangent = jnp.einsum(
        "ijk,jk->ik", director_collection, tangents * dilatation
    )
    torque_load += (
        jnp.cross(material_tangent.T, total_force.T).T * rest_lengths[jnp.newaxis, :]
    )
    return nodal_load, torque_load


class MemoryBlockMuscleArmJax(_CosseratRodMemoryBlock):
    """Packed JAX block for one or more open ``MuscleArm`` rods."""

    def __init__(
        self,
        systems: list[MuscleArm],
        system_idx_list: list[SystemIdxType],
        **kwargs: Any,
    ) -> None:
        assert systems, "MemoryBlockMuscleArmJax requires at least one MuscleArm."
        assert all(not rod.ring_rod_flag for rod in systems), (
            "MemoryBlockMuscleArmJax currently supports only open MuscleArm rods."
        )
        assert all(rod.muscle_config is not None for rod in systems), (
            "Every MuscleArm must be configured before block construction."
        )
        super().__init__(**kwargs)
        self(systems, system_idx_list)
        self._pack_muscle_configuration(systems)

    def _pack_muscle_configuration(self, systems: list[MuscleArm]) -> None:
        configs = [rod.muscle_config for rod in systems]
        assert all(config is not None for config in configs), (
            "Every MuscleArm must have a MuscleConfig."
        )
        max_muscles = max(config.ratio_muscle_position.shape[0] for config in configs)  # type: ignore[union-attr]
        n_elems = self.n_elems
        ratio = np.zeros((max_muscles, 3, n_elems))
        arrays = {
            name: np.zeros((max_muscles, n_elems))
            for name in (
                "rest_muscle_area",
                "max_muscle_stress",
                "transverse_muscle",
                "muscle_rest_length",
                "activation_offset",
                "activation_amplitude",
                "activation_frequency",
                "activation_phase",
                "force_length_coefficient",
            )
        }
        arrays["muscle_rest_length"].fill(1.0)
        for rod_idx, config in enumerate(configs):
            assert config is not None
            start = int(self.start_idx_in_rod_elems[rod_idx])
            end = int(self.end_idx_in_rod_elems[rod_idx])
            n_muscles = config.ratio_muscle_position.shape[0]
            ratio[:n_muscles, :, start:end] = config.ratio_muscle_position
            arrays["rest_muscle_area"][:n_muscles, start:end] = config.rest_muscle_area
            arrays["max_muscle_stress"][:n_muscles, start:end] = (
                config.max_muscle_stress
            )
            arrays["transverse_muscle"][:n_muscles, start:end] = (
                config.transverse_muscle[:, None]
            )
            assert config.muscle_rest_length is not None
            arrays["muscle_rest_length"][:n_muscles, start:end] = (
                config.muscle_rest_length
            )
            shape = (n_muscles, end - start)
            for name in (
                "activation_offset",
                "activation_amplitude",
                "activation_frequency",
                "activation_phase",
            ):
                arrays[name][:n_muscles, start:end] = np.broadcast_to(
                    np.asarray(getattr(config, name)), shape
                )
            arrays["force_length_coefficient"][:n_muscles, start:end] = (
                config.force_length_coefficient
            )

        device = self.device
        dtype = self._device_dtype
        self._muscle_metadata = {
            "ratio_muscle_position": jax.device_put(
                np.asarray(ratio, dtype=dtype), device=device
            )
        }
        for name, value in arrays.items():
            self._muscle_metadata[name] = jax.device_put(
                np.asarray(value, dtype=dtype), device=device
            )

    def jax_compute_internal_forces_and_torques(
        self,
        state: dict[str, jax.Array],
        time: np.float64,
    ) -> dict[str, jax.Array]:
        updated = super().jax_compute_internal_forces_and_torques(state, time)
        muscle_forces, muscle_torques = _jax_compute_muscle_loads(
            self._device_dtype.type(np.asarray(time)),
            updated["radius"],
            updated["sigma"],
            updated["kappa"],
            updated["rest_lengths"],
            updated["rest_voronoi_lengths"],
            updated["dilatation"],
            updated["voronoi_dilatation"],
            updated["tangents"],
            updated["director_collection"],
            self._muscle_metadata["ratio_muscle_position"],
            self._muscle_metadata["rest_muscle_area"],
            self._muscle_metadata["max_muscle_stress"],
            self._muscle_metadata["transverse_muscle"],
            self._muscle_metadata["muscle_rest_length"],
            self._muscle_metadata["activation_offset"],
            self._muscle_metadata["activation_amplitude"],
            self._muscle_metadata["activation_frequency"],
            self._muscle_metadata["activation_phase"],
            self._muscle_metadata["force_length_coefficient"],
            self._device_metadata["ghost_elems_idx"],
            self._device_metadata["ghost_voronoi_idx"],
        )
        updated["internal_forces"] = updated["internal_forces"] + muscle_forces
        updated["internal_torques"] = updated["internal_torques"] + muscle_torques
        return updated


# Factory using closure
def muscle_block_with(device, device_dtype) -> type[MemoryBlockMuscleArmJax]:
    class ConfiguredBlock(MemoryBlockMuscleArmJax):
        device = None

        def __init__(self, systems, system_idx_list):
            super().__init__(
                systems,
                system_idx_list,
                device=self.device,
                device_dtype=np.float64,
            )

    return ConfiguredBlock
