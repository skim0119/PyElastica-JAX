"""Cosserat rod with muscle forces embedded in its constitutive update."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from elastica._calculus import difference_kernel, quadrature_kernel
from elastica._linalg import _batch_cross, _batch_matvec
from elastica.rod.cosserat_rod import CosseratRod

if TYPE_CHECKING:
    from .batch_muscle import BatchMuscle


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MuscleConfig:
    """Element-wise muscle parameters and a sinusoidal activation program.

    The first axis indexes muscles and the last axis indexes rod elements.
    ``max_muscle_stress`` is signed: transverse muscles use negative stress,
    matching :class:`BatchMuscle`'s volume-preserving radial contraction.
    """

    ratio_muscle_position: FloatArray
    rest_muscle_area: FloatArray
    max_muscle_stress: FloatArray
    transverse_muscle: NDArray[np.bool_]
    activation_offset: float | FloatArray = 0.5
    activation_amplitude: float | FloatArray = 0.5
    activation_frequency: float | FloatArray = 1.0
    activation_phase: float | FloatArray = 0.0
    force_length_coefficient: float = 5.0
    muscle_rest_length: FloatArray | None = None

    def validated(self, n_elements: int) -> Self:
        ratio = np.asarray(self.ratio_muscle_position, dtype=np.float64)
        area = np.asarray(self.rest_muscle_area, dtype=np.float64)
        stress = np.asarray(self.max_muscle_stress, dtype=np.float64)
        transverse = np.asarray(self.transverse_muscle, dtype=np.bool_)
        assert ratio.ndim == 3 and ratio.shape[1:] == (
            3,
            n_elements,
        ), "ratio_muscle_position must have shape (n_muscles, 3, n_elements)."
        n_muscles = ratio.shape[0]
        assert n_muscles > 0, "MuscleConfig must contain at least one muscle."
        assert area.shape == (
            n_muscles,
            n_elements,
        ), "rest_muscle_area must have shape (n_muscles, n_elements)."
        assert stress.shape == (
            n_muscles,
            n_elements,
        ), "max_muscle_stress must have shape (n_muscles, n_elements)."
        assert transverse.shape == (
            n_muscles,
        ), "transverse_muscle must have shape (n_muscles,)."
        assert np.all(area >= 0.0), "rest_muscle_area must be nonnegative."
        assert (
            self.force_length_coefficient >= 0.0
        ), "force_length_coefficient must be nonnegative."
        rest_length = self.muscle_rest_length
        if rest_length is not None:
            rest_length = np.asarray(rest_length, dtype=np.float64)
            assert rest_length.shape == (
                n_muscles,
                n_elements,
            ), "muscle_rest_length must have shape (n_muscles, n_elements)."
            assert np.all(
                rest_length > 0.0
            ), "muscle_rest_length must be strictly positive."
        activation_parameters = {}
        for name in (
            "activation_offset",
            "activation_amplitude",
            "activation_frequency",
            "activation_phase",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape == (n_muscles,):
                value = value[:, None]
            assert np.broadcast_shapes(value.shape, (n_muscles, n_elements)) == (
                n_muscles,
                n_elements,
            ), f"{name} must broadcast to (n_muscles, n_elements)."
            activation_parameters[name] = np.broadcast_to(
                value, (n_muscles, n_elements)
            ).copy()
        return replace(
            self,
            ratio_muscle_position=ratio,
            rest_muscle_area=area,
            max_muscle_stress=stress,
            transverse_muscle=transverse,
            muscle_rest_length=rest_length,
            **activation_parameters,
        )

    @classmethod
    def from_batch_muscle(
        cls,
        batch_muscle: BatchMuscle,
        **activation_kwargs: Any,
    ) -> Self:
        """Create a configuration from an already blocked legacy actuator."""
        assert (
            batch_muscle._blocked
        ), "BatchMuscle must be blocked against a rod before conversion."
        stress = np.asarray(batch_muscle.max_muscle_stress, dtype=np.float64)
        return cls(
            ratio_muscle_position=np.asarray(
                batch_muscle.ratio_muscle_position, dtype=np.float64
            ).copy(),
            rest_muscle_area=np.asarray(
                batch_muscle.rest_muscle_area, dtype=np.float64
            ).copy(),
            max_muscle_stress=stress.copy(),
            transverse_muscle=np.all(stress < 0.0, axis=1),
            muscle_rest_length=np.asarray(
                batch_muscle.muscle_rest_length, dtype=np.float64
            ).copy(),
            **activation_kwargs,
        ).validated(batch_muscle.num_elements)


def _broadcast_activation_parameter(
    value: float | FloatArray, shape: tuple[int, int]
) -> FloatArray:
    return np.broadcast_to(np.asarray(value, dtype=np.float64), shape)


def _muscle_activation(config: MuscleConfig, time: float) -> FloatArray:
    shape = config.rest_muscle_area.shape
    offset = _broadcast_activation_parameter(config.activation_offset, shape)
    amplitude = _broadcast_activation_parameter(config.activation_amplitude, shape)
    frequency = _broadcast_activation_parameter(config.activation_frequency, shape)
    phase = _broadcast_activation_parameter(config.activation_phase, shape)
    activation = offset + amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
    return np.clip(activation, 0.0, 1.0)


def _muscle_kinematics(
    radius: FloatArray,
    sigma: FloatArray,
    kappa: FloatArray,
    rest_voronoi_lengths: FloatArray,
    voronoi_dilatation: FloatArray,
    config: MuscleConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    position = config.ratio_muscle_position * radius[None, None, :]
    position_derivative = (
        np.diff(position, axis=-1)
        / (rest_voronoi_lengths * voronoi_dilatation)[None, None, :]
    )
    average_position = 0.5 * (position[..., 1:] + position[..., :-1])
    kappa_cross_position = np.cross(
        kappa.T[None, :, :], average_position.transpose(0, 2, 1)
    ).transpose(0, 2, 1)
    voronoi_strain = kappa_cross_position + position_derivative
    strain = np.empty_like(position)
    strain[..., 0] = 0.5 * voronoi_strain[..., 0]
    strain[..., -1] = 0.5 * voronoi_strain[..., -1]
    strain[..., 1:-1] = 0.5 * (voronoi_strain[..., 1:] + voronoi_strain[..., :-1])
    strain += sigma[None, :, :]
    strain[:, 2, :] += 1.0
    norm = np.linalg.norm(strain, axis=1)
    length = np.where(config.transverse_muscle[:, None], 1.0 / np.sqrt(norm), norm)
    tangent = strain / length[:, None, :]
    return position, length, tangent


def _compute_muscle_loads(
    rod: CosseratRod,
    config: MuscleConfig,
    time: float,
) -> tuple[FloatArray, FloatArray]:
    position, length, tangent = _muscle_kinematics(
        rod.radius,
        rod.sigma,
        rod.kappa,
        rod.rest_voronoi_lengths,
        rod.voronoi_dilatation,
        config,
    )
    assert (
        config.muscle_rest_length is not None
    ), "muscle_rest_length must be initialized before computing muscle loads."
    normalized_length = length / config.muscle_rest_length
    weight = np.maximum(
        0.0,
        1.0 - config.force_length_coefficient * (normalized_length - 1.0) ** 2,
    )
    force_magnitude = (
        _muscle_activation(config, time)
        * config.max_muscle_stress
        * weight
        * config.rest_muscle_area
        / rod.dilatation[None, :]
    )
    muscle_force = force_magnitude[:, None, :] * tangent
    muscle_couple = 0.5 * (
        np.cross(
            position[..., :-1].transpose(0, 2, 1),
            muscle_force[..., :-1].transpose(0, 2, 1),
        )
        + np.cross(
            position[..., 1:].transpose(0, 2, 1),
            muscle_force[..., 1:].transpose(0, 2, 1),
        )
    ).transpose(0, 2, 1)

    total_force = np.sum(muscle_force, axis=0)
    force_lab = _batch_matvec(
        np.transpose(rod.director_collection, (1, 0, 2)), total_force
    )
    nodal_load = difference_kernel(force_lab)

    total_couple = np.sum(muscle_couple, axis=0)
    torque_load = difference_kernel(total_couple)
    torque_load += quadrature_kernel(
        _batch_cross(rod.kappa, total_couple) * rod.rest_voronoi_lengths
    )
    material_tangent = _batch_matvec(
        rod.director_collection, rod.tangents * rod.dilatation
    )
    torque_load += _batch_cross(material_tangent, total_force) * rod.rest_lengths
    return nodal_load, torque_load


class MuscleArm(CosseratRod):
    """Cosserat rod with time-programmed muscle fibers."""

    muscle_config: MuscleConfig | None

    def __init__(self, *args: Any, muscle_config: MuscleConfig | None = None) -> None:
        super().__init__(*args)
        self.muscle_config = None
        if muscle_config is not None:
            self.configure_muscles(muscle_config)

    @classmethod
    def straight_rod(
        cls, *args: Any, muscle_config: MuscleConfig | None = None, **kwargs: Any
    ) -> Self:
        rod = super().straight_rod(*args, **kwargs)
        if muscle_config is not None:
            rod.configure_muscles(muscle_config)
        return rod

    @classmethod
    def ring_rod(cls, *args: Any, **kwargs: Any) -> Self:
        raise NotImplementedError("MuscleArm currently supports only open rods.")

    def configure_muscles(self, muscle_config: MuscleConfig) -> None:
        config = muscle_config.validated(self.n_elems)
        if config.muscle_rest_length is None:
            _, rest_length, _ = _muscle_kinematics(
                self.radius,
                self.sigma,
                self.kappa,
                self.rest_voronoi_lengths,
                self.voronoi_dilatation,
                config,
            )
            config = replace(config, muscle_rest_length=rest_length)
        self.muscle_config = config

    def compute_internal_forces_and_torques(self, time: np.float64) -> None:
        super().compute_internal_forces_and_torques(time)
        if self.muscle_config is None:
            return
        muscle_forces, muscle_torques = _compute_muscle_loads(
            self, self.muscle_config, float(time)
        )
        self.internal_forces += muscle_forces
        self.internal_torques += muscle_torques
