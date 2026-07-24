"""Collection names synced between host rods and Block device state."""

from __future__ import annotations

_NODE_ATTRS: tuple[str, ...] = (
    "mass",
    "position_collection",
    "internal_forces",
    "external_forces",
    "velocity_collection",
    "acceleration_collection",
)
_ELEMENT_ATTRS: tuple[str, ...] = (
    "radius",
    "volume",
    "density",
    "lengths",
    "rest_lengths",
    "dilatation",
    "dilatation_rate",
    "tangents",
    "sigma",
    "rest_sigma",
    "internal_torques",
    "external_torques",
    "internal_stress",
    "director_collection",
    "mass_second_moment_of_inertia",
    "inv_mass_second_moment_of_inertia",
    "shear_matrix",
    "omega_collection",
    "alpha_collection",
)
_VORONOI_ATTRS: tuple[str, ...] = (
    "voronoi_dilatation",
    "rest_voronoi_lengths",
    "kappa",
    "rest_kappa",
    "internal_couple",
    "bend_matrix",
)
_SYNCABLE_ATTRS: tuple[str, ...] = _NODE_ATTRS + _ELEMENT_ATTRS + _VORONOI_ATTRS
