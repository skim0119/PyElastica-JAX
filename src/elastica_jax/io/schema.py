"""Schema levels and HDF5 layout constants for Save/Load."""

from __future__ import annotations

from elastica_jax.memory_block.memory_block_rod_jax import _SYNCABLE_ATTRS

IO_VERSION = 1

SCHEMA_LEVEL_0: tuple[str, ...] = (
    "position_collection",
    "director_collection",
    "radius",
)
SCHEMA_LEVEL_1: tuple[str, ...] = SCHEMA_LEVEL_0 + (
    "sigma",
    "kappa",
    "velocity_collection",
    "omega_collection",
)

ROD_GROUP = "rod"
BLOCKS_GROUP = "blocks"
TARGET_ROD = "rod"
TARGET_BLOCK = "block"
TARGET_SIMULATOR = "simulator"


def fields_for_schema_level(schema_level: int) -> tuple[str, ...]:
    """Return collection names included at the given schema level.

    Parameters
    ----------
    schema_level :
        One of ``0``, ``1``, or ``10``.

    Returns
    -------
    tuple[str, ...]
        Collection names written or expected for that level.
    """
    assert schema_level in {0, 1, 10}, (
        f"Unsupported schema level {schema_level}; expected 0, 1, or 10."
    )
    if schema_level == 0:
        return SCHEMA_LEVEL_0
    if schema_level == 1:
        return SCHEMA_LEVEL_1
    return _SYNCABLE_ATTRS
