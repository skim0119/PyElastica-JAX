"""Capsule contact operators for JAX rollouts."""

from elastica_jax.contact.capsule_contact import (
    BlockCapsuleMetadata,
    CapsuleContactOp,
    WallContactOp,
    build_block_capsule_metadata,
    capsule_kinematics_from_block_state,
    initialize_capsule_contact_state,
    install_capsule_contact_state,
)
from elastica_jax.contact.kernels import CONTACT_THRESHOLD
from elastica_jax.contact.spatial_hash import (
    SpatialHashPairBuffer,
    default_cell_size,
    estimate_max_pairs,
    rebuild_spatial_hash_pairs,
)

__all__ = [
    "BlockCapsuleMetadata",
    "CapsuleContactOp",
    "CONTACT_THRESHOLD",
    "SpatialHashPairBuffer",
    "WallContactOp",
    "build_block_capsule_metadata",
    "capsule_kinematics_from_block_state",
    "default_cell_size",
    "estimate_max_pairs",
    "initialize_capsule_contact_state",
    "install_capsule_contact_state",
    "rebuild_spatial_hash_pairs",
]
