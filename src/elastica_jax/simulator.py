"""Package Simulator: full JAX registration surface and Save/Load."""

from __future__ import annotations

import elastica as ea

from elastica_jax.io.hdf5_io_mixin import Hdf5IO
from elastica_jax.modules.jax_ops import JAXOps
from elastica_jax.modules.jax_ops_block import JAXOpsBlock
from elastica_jax.modules.jax_ops_rod_rod_block import JAXInteraction
from elastica_jax.modules.jax_ops_rod_rigid_body import JAXOpsRodRigidBody


class Simulator(
    ea.BaseSystemCollection,
    JAXOps,
    JAXOpsBlock,
    JAXInteraction,
    JAXOpsRodRigidBody,
    Hdf5IO,
):
    """JAX system collection with operator registration and Save/Load.

    Always exposes ``operate``, ``operate_block``, pairwise registration,
    rod–rigid-body registration, and HDF5 Save/Load. Unused registration
    paths simply leave stage lists empty after ``finalize``.

    Classic PyElastica modules (``Forcing``, ``Constraints``, …) are not
    included; use a separate CPU collection when needed.
    """

    pass
