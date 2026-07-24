from __future__ import annotations

from enum import Enum
from typing import ClassVar

import numpy as np

from elastica_jax.memory_block.protocol import JAXBlockState, RodView

type JAXTime = np.float64


class CommunicationScope(Enum):
    """How far a block operator must reach during synchronize."""

    LOCAL = "local"
    HALO_READ = "halo_read"
    GLOBAL = "global"


class NoBlockOpJax:
    """
    User-facing base class for custom JAX block operations.

    `NoBlockOpJax` supports two authoring styles.

    1. Block-wide methods
       Operate directly on the full authoritative Block state (packed or
       stacked). Use this when the operation is naturally expressed on the
       entire Block, for example contact across rods or transforms that index
       packed ghosts.

    2. Per-rod methods
       Operate on one Rod-local view at a time. During `finalize()`, the Block
       batches them with ``map_rods``. Use this when the logic is conceptually
       "same operator for every rod".

    The simulator-side lowering logic decides how to execute the operator:

    - `jax_block_operate_*` methods receive Block state as-is (no projection).
    - `jax_per_rod_operate_*` methods are applied through ``Block.map_rods``.

    Notes
    -----
    Do not implement both block-wide and per-rod variants for the same stage in
    one class. Choose one representation per stage.

    Implement any subset of:
    - `jax_block_operate_constrain_values(state, time)`
    - `jax_block_operate_synchronize(state, time)`
    - `jax_block_operate_constrain_rates(state, time)`
    - `jax_per_rod_operate_constrain_values(rod_view, time)`
    - `jax_per_rod_operate_synchronize(rod_view, time)`
    - `jax_per_rod_operate_constrain_rates(rod_view, time)`
    """

    communication_scope: ClassVar[CommunicationScope] = CommunicationScope.LOCAL
    halo_fields: ClassVar[tuple[str, ...]] = ()

    def jax_block_operate_constrain_values(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        """Apply a block-wide value constraint stage."""
        return state

    def jax_block_operate_synchronize(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        """Apply a block-wide synchronize stage."""
        return state

    def jax_block_operate_constrain_rates(
        self,
        state: JAXBlockState,
        time: JAXTime,
    ) -> JAXBlockState:
        """Apply a block-wide rate constraint stage."""
        return state

    def jax_per_rod_operate_constrain_values(
        self,
        rod_view: RodView,
        time: JAXTime,
    ) -> RodView:
        """Apply a per-rod value constraint; Block.map_rods batches across rods."""
        return rod_view

    def jax_per_rod_operate_synchronize(
        self,
        rod_view: RodView,
        time: JAXTime,
    ) -> RodView:
        """Apply a per-rod synchronize stage; Block.map_rods batches across rods."""
        return rod_view

    def jax_per_rod_operate_constrain_rates(
        self,
        rod_view: RodView,
        time: JAXTime,
    ) -> RodView:
        """Apply a per-rod rate constraint; Block.map_rods batches across rods."""
        return rod_view
