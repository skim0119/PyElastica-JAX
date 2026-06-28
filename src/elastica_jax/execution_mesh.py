"""Execution mesh for rod sharding across local workers."""

from __future__ import annotations


class ExecutionMesh:
    """
    Placeholder for an explicit rod-to-device placement policy.

    Today sharded blocks only need a JAX device list. ``configure_rod_block_sharded``
    accepts ``devices=...`` and assigns rods with a balanced
    ``rod_index % n_shards`` map when the block is finalized.

    This class may return later if we need:

    - custom placement (contiguous chunks, pinned rods, uneven splits)
    - checkpoint round-trips of a non-default ``rod_to_shard`` layout
    - placement decided before ``finalize()`` and frozen afterward

    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ExecutionMesh is reserved for a future explicit rod-placement API. "
            "For now, pass ``devices`` to ``configure_rod_block_sharded``; rods are "
            "split evenly across those devices at ``finalize()``."
        )
