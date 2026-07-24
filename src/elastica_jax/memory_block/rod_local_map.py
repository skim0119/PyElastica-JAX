"""Rod-local projection helpers for Block.map_rods."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import jax
import jax.numpy as jnp

from elastica_jax.memory_block.syncable_attrs import (
    _ELEMENT_ATTRS,
    _NODE_ATTRS,
    _SYNCABLE_ATTRS,
    _VORONOI_ATTRS,
)
from elastica_jax.memory_block.protocol import (
    JAXBlockState,
    PackedRodIndexLayout,
    RodLocalOp,
    RodView,
)
from elastica_jax.typing import Array


class RodLocalState:
    """Transient rod-shaped projection used by ``map_rods`` operators."""

    def __init__(
        self,
        state: JAXBlockState,
        *,
        updates: JAXBlockState | None = None,
    ) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_updates", {} if updates is None else dict(updates))

    def __getattr__(self, attr: str) -> Array:
        if attr.startswith("_"):
            raise AttributeError(attr)
        updates: JAXBlockState = object.__getattribute__(self, "_updates")
        state: JAXBlockState = object.__getattribute__(self, "_state")
        if attr in updates:
            return updates[attr]
        return state[attr]

    def __setattr__(self, attr: str, value: Array) -> None:
        if attr.startswith("_"):
            object.__setattr__(self, attr, value)
            return
        updates: JAXBlockState = object.__getattribute__(self, "_updates")
        updates[attr] = value

    def commit(self) -> JAXBlockState:
        """Return the projected collections with pending updates applied.

        Returns
        -------
        JAXBlockState
            Rod-local collections with pending attribute updates applied.
        """
        state: JAXBlockState = object.__getattribute__(self, "_state")
        updates: JAXBlockState = object.__getattribute__(self, "_updates")
        updated = dict(state)
        updated.update(updates)
        return updated


_ATTR_DOMAINS = (
    {attr: "node" for attr in _NODE_ATTRS}
    | {attr: "element" for attr in _ELEMENT_ATTRS}
    | {attr: "voronoi" for attr in _VORONOI_ATTRS}
)


def _uniform_index_matrix(
    start_idx: np.ndarray,
    end_idx: np.ndarray,
) -> np.ndarray:
    """Return a dense per-rod index matrix for packed block domains."""
    widths = end_idx - start_idx
    assert np.all(widths == widths[0]), (
        "Per-rod Block operators require uniform discretization across rods."
    )
    offsets = np.arange(int(widths[0]), dtype=np.int32)
    return start_idx[:, None].astype(np.int32) + offsets[None, :]


def _gather_attr(array: Array, indices: Array) -> Array:
    if array.ndim == 1:
        return jnp.take(array, indices, axis=-1)
    if array.ndim == 2:
        return jnp.moveaxis(jnp.take(array, indices, axis=-1), 1, 0)
    if array.ndim == 3:
        return jnp.moveaxis(jnp.take(array, indices, axis=-1), 2, 0)
    assert False, f"Unsupported array rank {array.ndim} for per-rod batching."


def _scatter_attr(
    array: Array,
    indices: Array,
    values: Array,
) -> Array:
    if array.ndim == 1:
        return array.at[indices].set(values)
    if array.ndim == 2:
        return array.at[:, indices].set(jnp.moveaxis(values, 0, 1))
    if array.ndim == 3:
        return array.at[:, :, indices].set(jnp.moveaxis(values, 0, 2))
    assert False, f"Unsupported array rank {array.ndim} for per-rod batching."


def _commit_view(
    updated_view: RodView | JAXBlockState,
    rod_view: RodLocalState,
) -> JAXBlockState:
    if isinstance(updated_view, RodLocalState):
        return updated_view.commit()
    if isinstance(updated_view, dict):
        return updated_view
    return rod_view.commit()


def _normalize_ops(
    op: RodLocalOp | Sequence[RodLocalOp],
    n_rods: int,
) -> tuple[RodLocalOp, ...]:
    if isinstance(op, Sequence) and not callable(op):
        ops = tuple(op)
    else:
        ops = (op,)  # type: ignore[assignment]
    if len(ops) == 1:
        ops = tuple(ops[0] for _ in range(n_rods))
    assert len(ops) == n_rods, f"Expected {n_rods} per-rod operators, got {len(ops)}."
    return ops


def _ops_are_identical(ops: tuple[RodLocalOp, ...]) -> bool:
    """Return True when every entry is the same callable object (``is``)."""
    first = ops[0]
    return all(operator is first for operator in ops)


def _apply_shared_op_vmap(
    batch: JAXBlockState,
    operator: RodLocalOp,
    time: np.float64 | float,
) -> JAXBlockState:
    def apply_one(single_state: JAXBlockState, t: np.float64 | float) -> JAXBlockState:
        rod_view = RodLocalState(dict(single_state))
        return _commit_view(operator(rod_view, t), rod_view)

    return jax.vmap(apply_one, in_axes=(0, None))(batch, time)


def _apply_ops_python_loop(
    batch: JAXBlockState,
    ops: tuple[RodLocalOp, ...],
    time: np.float64 | float,
) -> JAXBlockState:
    attrs = tuple(batch.keys())
    updated_rods: list[JAXBlockState] = []
    for rod_index, operator in enumerate(ops):
        single_state = {attr: batch[attr][rod_index] for attr in attrs}
        rod_view = RodLocalState(single_state)
        updated_view = operator(rod_view, time)
        updated_rods.append(_commit_view(updated_view, rod_view))
    return {
        attr: jnp.stack([rod[attr] for rod in updated_rods], axis=0) for attr in attrs
    }


def _apply_ops_to_batch(
    batch: JAXBlockState,
    ops: tuple[RodLocalOp, ...],
    time: np.float64 | float,
) -> JAXBlockState:
    if _ops_are_identical(ops):
        return _apply_shared_op_vmap(batch, ops[0], time)
    return _apply_ops_python_loop(batch, ops, time)


def map_rods_packed(
    block: PackedRodIndexLayout,
    state: JAXBlockState,
    op: RodLocalOp | Sequence[RodLocalOp],
    time: np.float64 | float,
) -> JAXBlockState:
    """Project packed horizontal Block state, apply rod ops, write back.

    Parameters
    ----------
    block :
        Packed Block providing per-rod index spans.
    state :
        Authoritative packed Block state.
    op :
        One shared Rod-local operator, or one operator per rod.
    time :
        Simulation time passed through to ``op``.

    Returns
    -------
    JAXBlockState
        Packed Block state after rod-local updates are scattered back.
    """
    indices = {
        "node": jnp.asarray(
            _uniform_index_matrix(
                block.start_idx_in_rod_nodes,
                block.end_idx_in_rod_nodes,
            )
        ),
        "element": jnp.asarray(
            _uniform_index_matrix(
                block.start_idx_in_rod_elems,
                block.end_idx_in_rod_elems,
            )
        ),
        "voronoi": jnp.asarray(
            _uniform_index_matrix(
                block.start_idx_in_rod_voronoi,
                block.end_idx_in_rod_voronoi,
            )
        ),
    }
    attrs = tuple(_SYNCABLE_ATTRS)
    batch = {
        attr: _gather_attr(state[attr], indices[_ATTR_DOMAINS[attr]]) for attr in attrs
    }
    n_rods = int(next(iter(batch.values())).shape[0])
    ops = _normalize_ops(op, n_rods)
    updated_batch = _apply_ops_to_batch(batch, ops, time)
    updated_state = dict(state)
    for attr in attrs:
        updated_state[attr] = _scatter_attr(
            state[attr],
            indices[_ATTR_DOMAINS[attr]],
            updated_batch[attr],
        )
    return updated_state


def map_rods_stacked(
    state: JAXBlockState,
    op: RodLocalOp | Sequence[RodLocalOp],
    time: np.float64 | float,
) -> JAXBlockState:
    """Apply rod ops on stacked (leading-rod-axis) Block state.

    Parameters
    ----------
    state :
        Authoritative stacked Block state.
    op :
        One shared Rod-local operator, or one operator per rod.
    time :
        Simulation time passed through to ``op``.

    Returns
    -------
    JAXBlockState
        Stacked Block state after rod-local updates.
    """
    attrs = tuple(_SYNCABLE_ATTRS)
    batch = {attr: state[attr] for attr in attrs}
    n_rods = int(next(iter(batch.values())).shape[0])
    ops = _normalize_ops(op, n_rods)
    updated_batch = _apply_ops_to_batch(batch, ops, time)
    updated_state = dict(state)
    updated_state.update(updated_batch)
    return updated_state
