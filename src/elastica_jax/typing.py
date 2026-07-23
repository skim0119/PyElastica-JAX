"""Shared array typing aliases backed by ``jaxtyping``.

Prefer these aliases over bare ``jax.Array`` / ``np.ndarray`` in public JAX
APIs so static checkers accept both device arrays and host scalars/arrays that
JAX will promote.
"""

from __future__ import annotations

from typing import TypeAlias

from jaxtyping import Array, ArrayLike, Bool, Float, Int, Real, Scalar, Shaped

# Common Cosserat layouts (shape strings are documentation for readers /
# runtime jaxtyping checks; mypy treats them as Array).
Vector3: TypeAlias = Float[Array, "3"]
Nodes3: TypeAlias = Float[Array, "3 n_nodes"]
Elements3: TypeAlias = Float[Array, "3 n_elems"]
Directors: TypeAlias = Float[Array, "3 3 n_elems"]
NodeScalars: TypeAlias = Float[Array, " n_nodes"]
ElementScalars: TypeAlias = Float[Array, " n_elems"]

__all__ = [
    "Array",
    "ArrayLike",
    "Bool",
    "Directors",
    "ElementScalars",
    "Elements3",
    "Float",
    "Int",
    "NodeScalars",
    "Nodes3",
    "Real",
    "Scalar",
    "Shaped",
    "Vector3",
]
