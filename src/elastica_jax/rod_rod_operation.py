from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np

JAXTime: TypeAlias = np.float64


class NoRodRodBlockOpJax:
    """
    User-facing base class for JAX rod-to-rod interaction operators.

    During ``finalize()``, the mixin injects:

    - ``_first_system``: the first ``CosseratRod`` in the pair
    - ``_second_system``: the second ``CosseratRod`` in the pair

    Implement ``jax_operation(rod_one_view, rod_two_view, time)`` and return
    the updated views, or ``None`` to commit in-place mutations on the views
    passed in.

    Both rods must currently reside in the same ``MemoryBlockCosseratRodJax`` on
    one device.
    """

    def jax_operation(
        self,
        rod_one_view: Any,
        rod_two_view: Any,
        time: JAXTime,
    ) -> tuple[Any, Any] | None:
        """Apply a rod-to-rod interaction on paired rod-local views."""
        return rod_one_view, rod_two_view
