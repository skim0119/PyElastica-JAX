"""Protocols for Cosserat rod memory-block factories."""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Iterator

from elastica.typing import RodType, SystemIdxType


class RodViewMetadata(Protocol):
    def slice_for_attr(self, attr: str) -> slice: ...


class RodView(Protocol):
    def __getattr__(self, attr: str) -> object: ...
    def __setattr__(self, attr: str, value: object) -> None: ...
    def commit(self) -> dict[str, object]: ...


@runtime_checkable
class RodBlockProtocol(Protocol):
    """
    Pre-configured rod block for ``enable_block_supports``.

    Returned by ``configure_rod_block*``. PyElastica builds the block by
    calling the instance as ``block(systems, system_idx_list)`` during
    ``finalize()``. The same object then appears in ``final_systems()``.
    """

    def __call__(
        self,
        systems: list[RodType],
        system_idx_list: list[SystemIdxType],
    ) -> RodBlockProtocol:
        """Pack ``systems`` into this block and return ``self``."""
        ...

    def iterate_rods(self) -> Iterator[RodView]:
        """Iterate over the rods in the block."""
        ...
