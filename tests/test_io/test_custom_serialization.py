"""Custom serializers must be honored or rejected before opening a file."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import h5py
import jax
import numpy as np
import pytest

import elastica as ea
import elastica_jax as eaj

jax.config.update("jax_enable_x64", True)


class CustomSerializedBlock(eaj._CosseratRodMemoryBlock):
    """Use a custom marker payload to identify serializer dispatch."""

    marker = 17

    def write_hdf5_state(self, handle: object, *, schema_level: int) -> None:
        """Write the custom marker to the supplied HDF5 handle."""
        cast(h5py.File, handle).attrs["custom_marker"] = self.marker

    def read_hdf5_state(
        self, handle: object, *, schema_level: int, check_device: bool = True
    ) -> None:
        """Load the custom marker from the supplied HDF5 handle."""
        self.marker = int(cast(h5py.File, handle).attrs["custom_marker"])


@pytest.fixture
def custom_simulation() -> tuple[eaj.Simulator, CustomSerializedBlock]:
    """Return a finalized one-rod simulator with a custom serializer."""
    simulator = eaj.Simulator()
    block = eaj.configure_rod_block(inner_block_cls=CustomSerializedBlock)
    simulator.enable_block_supports(ea.CosseratRod, block)
    simulator.append(
        ea.CosseratRod.straight_rod(
            4,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            0.5,
            0.01,
            1000.0,
            youngs_modulus=1.0e4,
        )
    )
    simulator.finalize()
    return simulator, block


@pytest.mark.parametrize("n_workers", [1, 2])
@pytest.mark.parametrize("target_simulator", [False, True])
def test_custom_serializer_dispatch(
    custom_simulation: tuple[eaj.Simulator, CustomSerializedBlock],
    tmp_path: Path,
    n_workers: int,
    target_simulator: bool,
) -> None:
    """Standalone serial Save delegates; unsupported paths fail explicitly."""
    simulator, block = custom_simulation
    target = simulator if target_simulator else block
    handle = MagicMock()
    handle.attrs = {}
    with (
        patch("elastica_jax.io.api.h5py.File") as open_file,
        patch.object(Path, "mkdir"),
    ):
        open_file.return_value.__enter__.return_value = handle
        if target_simulator or n_workers > 1:
            with pytest.raises(NotImplementedError, match="n_workers=1"):
                eaj.save(
                    target, tmp_path / "custom.h5", verbose=10, n_workers=n_workers
                )
            open_file.assert_not_called()
        else:
            eaj.save(block, tmp_path / "custom.h5", verbose=10)
            assert handle.attrs["custom_marker"] == 17
            block.marker = 0
            eaj.load(block, tmp_path / "custom.h5")
            assert block.marker == 17


@pytest.mark.parametrize("n_workers", [1, 2])
def test_custom_simulator_serializer_is_not_bypassed(
    custom_simulation: tuple[eaj.Simulator, CustomSerializedBlock],
    tmp_path: Path,
    n_workers: int,
) -> None:
    """A custom collection writer must run serially or fail before file creation."""

    class CustomSimulator(eaj.Simulator):
        """Add collection metadata through the existing serialization hook."""

        def write_hdf5_state(self, handle: h5py.File, *, schema_level: int) -> None:
            """Write standard payloads and a collection-specific marker."""
            super().write_hdf5_state(handle, schema_level=schema_level)
            handle.attrs["collection_marker"] = 23

    simulator = CustomSimulator()
    simulator.enable_block_supports(ea.CosseratRod, eaj.configure_rod_block())
    simulator.append(custom_simulation[1].rods()[0])
    simulator.finalize()
    handle = MagicMock()
    handle.attrs = {}
    with (
        patch("elastica_jax.io.api.h5py.File") as open_file,
        patch.object(Path, "mkdir"),
    ):
        open_file.return_value.__enter__.return_value = handle
        if n_workers > 1:
            with pytest.raises(NotImplementedError, match="n_workers=1"):
                eaj.save(simulator, tmp_path / "custom.h5", n_workers=n_workers)
            open_file.assert_not_called()
        else:
            eaj.save(simulator, tmp_path / "custom.h5")
            assert handle.attrs["collection_marker"] == 23
