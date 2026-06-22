"""Synchronize periodic boundaries for circular rods in block memory."""


def _synchronize_periodic_boundary_of_vector_collection(input, periodic_idx):
    input[..., periodic_idx[0, :]] = input[..., periodic_idx[1, :]]


def _synchronize_periodic_boundary_of_matrix_collection(input, periodic_idx):
    input[..., periodic_idx[0, :]] = input[..., periodic_idx[1, :]]


def _synchronize_periodic_boundary_of_scalar_collection(input, periodic_idx):
    input[..., periodic_idx[0, :]] = input[..., periodic_idx[1, :]]
