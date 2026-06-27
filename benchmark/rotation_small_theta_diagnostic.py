"""Small-theta diagnostic for Numba vs JAX rotation kernels."""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import elastica_jax as eaj

jax.config.update("jax_enable_x64", True)

from elastica._jax_linalg import _jax_batch_matmul
from elastica._jax_rotations import _jax_get_rotation_matrix, _jax_inv_rotate
from elastica._linalg import _batch_matmul
from elastica._rotations import _get_rotation_matrix, _inv_rotate


def _make_director() -> np.ndarray:
    axis = np.asarray([0.3, -0.4, 0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = np.float64(0.7)
    rotation = _get_rotation_matrix(angle, axis.reshape(3, 1))[:, :, 0]
    return rotation.reshape(3, 3, 1)


def _compute_errors(
    theta_values: np.ndarray,
    *,
    device: jax.Device,
) -> dict[str, np.ndarray]:
    axis = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    director = _make_director()

    rotation_error = np.zeros_like(theta_values)
    director_error = np.zeros_like(theta_values)
    kappa_error = np.zeros_like(theta_values)

    with jax.default_device(device):
        for idx, theta in enumerate(theta_values):
            scaled_axis = (axis * theta).reshape(3, 1)

            rotation_numba = _get_rotation_matrix(np.float64(1.0), scaled_axis)
            rotation_jax = np.asarray(
                _jax_get_rotation_matrix(
                    np.float64(1.0),
                    jnp.asarray(scaled_axis, dtype=np.float64),
                )
            )
            rotation_error[idx] = np.max(np.abs(rotation_numba - rotation_jax))

            director_numba = _batch_matmul(rotation_numba, director)
            director_jax = np.asarray(
                _jax_batch_matmul(
                    jnp.asarray(rotation_jax, dtype=np.float64),
                    jnp.asarray(director, dtype=np.float64),
                )
            )
            director_error[idx] = np.max(np.abs(director_numba - director_jax))

            director_pair_numba = np.concatenate((director, director_numba), axis=2)
            director_pair_jax = np.concatenate((director, director_jax), axis=2)
            kappa_numba = _inv_rotate(director_pair_numba)
            kappa_jax = np.asarray(
                _jax_inv_rotate(jnp.asarray(director_pair_jax, dtype=np.float64))
            )
            kappa_error[idx] = np.max(np.abs(kappa_numba - kappa_jax))

    return {
        "rotation_error": rotation_error,
        "director_error": director_error,
        "kappa_error": kappa_error,
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--n-samples", type=int, default=200, show_default=True)
@click.option(
    "--backend",
    type=click.Choice(("cpu", "cuda"), case_sensitive=False),
    default="cpu",
    show_default=True,
    help="JAX backend to run the diagnostic on.",
)
def main(n_samples: int, backend: str) -> None:
    output = Path("rotation_small_theta_diagnostic.png")

    device = eaj.resolve_backend_devices(backend)[0]
    print(f"backend: {backend}")
    print(f"device: {device}")

    theta_values = np.logspace(-16, -2, n_samples, dtype=np.float64)
    errors = _compute_errors(theta_values, device=device)

    one_minus_cos = 2.0 * np.sin(theta_values / 2.0) ** 2
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

    axes[0].loglog(theta_values, one_minus_cos, label=r"$1-\cos(\theta)$")
    axes[0].loglog(theta_values, theta_values**2 / 2.0, "--", label=r"$\theta^2/2$")
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel("magnitude")
    axes[0].set_title("Small-angle scale")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].loglog(theta_values, errors["rotation_error"], label="rotation matrix")
    axes[1].loglog(theta_values, errors["director_error"], label="director update")
    axes[1].loglog(theta_values, errors["kappa_error"], label="kappa")
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel("max abs discrepancy")
    axes[1].set_title("Numba vs JAX discrepancy")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote plot: {output}")

    for name, values in errors.items():
        max_idx = int(np.argmax(values))
        print(f"{name}: max={values[max_idx]!r} at theta={theta_values[max_idx]!r}")


if __name__ == "__main__":
    main()
