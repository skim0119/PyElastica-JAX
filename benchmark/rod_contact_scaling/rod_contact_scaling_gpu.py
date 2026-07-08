"""GPU weak-scaling benchmark for rod-rod contact."""

from __future__ import annotations

from _rod_contact_scaling_sweep import scaling_cli

main = scaling_cli(
    backend="cuda",
    label="jax-cuda",
    default_plot="output/rod_contact_scaling_gpu.png",
)

if __name__ == "__main__":
    main()
