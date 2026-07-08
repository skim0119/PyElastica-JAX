"""CPU weak-scaling benchmark for rod-rod contact."""

from __future__ import annotations

from _rod_contact_scaling_sweep import scaling_cli

main = scaling_cli(
    backend="cpu",
    label="jax-cpu",
    default_plot="output/rod_contact_scaling_cpu.png",
)

if __name__ == "__main__":
    main()
