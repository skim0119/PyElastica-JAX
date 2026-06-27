#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="${SCRIPT_DIR}/jax_snake_throughput.py"
COMMON_ARGS=(--steps 1000)

echo "CUDA snake throughput sweep (n-snakes-exp 6..14)"
for exponent in {6..14}; do
    echo "CUDA n-snakes-exp=${exponent}"
    uv run --no-sync python "${BENCHMARK}" \
        "${COMMON_ARGS[@]}" \
        --backend cuda \
        --n-snakes-exp "${exponent}"
done

echo "PyElastica snake throughput sweep (n-snakes-exp 6..12)"
for exponent in {6..12}; do
    echo "PyElastica n-snakes-exp=${exponent}"
    uv run --no-sync python "${BENCHMARK}" \
        "${COMMON_ARGS[@]}" \
        --backend pyelastica \
        --n-snakes-exp "${exponent}"
done
