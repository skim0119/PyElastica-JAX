#!/bin/sh
#SBATCH --job-name=rod_contact_gpu2x
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --partition=gh
#SBATCH --output=logs/slurm_rod_contact_gpu2x_%j.out
#SBATCH --error=logs/slurm_rod_contact_gpu2x_%j.err

# Single-node two-GPU rod-rod contact scaling (horizontal + vertical).
# - vertical: one stacked block sharded across both GPUs (shard_map)
# - horizontal: two MPI ranks + halo CapsuleContact (one GPU per rank)
# Copy this script into the job folder and edit REPO_ROOT / partition as needed.

set -euo pipefail

source ~/localrc.sh

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
REPO_ROOT="${SCRATCH}/PyElastica-JAX"
BENCH_DIR="${REPO_ROOT}/benchmark/rod-contact-scaling"

mkdir -p logs
mkdir -p "${BENCH_DIR}/output"

cd "${REPO_ROOT}"
uv sync --python 3.11
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"

cd "${BENCH_DIR}"

"${PYTHON_BIN}" "sweep_jax_rod_contact_gpu2x_throughput.py" \
    --backend cuda \
    --layout both \
    --min-exp 1 \
    --max-exp 10 \
    --steps 200 \
    --warmup-runs 1 \
    --python "${PYTHON_BIN}" \
    --output "output/rod_contact_gpu2x.png" \
    --csv-output "output/rod_contact_gpu2x.csv"
