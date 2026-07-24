#!/bin/sh
#SBATCH --job-name=rod_contact_gpu2x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --partition=gh
#SBATCH --output=logs/slurm_rod_contact_gpu2x_%j.out
#SBATCH --error=logs/slurm_rod_contact_gpu2x_%j.err

# Single-node two-GPU rod-rod contact scaling.
# Vertical path: one stacked block sharded across both GPUs (ticket 03/04).
# Horizontal path (MPI 2 ranks + halo CapsuleContact) waits on ticket 05.
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
    --min-exp 1 \
    --max-exp 10 \
    --steps 200 \
    --warmup-runs 1 \
    --output "output/rod_contact_gpu2x_vertical.png" \
    --csv-output "output/rod_contact_gpu2x_vertical.csv"

echo "horizontal two-GPU (MPI+halo) skipped until ticket 05 lands"
