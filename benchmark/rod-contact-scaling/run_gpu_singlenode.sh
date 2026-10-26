#!/bin/sh
#SBATCH --job-name=rod_contact_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gh
#SBATCH --output=logs/slurm_rod_contact_gpu_%j.out
#SBATCH --error=logs/slurm_rod_contact_gpu_%j.err

# Single-node rod-rod contact scaling on one GPU.
# Cases: JAX CUDA horizontal; JAX CUDA vertical.
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

"${PYTHON_BIN}" "rod_contact_scaling_all.py" \
    --skip-pyelastica \
    --skip-cpu \
    --min-exp 1 \
    --max-exp 10 \
    --steps 200 \
    --warmup-runs 1 \
    --output "output/rod_contact_singlenode_gpu.png" \
    --csv-output "output/rod_contact_singlenode_gpu.csv"
