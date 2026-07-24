#!/bin/sh
#SBATCH --job-name=rod_contact_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=gg
#SBATCH --output=logs/slurm_rod_contact_cpu_%j.out
#SBATCH --error=logs/slurm_rod_contact_cpu_%j.err

# Single-node rod-rod contact scaling on CPU.
# Cases: PyElastica; JAX CPU horizontal; JAX CPU vertical.
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
    --skip-gpu \
    --min-exp 1 \
    --max-exp 8 \
    --steps 200 \
    --warmup-runs 1 \
    --output "output/rod_contact_singlenode_cpu.png" \
    --csv-output "output/rod_contact_singlenode_cpu.csv"
