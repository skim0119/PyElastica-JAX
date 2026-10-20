#!/bin/sh
#SBATCH --job-name=rod_contact_mpi_gpu
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=gh
#SBATCH --output=logs/slurm_rod_contact_mpi_gpu_%j.out
#SBATCH --error=logs/slurm_rod_contact_mpi_gpu_%j.err

# Multi-node rod-rod contact weak scaling: one GPU per node.
# Emits separate CSV/PNG for horizontal and vertical layouts.
# Uses MPI ranks with halo CapsuleContact (HALO_READ) so contact stays coupled.
# Copy this script into the job folder and edit REPO_ROOT / partition as needed.

set -euo pipefail

source ~/localrc.sh

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
REPO_ROOT="${SCRATCH}/PyElastica-JAX"
BENCH_DIR="${REPO_ROOT}/benchmark/rod-contact-scaling"

mkdir -p logs
mkdir -p "${BENCH_DIR}/output"

setup_start=$SECONDS
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "
        source ~/localrc.sh
        cd \"\${REPO_ROOT}\"
        uv sync --python 3.11
        source \"\${UV_PROJECT_ENVIRONMENT}/bin/activate\"
        uv pip install -U \"jax[cuda13]\"
        uv pip install -U numba
    "
echo "node venv setup finished in $((SECONDS - setup_start))s"

MAX_NODES=${SLURM_NNODES}
MAX_MPI=${SLURM_NTASKS}

MPI_SIZES=()
for ((mpi_size=1; mpi_size<=MAX_MPI; mpi_size+=1)); do
    MPI_SIZES+=("${mpi_size}")
done
IFS=,

cd "${BENCH_DIR}"

"${PYTHON_BIN}" "sweep_jax_rod_contact_mpi_throughput.py" \
    --backend cuda \
    --mpi-sizes "${MPI_SIZES[*]}" \
    --rods-per-rank-exp 4 \
    --steps 200 \
    --warmup-runs 1 \
    --python "${PYTHON_BIN}" \
    --output "output/rod_contact_mpi_gpu_N${MAX_NODES}_horizontal.png"

"${PYTHON_BIN}" "sweep_jax_rod_contact_mpi_throughput.py" \
    --backend cuda \
    --mpi-sizes "${MPI_SIZES[*]}" \
    --rods-per-rank-exp 4 \
    --steps 200 \
    --warmup-runs 1 \
    --vertical \
    --python "${PYTHON_BIN}" \
    --output "output/rod_contact_mpi_gpu_N${MAX_NODES}_vertical.png"
