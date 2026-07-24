#!/bin/sh
#SBATCH --job-name=rod_contact_mpi_cpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH --partition=gg
#SBATCH --output=logs/slurm_rod_contact_mpi_cpu_%j.out
#SBATCH --error=logs/slurm_rod_contact_mpi_cpu_%j.err

# Multi-node rod-rod contact weak scaling on CPU ranks.
# Emits separate CSV/PNG for horizontal and vertical layouts.
#
# Review-only until package blockers land:
# - ticket 05: MPI halo CapsuleContact (cross-rank coupling)
# - ticket 01: vertical CapsuleContact (stacked layout kinematics)
# Until then, timings measure rank-local contact and vertical may fail.

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
    "
echo "node venv setup finished in $((SECONDS - setup_start))s"

MAX_NODES=${SLURM_NNODES}
MAX_MPI=${SLURM_NTASKS}

MPI_SIZES=()
# Step by ranks-per-node so each point uses whole nodes.
RANKS_PER_NODE=${SLURM_NTASKS_PER_NODE}
for ((mpi_size=RANKS_PER_NODE; mpi_size<=MAX_MPI; mpi_size+=RANKS_PER_NODE)); do
    MPI_SIZES+=("${mpi_size}")
done
IFS=,

cd "${BENCH_DIR}"

"${PYTHON_BIN}" "sweep_jax_rod_contact_mpi_throughput.py" \
    --backend cpu \
    --mpi-sizes "${MPI_SIZES[*]}" \
    --rods-per-rank-exp 3 \
    --steps 200 \
    --warmup-runs 1 \
    --python "${PYTHON_BIN}" \
    --output "output/rod_contact_mpi_cpu_N${MAX_NODES}_horizontal.png"

"${PYTHON_BIN}" "sweep_jax_rod_contact_mpi_throughput.py" \
    --backend cpu \
    --mpi-sizes "${MPI_SIZES[*]}" \
    --rods-per-rank-exp 3 \
    --steps 200 \
    --warmup-runs 1 \
    --vertical \
    --python "${PYTHON_BIN}" \
    --output "output/rod_contact_mpi_cpu_N${MAX_NODES}_vertical.png"
