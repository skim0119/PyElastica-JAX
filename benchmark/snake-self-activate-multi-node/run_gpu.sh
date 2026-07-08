#!/bin/sh
#SBATCH --job-name=multi_node_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=gh-dev
#SBATCH --output=logs/slurm_multi_node_gpu_%j.out
#SBATCH --error=logs/slurm_multi_node_gpu_%j.err

set -euo pipefail

source ~/localrc.sh

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
REPO_ROOT="${SCRATCH}/PyElastica-JAX"

# Match run.sh --ntasks-per-node so one GPU rank equals one CPU node of work.
CPU_RANKS_PER_NODE=144

setup_start=$SECONDS
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "
        source ~/localrc.sh
        cd \"\${REPO_ROOT}\"
        uv sync --python 3.11
        source \"\${UV_PROJECT_ENVIRONMENT}/bin/activate\"
        uv pip install -U "jax[cuda13]"
        uv pip install -U "numba"
    "
echo "node venv setup finished in $((SECONDS - setup_start))s"

MAX_NODES=${SLURM_NNODES}
MAX_MPI=${SLURM_NTASKS}

MPI_SIZES=()
for ((mpi_size=1; mpi_size<=MAX_MPI; mpi_size+=1)); do
    MPI_SIZES+=("${mpi_size}")
done
IFS=,

cd "${REPO_ROOT}/benchmark/snake-self-activate-multi-node"
"${PYTHON_BIN}" "sweep_jax_snake_mpi_throughput.py" \
    --backend cuda \
    --mpi-sizes "${MPI_SIZES[*]}" \
    --snakes-per-rank-exp 6 \
    --snakes-per-rank-multiplier "${CPU_RANKS_PER_NODE}" \
    --steps 1000 \
    --warmup-runs 5 \
    --python "${PYTHON_BIN}" \
    --output "scaling_plot_gpu_N${MAX_NODES}.png"
