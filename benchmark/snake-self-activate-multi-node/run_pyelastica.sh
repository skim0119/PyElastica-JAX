#!/bin/sh
#SBATCH --job-name=multi_node_pyelastica
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=144
#SBATCH --time=24:00:00
#SBATCH --partition=gg
#SBATCH --output=logs/slurm_multi_node_pyelastica_%j.out
#SBATCH --error=logs/slurm_multi_node_pyelastica_%j.err

set -euo pipefail

source ~/localrc.sh

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
REPO_ROOT="${SCRATCH}/PyElastica-JAX"

setup_start=$SECONDS
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "
        source ~/localrc.sh
        cd \"\${REPO_ROOT}\"
        uv sync --python 3.11
        source \"\${UV_PROJECT_ENVIRONMENT}/bin/activate\"
        uv pip install -U numba
    "
echo "node venv setup finished in $((SECONDS - setup_start))s"

MAX_NODES=${SLURM_NNODES}
MAX_MPI=${SLURM_NTASKS}

MPI_SIZES=()
for ((mpi_size=32; mpi_size<=MAX_MPI; mpi_size+=32)); do
    MPI_SIZES+=("${mpi_size}")
done
IFS=,

cd "${REPO_ROOT}/benchmark/snake-self-activate-multi-node"
"${PYTHON_BIN}" "sweep_jax_snake_mpi_throughput.py" \
    --backend pyelastica \
    --mpi-sizes "${MPI_SIZES[*]}" \
    --snakes-per-rank-exp 6 \
    --steps 1000 \
    --warmup-runs 5 \
    --python "${PYTHON_BIN}" \
    --output "scaling_plot_pyelastica_N${MAX_NODES}.png"
