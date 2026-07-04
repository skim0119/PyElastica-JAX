#SBATCH --job-name=multi_node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=144
#SBATCH --time=02:00:00
#SBATCH --partition=development
#SBATCH --output=logs/slurm_multi_node_%j.out
#SBATCH --error=logs/slurm_multi_node_%j.err

set -euo pipefail

source ~/localrc.sh

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
REPO_ROOT="${SCRATCH}/PyElastica-JAX"

setup_start=$SECONDS
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "
        source ~/localrc.sh
        cd \"\${REPO_ROOT}\"
        source \"\${UV_PROJECT_ENVIRONMENT}/bin/activate\"
    "
echo "node venv setup finished in $((SECONDS - setup_start))s"

ibrun -n "${SLURM_NTASKS}" \
    "${PYTHON_BIN}" "sweep_jax_snake_mpi_throughput.py" \
        --mpi-sizes "1,2,4,8" \
        --snakes-per-rank-exp 8 \
        --steps 1000 \
        --warmup-runs 5
