#!/usr/bin/env bash
#SBATCH --job-name=era5-ddp
#SBATCH --nodes=2                  # <-- number of nodes
#SBATCH --ntasks-per-node=1        # one launcher (torchrun) per node
#SBATCH --gpus-per-node=4          # <-- GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out
#
# Multi-node DDP benchmark over the Taiwan ERA5 Zarr store.
# Submit with:  sbatch slurm/ddp_multinode.sbatch
# Adjust the SBATCH directives + module/venv lines for your cluster.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- environment (edit for your cluster) ---
# module load cuda/12.4
# Activate your env here, e.g.:  source .venv/bin/activate
PY=${PYTHON:-.venv/bin/python}

GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
NNODES=$SLURM_NNODES
# Rendezvous on the first node in the allocation.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "nodes=$NNODES gpus_per_node=$GPUS_PER_NODE master=$MASTER_ADDR"

# One torchrun per node (ntasks-per-node=1); torchrun spawns GPUS_PER_NODE workers.
srun --kill-on-bad-exit=1 bash -c "
  $PY -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=\$SLURM_NODEID \
    --rdzv_id=\$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/07_bench_ddp.py --num-workers $SLURM_CPUS_PER_TASK
"

$PY scripts/08_report.py
