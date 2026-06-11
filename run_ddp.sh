#!/usr/bin/env bash
# Single-node DDP launcher. Sweeps world sizes to produce a strong-scaling curve.
#
#   ./run_ddp.sh                 # sweep 1..NGPU on this node
#   ./run_ddp.sh 1 2 4           # explicit world sizes
#   CHUNKING=daily ./run_ddp.sh  # benchmark a specific chunking store
#
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
CHUNKING=${CHUNKING:-}
ARGS=()
[ -n "$CHUNKING" ] && ARGS+=(--chunking "$CHUNKING")

MAXGPU=$($PY -c "import torch;print(max(1,torch.cuda.device_count()))")
if [ "$#" -gt 0 ]; then
  SIZES=("$@")
else
  SIZES=()
  n=1; while [ "$n" -le "$MAXGPU" ]; do SIZES+=("$n"); n=$((n*2)); done
fi

for ws in "${SIZES[@]}"; do
  if [ "$ws" -gt "$MAXGPU" ]; then
    echo "skip world_size=$ws (> $MAXGPU GPUs)"; continue
  fi
  echo "### DDP world_size=$ws"
  $PY -m torch.distributed.run --standalone --nproc_per_node="$ws" scripts/07_bench_ddp.py "${ARGS[@]}"
done

$PY scripts/08_report.py
