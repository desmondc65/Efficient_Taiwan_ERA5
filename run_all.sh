#!/usr/bin/env bash
# End-to-end driver. By default it runs on SYNTHETIC data so you can validate the
# whole pipeline with no CDS key. Set SOURCE=cds to download real ERA5 instead.
#
#   ./run_all.sh                 # synthetic (default), no key needed
#   SOURCE=cds ./run_all.sh      # download real ERA5 from the CDS first
#   STEPS=240 ./run_all.sh       # smaller synthetic dataset (faster)
#   SKIP_TORCH=1 ./run_all.sh    # skip dataloader/DDP (no PyTorch installed)
#
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
SOURCE=${SOURCE:-synthetic}
STEPS=${STEPS:-744}
SKIP_TORCH=${SKIP_TORCH:-0}

echo "### 00 setup check"
$PY scripts/00_check_setup.py || true

if [ "$SOURCE" = "cds" ]; then
  echo "### 01 download ERA5 from CDS"
  $PY scripts/01_download_era5.py
else
  echo "### 02 generate synthetic ERA5 (${STEPS} steps)"
  $PY scripts/02_make_synthetic.py --steps "$STEPS"
fi

echo "### 03 build production zarr"
$PY scripts/03_build_zarr.py

echo "### 04 codec benchmark"
$PY scripts/04_bench_codecs.py

echo "### 05 chunking benchmark (keeping stores for 06)"
$PY scripts/05_bench_chunking.py --keep

if [ "$SKIP_TORCH" != "1" ] && $PY -c "import torch" 2>/dev/null; then
  echo "### 06 dataloader benchmark"
  $PY scripts/06_bench_dataloader.py
  echo "### 07 DDP benchmark (single node, all visible GPUs)"
  NGPU=$($PY -c "import torch;print(max(1,torch.cuda.device_count()))" 2>/dev/null || echo 1)
  torchrun --standalone --nproc_per_node="$NGPU" scripts/07_bench_ddp.py || \
    $PY scripts/07_bench_ddp.py
else
  echo "### 06/07 skipped (PyTorch not available or SKIP_TORCH=1)"
fi

echo "### 08 report"
$PY scripts/08_report.py --plots || $PY scripts/08_report.py

echo "### done. See results/ for CSV/JSONL metrics and results/plots/ for figures."
