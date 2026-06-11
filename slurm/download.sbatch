#!/usr/bin/env bash
#SBATCH --job-name=era5-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-download-%j.out
#
# Download regional ERA5 from the CDS on a login/compute node with internet.
# Submit with:  sbatch slurm/download.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Activate your env here, e.g.:  source .venv/bin/activate
PY=${PYTHON:-.venv/bin/python}
$PY scripts/01_download_era5.py --concurrency 4
