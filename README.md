# Efficient Taiwan ERA5 → Zarr (Team 46)

A ready-to-run toolkit that turns global ERA5 into a **lightweight, chunked,
compressed Zarr dataset for the Taiwan region**, suitable for regional weather-ML
training (StormCast / CorrDiff-style), and **benchmarks the parallel/distributed
properties** that matter for HPC training: chunking strategy, compression codec,
multi-worker data loading, and multi-node multi-GPU (DDP) throughput.

> You only need a CDS API token to fetch real data. Everything else — including a
> **synthetic data generator** — runs offline so you can validate the full
> pipeline before downloading anything.

---

## Open dataset (download instead of the CDS)

A ready-to-train **3.42-year Taiwan ERA5 Zarr** (`data/zarr/taiwan_era5_full.zarr`,
hourly 2019-08 → 2022-12, 24 channels on a ~2 km 224×128 grid, `full_field`
chunking + lossless blosc-zstd5, 43.3 GB) is provided so you can skip the slow CDS
retrieval entirely. **See [`DATASET.md`](DATASET.md)** for the full contents and
copy-paste Python recipes to open it and extract variables.

```bash
python scripts/91_build_full_zarr.py --src <LowRes>   # (re)build the full store
```

---

## What each metric maps to (proposal → script)

| Proposal goal | Script | Records |
|---|---|---|
| Trim regional ERA5 → Zarr, **reduced storage footprint** | `03_build_zarr.py` | raw vs uncompressed vs zarr size, compression ratio, regional crop factor |
| **Download** ERA5, record **time** & throughput | `01_download_era5.py` | per-request + aggregate MB/s, wall time (parallel requests) |
| Profile **codecs**: compression vs data-loading time | `04_bench_codecs.py` | ratio, write/read MB/s, random-frame latency |
| **Chunking strategies** for weather-ML access patterns | `05_bench_chunking.py` | full-field/s vs point-series/s, #chunks, size |
| Chunking × **parallel DataLoader** workers | `06_bench_dataloader.py` | samples/s vs num_workers |
| **Multi-node multi-GPU** training | `07_bench_ddp.py` | global samples/s, per-GPU, data-bound %, strong scaling |
| Summary tables + figures | `08_report.py` | console tables, `results/plots/*.png` |

All scripts append metrics to `results/<name>.csv` and `results/<name>.jsonl`
(every row carries a hardware/library fingerprint for reproducibility).

---

## Quick start (no key needed)

```bash
# 1. install core deps into a venv (run_all.sh defaults to ./.venv)
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
#   or with conda:  conda env create -f environment.yml && conda activate era5tw

# 2. run the whole thing on synthetic data
./run_all.sh
#   -> generates synthetic ERA5, builds zarr, runs codec/chunking/dataloader/DDP
#      benchmarks, and prints a report. Add PyTorch for steps 06/07.
```

For the DataLoader/DDP steps install PyTorch matching your CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # RTX 3090
```

## Real ERA5 (needs a CDS token)

1. Make a free account at <https://cds.climate.copernicus.eu>, accept the ERA5
   licence, and copy your **Personal Access Token** from your profile.
2. Store it (writes `~/.cdsapirc` for the new CDS endpoint):
   ```bash
   python scripts/00_check_setup.py --write-key <YOUR_TOKEN>
   ```
3. Download + build:
   ```bash
   SOURCE=cds ./run_all.sh
   # or step by step:
   python scripts/01_download_era5.py     # parallel requests, timed
   python scripts/03_build_zarr.py        # -> data/zarr/taiwan_era5.zarr
   ```

---

## Configuration — `config.yaml`

One file controls everything: region/area (defaults to a **41×41** box centred on
Taiwan), time range (defaults to **one month** for a fast first run), variable
list (5 surface + 5×5 pressure-level = 30 channels), the **chunking strategies**
and **codecs** to benchmark, and the default "production" chunking/codec.

Widen the dataset by editing `time.months` / `time.years`. Change the benchmark
grids under `benchmark:`.

### Chunking strategies (time, lat, lon)
- `full_field (1,41,41)` — one timestep per chunk → fast random spatial fields (CNN/diffusion).
- `daily (24,41,41)`, `weekly (168,41,41)`, `monthly (720,41,41)` — coarser temporal blocks.
- `spatial_block (24,16,16)` — space-tiled.
- `timeseries (T,1,1)` — full series per point → fast for LSTM, slow for spatial fields.

### Codecs
`none`, `blosc_lz4`, `blosc_zstd{3,5,9}`, `zstd5`, `gzip5` (numcodecs).

---

## Layout

```
config.yaml              # single source of truth
src/era5tw/              # library: config, metrics, codecs, chunking, zarr IO, dataset
scripts/
  00_check_setup.py      # deps + config + credentials pre-flight
  01_download_era5.py    # CDS download, parallel requests, timed
  02_make_synthetic.py   # offline ERA5-like data generator
  03_build_zarr.py       # production zarr + storage-footprint report
  04_bench_codecs.py     # compression vs read/write speed
  05_bench_chunking.py   # access-pattern trade-off (the core result)
  06_bench_dataloader.py # DataLoader worker scaling
  07_bench_ddp.py        # multi-node multi-GPU DDP (torchrun)
  08_report.py           # tables + plots
run_all.sh               # end-to-end driver (synthetic by default)
run_ddp.sh               # single-node DDP strong-scaling sweep
slurm/                   # download.sbatch + ddp_multinode.sbatch (HPC templates)
results/                 # metrics CSV/JSONL + plots (gitignored)
```

---

## Distributed/parallel contributions (for the report)

- **Parallel I/O download**: `01` issues CDS requests concurrently
  (`max_concurrent_requests`), hiding the slow CDS queue; aggregate vs per-request
  throughput is recorded.
- **Chunking ⇄ access pattern**: `05` quantifies the (time,lat,lon) vs (T,1,1)
  trade-off the proposal describes, in reads/s and MB/s.
- **Embarrassingly-parallel chunk reads**: `06` shows DataLoader samples/s scaling
  with worker count — each chunk is an independent object.
- **Data-parallel training**: `07` shards the time axis with `DistributedSampler`
  across ranks/nodes and reports strong-scaling efficiency and the data-bound
  fraction (is the chunked store fast enough to keep GPUs busy?).

## Notes
- `zarr` is pinned `<3` for the stable numcodecs Blosc encoding path used here.
- The new CDS sometimes returns a `.zip` even for `unarchived`; the loader handles
  both transparently.

scp -P 2027 davidlcs@140.112.176.245:/home3/davidlcs/Econ-Rag/Local_LLM/test_meeting/parallel/Efficient_Taiwan_ERA5/data/zarr/taiwan_era5_2019_08_01_2020_12_31.zip .
