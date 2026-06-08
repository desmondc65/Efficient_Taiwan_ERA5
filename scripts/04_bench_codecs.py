#!/usr/bin/env python3
"""Benchmark compression codecs at a fixed chunking strategy.

For each codec in config.zarr.codecs it records:
  * write time + write throughput
  * on-disk size + compression ratio (vs uncompressed float32)
  * sequential full-read (decompress-everything) time + read throughput
  * random single-frame read latency (the data-loading hot path)

Usage:
    python scripts/04_bench_codecs.py
    python scripts/04_bench_codecs.py --chunking daily --codecs none blosc_zstd5 gzip5
    python scripts/04_bench_codecs.py --keep        # keep the written stores
"""
import _bootstrap  # noqa: F401
import argparse
import shutil

import numpy as np

from era5tw.codecs import codec_label, get_compressor
from era5tw.config import load_config
from era5tw.dataset import ZarrFieldReader
from era5tw.metrics import MetricsLogger, Timer, dir_size_bytes
from era5tw.zarrio import dataset_nbytes, load_raw_dataset


def bench_reads(path, n_frames, rng):
    reader = ZarrFieldReader(path)
    # sequential full read (force decompression of the entire store)
    with Timer() as t_full:
        total = 0
        for v in reader.variables:
            total += int(reader._store()[v][:].sum() * 0)  # touch all bytes
    # random single-frame reads
    idx = rng.integers(0, reader.n_time, size=n_frames)
    with Timer() as t_rand:
        for t in idx:
            reader.read_frame(int(t))
    nbytes_frame = reader.n_channels * reader.n_lat * reader.n_lon * 4
    return {
        "full_read_s": t_full.elapsed,
        "full_read_mb_per_s": (reader.n_channels * reader.n_time *
                               reader.n_lat * reader.n_lon * 4 / 1e6 / t_full.elapsed)
        if t_full.elapsed else 0,
        "rand_frame_ms": t_rand.elapsed / n_frames * 1e3,
        "rand_frame_mb_per_s": nbytes_frame * n_frames / 1e6 / t_rand.elapsed
        if t_rand.elapsed else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunking", default=None)
    ap.add_argument("--codecs", nargs="+", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-frames", type=int, default=64)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from era5tw.zarrio import write_zarr
    chunking = args.chunking or cfg["zarr"]["default_chunking"]
    strategy = cfg["zarr"]["chunking_strategies"][chunking]
    codecs = args.codecs or list(cfg["zarr"]["codecs"])
    rng = np.random.default_rng(cfg["benchmark"]["seed"])

    raw_dir = cfg.abspath(args.raw_dir) if args.raw_dir else cfg.raw_dir
    print(f"Loading raw ERA5 from {raw_dir} ...")
    ds = load_raw_dataset(raw_dir, flatten_levels=cfg["zarr"]["flatten_levels"])
    nbytes = dataset_nbytes(ds)
    print(f"  {len(ds.data_vars)} channels, uncompressed={nbytes/1e6:.1f} MB, "
          f"chunking={chunking}\n")

    bench_dir = cfg.zarr_dir / "bench_codecs"
    log = MetricsLogger(cfg.results_dir, "codecs")

    for name in codecs:
        spec = cfg["zarr"]["codecs"][name]
        comp = get_compressor(spec)
        path = bench_dir / f"{name}.zarr"
        with Timer() as t_w:
            write_zarr(ds, path, strategy, comp)
        size_mb = dir_size_bytes(path) / 1e6
        reads = bench_reads(path, args.n_frames, rng)
        log.log(
            codec=name, codec_detail=codec_label(spec), chunking=chunking,
            uncompressed_mb=round(nbytes / 1e6, 2),
            zarr_mb=round(size_mb, 2),
            compression_ratio=round(nbytes / 1e6 / size_mb, 3) if size_mb else 0,
            write_s=round(t_w.elapsed, 3),
            write_mb_per_s=round(nbytes / 1e6 / t_w.elapsed, 1) if t_w.elapsed else 0,
            full_read_s=round(reads["full_read_s"], 3),
            full_read_mb_per_s=round(reads["full_read_mb_per_s"], 1),
            rand_frame_ms=round(reads["rand_frame_ms"], 3),
            rand_frame_mb_per_s=round(reads["rand_frame_mb_per_s"], 1),
        )
        if not args.keep:
            shutil.rmtree(path, ignore_errors=True)

    log.flush_csv()
    if not args.keep:
        shutil.rmtree(bench_dir, ignore_errors=True)
    print(f"\nMetrics -> {log.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
