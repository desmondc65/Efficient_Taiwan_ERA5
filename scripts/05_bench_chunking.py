#!/usr/bin/env python3
"""Benchmark chunking strategies at a fixed codec -- the core proposal result.

It contrasts the two canonical weather-ML access patterns from the proposal:
  * full spatial field at a random time   (CorrDiff/StormCast, CNN/diffusion)
        -> favoured by (1, lat, lon)-style chunks
  * full time series at a random point     (LSTM / point forecasting)
        -> favoured by (time, 1, 1) chunks

For each strategy it records write time, store size, chunk/file count, and the
throughput of BOTH access patterns, so the trade-off is quantified.

Usage:
    python scripts/05_bench_chunking.py
    python scripts/05_bench_chunking.py --codec blosc_zstd5 --strategies full_field timeseries
    python scripts/05_bench_chunking.py --keep     # keep stores for 06_bench_dataloader
"""
import _bootstrap  # noqa: F401
import argparse
import shutil

import numpy as np

from era5tw.chunking import chunk_tuple, n_chunks, resolve_chunks
from era5tw.codecs import codec_label, get_compressor
from era5tw.config import load_config
from era5tw.dataset import ZarrFieldReader
from era5tw.metrics import MetricsLogger, Timer, count_files, dir_size_bytes
from era5tw.paths import bench_zarr
from era5tw.zarrio import data_sizes, dataset_nbytes, load_raw_dataset, write_zarr


def bench_access(path, n_frames, n_points, rng):
    reader = ZarrFieldReader(path)
    # Pattern A: random full spatial field (one timestep, all channels)
    t_idx = rng.integers(0, reader.n_time, size=n_frames)
    with Timer() as ta:
        for t in t_idx:
            reader.read_frame(int(t))
    frame_bytes = reader.n_channels * reader.n_lat * reader.n_lon * 4
    # Pattern B: random full time series (one point, all channels)
    ys = rng.integers(0, reader.n_lat, size=n_points)
    xs = rng.integers(0, reader.n_lon, size=n_points)
    with Timer() as tb:
        for y, x in zip(ys, xs):
            reader.read_point_series(int(y), int(x))
    series_bytes = reader.n_channels * reader.n_time * 4
    return {
        "field_ms": ta.elapsed / n_frames * 1e3,
        "field_per_s": n_frames / ta.elapsed if ta.elapsed else 0,
        "field_mb_per_s": frame_bytes * n_frames / 1e6 / ta.elapsed if ta.elapsed else 0,
        "series_ms": tb.elapsed / n_points * 1e3,
        "series_per_s": n_points / tb.elapsed if tb.elapsed else 0,
        "series_mb_per_s": series_bytes * n_points / 1e6 / tb.elapsed if tb.elapsed else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--codec", default=None)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-frames", type=int, default=128)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--keep", action="store_true",
                    help="keep stores under data/zarr/bench (reused by 06)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    codec = args.codec or cfg["zarr"]["default_codec"]
    codec_spec = cfg["zarr"]["codecs"][codec]
    comp = get_compressor(codec_spec)
    strategies = args.strategies or list(cfg["zarr"]["chunking_strategies"])
    rng = np.random.default_rng(cfg["benchmark"]["seed"])

    raw_dir = cfg.abspath(args.raw_dir) if args.raw_dir else cfg.raw_dir
    print(f"Loading raw ERA5 from {raw_dir} ...")
    ds = load_raw_dataset(raw_dir, flatten_levels=cfg["zarr"]["flatten_levels"])
    sizes = data_sizes(ds)
    nbytes = dataset_nbytes(ds)
    print(f"  {len(ds.data_vars)} channels, dims={sizes}, "
          f"uncompressed={nbytes/1e6:.1f} MB, codec={codec}\n")

    log = MetricsLogger(cfg.results_dir, "chunking")

    for name in strategies:
        strategy = cfg["zarr"]["chunking_strategies"][name]
        resolved = resolve_chunks(strategy, sizes)
        ctuple = chunk_tuple(strategy, ("time", "lat", "lon"), sizes)
        path = bench_zarr(cfg, name, codec)
        with Timer() as t_w:
            write_zarr(ds, path, strategy, comp)
        size_mb = dir_size_bytes(path) / 1e6
        acc = bench_access(path, args.n_frames, args.n_points, rng)
        log.log(
            chunking=name, chunk_shape=str(ctuple), codec=codec_label(codec_spec),
            chunks_per_var=n_chunks(resolved, sizes),
            n_files=count_files(path),
            zarr_mb=round(size_mb, 2),
            compression_ratio=round(nbytes / 1e6 / size_mb, 3) if size_mb else 0,
            write_s=round(t_w.elapsed, 3),
            field_read_ms=round(acc["field_ms"], 3),
            field_per_s=round(acc["field_per_s"], 1),
            field_mb_per_s=round(acc["field_mb_per_s"], 1),
            series_read_ms=round(acc["series_ms"], 3),
            series_per_s=round(acc["series_per_s"], 1),
            series_mb_per_s=round(acc["series_mb_per_s"], 1),
        )
        if not args.keep:
            shutil.rmtree(path, ignore_errors=True)

    log.flush_csv()
    print(f"\nMetrics -> {log.csv_path}")
    if args.keep:
        print(f"Stores kept under {cfg.zarr_dir/'bench'} (reused by 06_bench_dataloader).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
