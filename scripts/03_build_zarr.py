#!/usr/bin/env python3
"""Build the distributable Zarr from raw ERA5, with the configured default
chunking + codec. This is the 'production' artifact you ship for download.

Usage:
    python scripts/03_build_zarr.py                       # defaults from config
    python scripts/03_build_zarr.py --chunking daily --codec blosc_zstd9
    python scripts/03_build_zarr.py --raw-glob data/raw   # explicit raw source

Records: raw NetCDF size, uncompressed in-memory size, Zarr size, compression
ratio, storage-vs-global extrapolation, and write time/throughput.
"""
import _bootstrap  # noqa: F401
import argparse

from era5tw.codecs import codec_label, get_compressor
from era5tw.config import load_config
from era5tw.metrics import (MetricsLogger, Timer, count_files, dir_size_bytes)
from era5tw.paths import production_zarr
from era5tw.zarrio import (data_sizes, dataset_nbytes, load_raw_dataset, write_zarr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunking", default=None, help="chunking strategy name")
    ap.add_argument("--codec", default=None, help="codec name")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    chunking = args.chunking or cfg["zarr"]["default_chunking"]
    codec = args.codec or cfg["zarr"]["default_codec"]
    strategy = cfg["zarr"]["chunking_strategies"][chunking]
    codec_spec = cfg["zarr"]["codecs"][codec]
    compressor = get_compressor(codec_spec)

    raw_dir = cfg.abspath(args.raw_dir) if args.raw_dir else cfg.raw_dir
    out = cfg.abspath(args.out) if args.out else production_zarr(cfg)

    print(f"Loading raw ERA5 from {raw_dir} ...")
    with Timer() as t_load:
        ds = load_raw_dataset(raw_dir, flatten_levels=cfg["zarr"]["flatten_levels"])
    sizes = data_sizes(ds)
    nbytes = dataset_nbytes(ds)
    raw_mb = dir_size_bytes(raw_dir) / 1e6
    print(f"  loaded {len(ds.data_vars)} channels, dims={sizes}, "
          f"uncompressed={nbytes/1e6:.1f} MB, load={t_load.elapsed:.1f}s")

    print(f"Writing Zarr -> {out}  (chunking={chunking}, codec={codec})")
    with Timer() as t_write:
        meta = write_zarr(ds, out, strategy, compressor)
    zarr_mb = dir_size_bytes(out) / 1e6
    ratio = nbytes / 1e6 / zarr_mb if zarr_mb else float("nan")

    log = MetricsLogger(cfg.results_dir, "build_zarr")
    log.log(
        chunking=chunking, codec=codec_label(codec_spec),
        chunk_shape=str(tuple(meta["chunks"].values())),
        channels=len(ds.data_vars), dims=str(sizes),
        raw_netcdf_mb=round(raw_mb, 2),
        uncompressed_mb=round(nbytes / 1e6, 2),
        zarr_mb=round(zarr_mb, 2),
        compression_ratio=round(ratio, 3),
        n_chunks_files=count_files(out),
        load_seconds=round(t_load.elapsed, 2),
        write_seconds=round(t_write.elapsed, 2),
        write_mb_per_s=round(nbytes / 1e6 / t_write.elapsed, 1) if t_write.elapsed else 0,
        out=str(out),
    )
    log.flush_csv()

    # Storage story: how much smaller is the regional+compressed product?
    print("\n=== storage footprint ===")
    print(f"  raw regional NetCDF : {raw_mb:8.1f} MB")
    print(f"  uncompressed (f32)  : {nbytes/1e6:8.1f} MB")
    print(f"  zarr (compressed)   : {zarr_mb:8.1f} MB   "
          f"({ratio:.2f}x vs uncompressed)")
    # Global-vs-regional ratio for context (global 0.25deg grid = 721 x 1440).
    region_pts = sizes.get("lat", 1) * sizes.get("lon", 1)
    global_pts = 721 * 1440
    print(f"  regional crop keeps {region_pts}/{global_pts} grid points "
          f"= 1/{global_pts/region_pts:.0f} of the globe")
    print(f"\nWrote {out}\nMetrics -> {log.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
