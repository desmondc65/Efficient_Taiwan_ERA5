#!/usr/bin/env python3
"""Benchmark a real PyTorch DataLoader over Zarr: how chunking x num_workers
drives training-time data throughput. This is the end-to-end "does it actually
feed the GPU fast enough" metric.

Each chunk is an independent object, so DataLoader workers read in parallel --
this script quantifies that scaling for each chunking strategy.

Reuses stores written by 05_bench_chunking.py --keep when present; otherwise
builds them with the default codec.

Usage:
    python scripts/06_bench_dataloader.py
    python scripts/06_bench_dataloader.py --strategies full_field daily timeseries
    python scripts/06_bench_dataloader.py --workers 0 1 2 4 8 16
"""
import _bootstrap  # noqa: F401
import argparse

from era5tw.codecs import get_compressor
from era5tw.config import load_config
from era5tw.metrics import MetricsLogger, Timer
from era5tw.paths import bench_zarr


def ensure_store(cfg, name, codec):
    path = bench_zarr(cfg, name, codec)
    if path.exists():
        return path
    from era5tw.zarrio import load_raw_dataset, write_zarr
    print(f"  building missing store {path} ...")
    ds = load_raw_dataset(cfg.raw_dir, flatten_levels=cfg["zarr"]["flatten_levels"])
    strategy = cfg["zarr"]["chunking_strategies"][name]
    write_zarr(ds, path, strategy, get_compressor(cfg["zarr"]["codecs"][codec]))
    return path


def run_loader(path, batch_size, num_batches, num_workers, window, pin):
    import torch
    from torch.utils.data import DataLoader

    from era5tw.dataset import make_torch_dataset
    ds = make_torch_dataset(path, window=window)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    it = iter(loader)
    # warmup (spawn workers, fill prefetch)
    x, y = next(it)
    n_channels = x.shape[1]
    bytes_per_batch = x.numel() * x.element_size() + y.numel() * y.element_size()

    seen = 0
    with Timer() as t:
        for _ in range(num_batches):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            seen += x.shape[0]
    del loader, it
    return {
        "samples": seen,
        "seconds": t.elapsed,
        "samples_per_s": seen / t.elapsed if t.elapsed else 0,
        "mb_per_s": bytes_per_batch * num_batches / 1e6 / t.elapsed if t.elapsed else 0,
        "n_channels": n_channels,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--codec", default=None)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--workers", nargs="+", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-batches", type=int, default=None)
    ap.add_argument("--pin-memory", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dl = cfg["benchmark"]["dataloader"]
    codec = args.codec or cfg["zarr"]["default_codec"]
    # spatial strategies are the realistic ones for full-field training
    strategies = args.strategies or ["full_field", "daily", "weekly", "timeseries"]
    workers = args.workers or dl["num_workers_grid"]
    bs = args.batch_size or dl["batch_size"]
    nb = args.num_batches or dl["num_batches"]
    window = dl["window"]

    log = MetricsLogger(cfg.results_dir, "dataloader")
    print(f"DataLoader bench: strategies={strategies}, workers={workers}, "
          f"batch_size={bs}, num_batches={nb}\n")

    for name in strategies:
        path = ensure_store(cfg, name, codec)
        for nw in workers:
            r = run_loader(path, bs, nb, nw, window, args.pin_memory)
            log.log(
                chunking=name, codec=codec, num_workers=nw,
                batch_size=bs, num_batches=nb, channels=r["n_channels"],
                samples=r["samples"], seconds=round(r["seconds"], 3),
                samples_per_s=round(r["samples_per_s"], 1),
                mb_per_s=round(r["mb_per_s"], 1),
            )

    log.flush_csv()
    print(f"\nMetrics -> {log.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
