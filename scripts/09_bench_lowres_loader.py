#!/usr/bin/env python3
"""Benchmark PyTorch DataLoader throughput on the RELEASED LowRes ERA5 dataset
in its native layout, and contrast access patterns.

The open dataset is one 4-D array ``LowRes`` (time, channel, y, x) chunked
(1326, 3, 28, 16) -- a deep time axis. This script wraps it in a fork-safe
PyTorch Dataset (re-opening the Zarr per worker, so reads parallelize) and
measures next-frame loading throughput, plus the chunk read-amplification, for
two access patterns:

  * ``random``     -- random single timesteps (the StormCast/CorrDiff training
                      pattern). Worst case for a deep-time chunking: one frame
                      touches every spatial/channel chunk of its time block.
  * ``sequential`` -- contiguous timesteps (streaming / page-cache friendly).

Compare the numbers here against scripts/06 on the rechunked ``full_field``
store to see what a one-time rechunk buys.

Usage:
    python scripts/09_bench_lowres_loader.py --src <LowRes> --access random \
        --workers 0 4 --n-frames 8 --batch-size 2
"""
import _bootstrap  # noqa: F401
import argparse
import math

import numpy as np

from era5tw.config import load_config
from era5tw.metrics import MetricsLogger, Timer


def _open_lowres(path: str):
    import zarr
    try:
        g = zarr.open_consolidated(path, mode="r")
    except Exception:
        g = zarr.open_group(path, mode="r")
    return g["LowRes"]


def amplification(arr) -> tuple[int, float, float]:
    """(#chunks touched, GB read, amp factor) for a single-frame [t,:,:,:] read."""
    shape, chunks = arr.shape, arr.chunks
    touched = (1
               * math.ceil(shape[1] / chunks[1])
               * math.ceil(shape[2] / chunks[2])
               * math.ceil(shape[3] / chunks[3]))
    chunk_bytes = int(np.prod(chunks)) * 4
    frame_bytes = shape[1] * shape[2] * shape[3] * 4
    return touched, touched * chunk_bytes / 1e9, touched * chunk_bytes / frame_bytes


def make_dataset(path: str, indices, window: int):
    import torch
    import torch.utils.data as tud

    class _DS(tud.Dataset):
        def __init__(self):
            self.path = path
            self.indices = list(indices)
            self.window = window
            self._a = None

        def _arr(self):
            if self._a is None:
                self._a = _open_lowres(self.path)
            return self._a

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            a = self._arr()
            t = self.indices[i]
            x = np.asarray(a[t], dtype=np.float32)
            y = np.asarray(a[t + self.window - 1], dtype=np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)

    return _DS()


def run(path, indices, frame_bytes, amp, batch_size, num_workers, window):
    import torch
    from torch.utils.data import DataLoader
    ds = make_dataset(path, indices, window)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False,
                        persistent_workers=False,
                        prefetch_factor=(2 if num_workers > 0 else None))
    seen = 0
    with Timer() as t:
        for x, y in loader:
            seen += x.shape[0]
    sps = seen / t.elapsed if t.elapsed else 0
    eff = frame_bytes * seen / 1e6 / t.elapsed if t.elapsed else 0
    return {"samples": seen, "seconds": t.elapsed, "samples_per_s": sps,
            "eff_mb_per_s": eff, "phys_mb_per_s": eff * amp}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="LowRes dir with stormcast_test_{train,valid}.zarr")
    ap.add_argument("--split", default="train", choices=["train", "valid"])
    ap.add_argument("--access", default="random", choices=["random", "sequential"])
    ap.add_argument("--workers", nargs="+", type=int, default=[0, 4])
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--start", type=int, default=0, help="start index for sequential access")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = f"{args.src}/stormcast_test_{args.split}.zarr"
    arr = _open_lowres(path)
    n_time = arr.shape[0]
    frame_bytes = arr.shape[1] * arr.shape[2] * arr.shape[3] * 4
    touched, gb, amp = amplification(arr)
    print(f"LowRes {arr.shape} chunks {arr.chunks}  frame={frame_bytes/1e6:.2f} MB")
    print(f"single-frame read touches {touched} chunks = {gb:.2f} GB  (amp ~{amp:.0f}x)\n")

    rng = np.random.default_rng(cfg["benchmark"]["seed"])
    hi = n_time - (args.window - 1)
    if args.access == "random":
        idx = rng.integers(0, hi, size=args.n_frames)
    else:
        idx = np.arange(args.start, min(args.start + args.n_frames, hi))

    log = MetricsLogger(cfg.results_dir, "lowres_loader")
    for nw in args.workers:
        r = run(path, idx, frame_bytes, amp, args.batch_size, nw, args.window)
        log.log(store="native_lowres", access=args.access, split=args.split,
                num_workers=nw, batch_size=args.batch_size, n_frames=int(len(idx)),
                window=args.window, chunk_shape=str(tuple(arr.chunks)),
                amp_factor=round(amp, 1),
                samples=r["samples"], seconds=round(r["seconds"], 2),
                samples_per_s=round(r["samples_per_s"], 3),
                eff_mb_per_s=round(r["eff_mb_per_s"], 1),
                phys_mb_per_s=round(r["phys_mb_per_s"], 1))
    log.flush_csv()
    print(f"\nMetrics -> {log.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
