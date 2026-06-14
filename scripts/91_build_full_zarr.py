#!/usr/bin/env python3
"""Convert the FULL 3.42-year StormCast LowRes ERA5 store into a single
analysis-ready, training-optimized Zarr (the artifact we openly release).

Optimal settings (from our benchmarks, scripts/04--06):
  * chunking = full_field (time=1, full spatial)  -> fastest random-frame reads
    for StormCast/CorrDiff training (453 samples/s at 8 workers).
  * codec    = blosc-zstd level 5 + shuffle       -> lossless, good read speed.

It streams the conversion one source time-chunk at a time (a few GB in RAM), so
it never materializes the whole 82.5 GB set, and splits the 24-channel LowRes
array into named (time, y, x) channels (mslp, t2m, z500, ...). The curvilinear
latitude/longitude(y, x) and real datetimes are carried as coordinates.

Usage:
    python scripts/91_build_full_zarr.py --src <LowRes> [--out <dir>] [--max-blocks N]
"""
import _bootstrap  # noqa: F401
import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from numcodecs import Blosc

from era5tw.config import load_config
from era5tw.metrics import dir_size_bytes


def _block_size(a) -> int:
    """Source time-chunk length, so each read aligns to on-disk chunks."""
    try:
        return int(a.chunks[0][0])          # dask chunk along time
    except Exception:
        return 1024


def convert_split(ds, names, lat2d, lon2d, out, compressor, first, max_blocks, done):
    a = ds["LowRes"]                         # (time, channel, y, x)
    T = a.sizes["time"]
    block = _block_size(a)
    nblk = 0
    for t0 in range(0, T, block):
        if max_blocks and done[0] >= max_blocks:
            return
        t1 = min(t0 + block, T)
        t_read = time.time()
        sub = a.isel(time=slice(t0, t1)).transpose("time", "channel", "y", "x").values
        out_ds = xr.Dataset(
            {nm: (("time", "y", "x"), sub[:, i].astype("float32"))
             for i, nm in enumerate(names)},
            coords={
                "time": ds["time"].isel(time=slice(t0, t1)).values,
                "latitude": (("y", "x"), lat2d),
                "longitude": (("y", "x"), lon2d),
            },
        )
        for v in out_ds.data_vars:
            out_ds[v].attrs["long_name"] = v
        out_ds = out_ds.chunk({"time": 1, "y": -1, "x": -1})
        if first[0]:
            out_ds.attrs.update(
                title="Taiwan ERA5 (StormCast LowRes), analysis-ready Zarr",
                grid="224x128 curvilinear ~2km", chunking="full_field (1,224,128)",
                codec="blosc-zstd5-shuffle (lossless)")
            enc = {v: {"compressor": compressor} for v in out_ds.data_vars}
            out_ds.to_zarr(out, mode="w", encoding=enc, consolidated=False)
            first[0] = False
        else:
            out_ds.to_zarr(out, mode="a", append_dim="time", consolidated=False)
        done[0] += 1
        nblk += 1
        print(f"  block {done[0]:>2} t[{t0}:{t1}] of split-T={T}  "
              f"read+write {time.time()-t_read:5.1f}s  store={dir_size_bytes(out)/1e9:6.2f} GB",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-blocks", type=int, default=0, help="0 = all (for testing)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out) if args.out else cfg.zarr_dir / "taiwan_era5_full.zarr"
    out.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    train = xr.open_zarr(f"{args.src}/stormcast_test_train.zarr", consolidated=True)
    valid = xr.open_zarr(f"{args.src}/stormcast_test_valid.zarr", consolidated=True)
    names = [str(c) for c in train["channel"].values]
    lat2d = np.asarray(train["latitude"].values)
    lon2d = np.asarray(train["longitude"].values)
    print(f"Source: {len(names)} channels, grid {lat2d.shape}, "
          f"train T={train.sizes['time']} + valid T={valid.sizes['time']}")
    print(f"Writing -> {out}  (full_field chunks, blosc_zstd5)\n")

    first, done = [True], [0]
    t0 = time.time()
    convert_split(train, names, lat2d, lon2d, out, compressor, first, args.max_blocks, done)
    convert_split(valid, names, lat2d, lon2d, out, compressor, first, args.max_blocks, done)
    print("\nConsolidating metadata ...")
    zarr.consolidate_metadata(str(out))

    final = xr.open_zarr(out, consolidated=True)
    size = dir_size_bytes(out)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    print(f"  store     : {out}")
    print(f"  size      : {size/1e9:.2f} GB ({size/1e6:.0f} MB)")
    print(f"  time steps: {final.sizes['time']}  ({final.time.values[0]} .. {final.time.values[-1]})")
    print(f"  variables : {len(final.data_vars)}  dims={dict(final.sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
