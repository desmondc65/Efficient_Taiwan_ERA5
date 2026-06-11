#!/usr/bin/env python3
"""Bridge the StormCast LowRes ERA5 zarr into the era5tw benchmark pipeline.

The StormCast store keeps one 4-D array `LowRes` (time, channel, y, x) with
2-D curvilinear latitude/longitude. The era5tw scripts expect per-channel 3-D
(time, lat, lon) variables read from raw NetCDF. This script:

  1. reads the FULL train+valid horizon metadata and reports the true on-disk /
     uncompressed size, plus the extrapolated size of an equivalent GLOBAL
     dataset over the same time horizon (at this grid's ~2 km resolution and at
     standard ERA5 0.25 deg) -> results/dataset_size_report.{json,txt};
  2. extracts a representative time subset (default 1 month = 744 hourly steps,
     full 224x128 grid, all 24 channels) into <raw_dir>/era5_taiwan_lowres.nc so
     the existing 03/04/05/06/07 scripts run unchanged.

Usage:
    python scripts/90_ingest_stormcast.py --src <LowRes dir> --steps 744
"""
import _bootstrap  # noqa: F401
import argparse
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr

from era5tw.config import load_config
from era5tw.metrics import Timer, dir_size_bytes

EARTH_AREA_KM2 = 5.100656e8        # mean Earth surface area
ERA5_GLOBAL_025 = (721, 1440)      # global 0.25 deg grid (lat, lon)


def haversine_km(la1, lo1, la2, lo2):
    R = 6371.0088
    p = math.pi / 180.0
    a = (math.sin((la2 - la1) * p / 2) ** 2
         + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def grid_cell_km2(lat2d, lon2d):
    """Mean grid-cell area (km^2) from the 2-D curvilinear lat/lon."""
    ny, nx = lat2d.shape
    dys, dxs = [], []
    for j in range(0, nx, max(1, nx // 16)):
        col_lat, col_lon = lat2d[:, j], lon2d[:, j]
        dys += [haversine_km(col_lat[i], col_lon[i], col_lat[i + 1], col_lon[i + 1])
                for i in range(0, ny - 1, max(1, ny // 16))]
    for i in range(0, ny, max(1, ny // 16)):
        row_lat, row_lon = lat2d[i, :], lon2d[i, :]
        dxs += [haversine_km(row_lat[j], row_lon[j], row_lat[j + 1], row_lon[j + 1])
                for j in range(0, nx - 1, max(1, nx // 16))]
    return float(np.mean(dys)), float(np.mean(dxs))


def size_report(src: Path, results_dir: Path):
    train = xr.open_zarr(f"{src}/stormcast_test_train.zarr", consolidated=True)
    valid = xr.open_zarr(f"{src}/stormcast_test_valid.zarr", consolidated=True)
    nt_tr, nt_va = train.sizes["time"], valid.sizes["time"]
    nC, ny, nx = (train.sizes["channel"], train.sizes["y"], train.sizes["x"])
    total_steps = nt_tr + nt_va
    bytes_per_step = nC * ny * nx * 4
    uncompressed = total_steps * bytes_per_step
    on_disk = dir_size_bytes(src)

    lat2d = np.asarray(train["latitude"].values)
    lon2d = np.asarray(train["longitude"].values)
    dy_km, dx_km = grid_cell_km2(lat2d, lon2d)
    cell_km2 = dy_km * dx_km
    region_pts = ny * nx
    lat_span = float(lat2d.max() - lat2d.min())
    lon_span = float(lon2d.max() - lon2d.min())

    # GLOBAL extrapolation over the SAME horizon + SAME channels.
    # (a) equal-area tiling of the globe at this cell size
    global_pts_ea = EARTH_AREA_KM2 / cell_km2
    factor_ea = global_pts_ea / region_pts
    # (b) angular grid product at this resolution (upper bound; meridian convergence ignored)
    pts_deg_lat, pts_deg_lon = ny / lat_span, nx / lon_span
    global_pts_grid = (180 * pts_deg_lat) * (360 * pts_deg_lon)
    factor_grid = global_pts_grid / region_pts
    # (c) standard ERA5 0.25 deg global product
    global_pts_025 = ERA5_GLOBAL_025[0] * ERA5_GLOBAL_025[1]
    bytes_025 = global_pts_025 * nC * total_steps * 4

    rep = {
        "source": str(src),
        "horizon": {
            "train": [str(train.time.values[0]), str(train.time.values[-1]), nt_tr],
            "valid": [str(valid.time.values[0]), str(valid.time.values[-1]), nt_va],
            "total_hourly_steps": total_steps,
            "approx_years": round(total_steps / 24 / 365.25, 2),
        },
        "grid": {"channels": nC, "y": ny, "x": nx, "region_points": region_pts,
                 "lat_span_deg": round(lat_span, 3), "lon_span_deg": round(lon_span, 3),
                 "cell_km": [round(dy_km, 3), round(dx_km, 3)],
                 "cell_km2": round(cell_km2, 3)},
        "regional_dataset": {
            "uncompressed_bytes": uncompressed,
            "uncompressed_GB": round(uncompressed / 1e9, 2),
            "on_disk_bytes": on_disk,
            "on_disk_GB": round(on_disk / 1e9, 2),
        },
        "global_same_horizon": {
            "equal_area_2km": {"points": int(global_pts_ea), "factor_vs_region": round(factor_ea, 1),
                               "uncompressed_TB": round(uncompressed * factor_ea / 1e12, 2)},
            "grid_product_2km": {"points": int(global_pts_grid), "factor_vs_region": round(factor_grid, 1),
                                 "uncompressed_TB": round(uncompressed * factor_grid / 1e12, 2)},
            "era5_0p25deg": {"points": global_pts_025,
                             "uncompressed_TB": round(bytes_025 / 1e12, 2)},
        },
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "dataset_size_report.json").write_text(json.dumps(rep, indent=2))

    lines = [
        "=================== ORIGINAL DATASET SIZE (full 3.42-yr horizon) ===================",
        f"  source            : {src}",
        f"  horizon           : {rep['horizon']['train'][0][:10]} -> {rep['horizon']['valid'][1][:10]}"
        f"  ({total_steps} hourly steps, ~{rep['horizon']['approx_years']} yr)",
        f"  grid              : {nC} channels x {ny} x {nx}  (~{dy_km:.2f}x{dx_km:.2f} km cells, Taiwan box)",
        f"  uncompressed      : {uncompressed/1e9:8.2f} GB",
        f"  on-disk (du)      : {on_disk/1e9:8.2f} GB  (stored uncompressed, compressor=null)",
        "",
        "=================== EQUIVALENT GLOBAL DATASET (same horizon + channels) ============",
        f"  (a) globe @ ~2 km  equal-area : {global_pts_ea/1e6:8.1f} M pts  "
        f"= {factor_ea:6.0f}x region  -> {uncompressed*factor_ea/1e12:8.1f} TB uncompressed",
        f"  (b) globe @ ~2 km  grid-prod  : {global_pts_grid/1e6:8.1f} M pts  "
        f"= {factor_grid:6.0f}x region  -> {uncompressed*factor_grid/1e12:8.1f} TB  (upper bound)",
        f"  (c) globe @ ERA5 0.25 deg     : {global_pts_025/1e6:8.3f} M pts            "
        f"          -> {bytes_025/1e12:8.2f} TB uncompressed",
        "====================================================================================",
    ]
    txt = "\n".join(lines)
    (results_dir / "dataset_size_report.txt").write_text(txt + "\n")
    print(txt)
    train.close(); valid.close()
    return rep


def extract_subset(src: Path, raw_dir: Path, steps: int, split: str, start: int):
    p = f"{src}/stormcast_test_{split}.zarr"
    ds = xr.open_zarr(p, consolidated=True)
    n = ds.sizes["time"]
    end = min(start + steps, n)
    sub = ds["LowRes"].isel(time=slice(start, end))      # (time, channel, y, x)
    chan = [str(c) for c in ds["channel"].values]
    print(f"\nExtracting {split}[{start}:{end}] -> {len(chan)} channels x "
          f"{sub.sizes['y']}x{sub.sizes['x']}, {end-start} steps")
    with Timer() as t:
        arr = sub.transpose("time", "channel", "y", "x").values   # load subset into RAM
    print(f"  read {arr.nbytes/1e9:.2f} GB in {t.elapsed:.1f}s")

    out = xr.Dataset()
    for i, name in enumerate(chan):
        out[name] = (("time", "lat", "lon"), arr[:, i].astype("float32"))
    out = out.assign_coords(
        time=ds["time"].isel(time=slice(start, end)).values,
        lat=np.arange(sub.sizes["y"], dtype="int32"),
        lon=np.arange(sub.sizes["x"], dtype="int32"),
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "era5_taiwan_lowres.nc"
    enc = {v: {"zlib": False} for v in out.data_vars}     # raw = uncompressed netcdf
    with Timer() as tw:
        out.to_netcdf(target, encoding=enc, engine="netcdf4")
    print(f"  wrote {target}  ({target.stat().st_size/1e9:.2f} GB) in {tw.elapsed:.1f}s")
    ds.close()
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="LowRes dir with stormcast_test_{train,valid}.zarr")
    ap.add_argument("--steps", type=int, default=744, help="time steps to extract for benchmarks")
    ap.add_argument("--split", default="train", choices=["train", "valid"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-extract", action="store_true", help="only print the size report")
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = Path(args.src)
    size_report(src, cfg.results_dir)
    if not args.no_extract:
        extract_subset(src, cfg.raw_dir, args.steps, args.split, args.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
