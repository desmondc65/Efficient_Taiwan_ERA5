#!/usr/bin/env python3
"""Generate a synthetic ERA5-like NetCDF so the WHOLE pipeline runs without a
CDS key. Same grid, dims, variable set, and naming quirks (valid_time/latitude/
longitude/pressure_level) as the real CDS output, so the loader is exercised too.

Fields are smooth (advected sinusoids + mild noise) so codec compression ratios
are realistic -- not trivially compressible like constants nor incompressible
like white noise.

Usage:
    python scripts/02_make_synthetic.py                 # ~1 month (744 steps)
    python scripts/02_make_synthetic.py --steps 240     # fewer timesteps (faster)
    python scripts/02_make_synthetic.py --out data/raw/era5_synthetic.nc
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from era5tw.config import load_config

# CDS long name -> short name used in NetCDF output.
SURFACE_SHORT = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "mean_sea_level_pressure": "msl",
    "total_precipitation": "tp",
}
PL_SHORT = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
}


def _smooth_field(rng, nt, nlat, nlon, base, amp, seed_phase):
    """A drifting multi-scale field: (nt, nlat, nlon) float32."""
    lat = np.linspace(0, np.pi, nlat)[None, :, None]
    lon = np.linspace(0, 2 * np.pi, nlon)[None, None, :]
    t = np.linspace(0, 4 * np.pi, nt)[:, None, None]
    field = base + amp * (
        np.sin(2 * lat + 0.3 * t + seed_phase)
        + np.cos(3 * lon - 0.2 * t)
        + 0.5 * np.sin(lat + lon + 0.1 * t)
    )
    field = field + amp * 0.05 * rng.standard_normal((nt, nlat, nlon))
    return field.astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=744, help="number of hourly timesteps")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n, w, s, e = cfg.area
    res = cfg["region"]["resolution"]
    lats = np.arange(n, s - res / 2, -res, dtype="float32")   # descending (ERA5)
    lons = np.arange(w, e + res / 2, res, dtype="float32")
    nlat, nlon = len(lats), len(lons)
    nt = args.steps
    rng = np.random.default_rng(args.seed)

    times = (np.datetime64("2020-01-01T00") + np.arange(nt)).astype("datetime64[ns]")

    print(f"Synthetic grid: time={nt}, lat={nlat}, lon={nlon}")

    data_vars = {}
    # surface (instantaneous-style) fields
    surf_params = {"u10": (-2, 8), "v10": (1, 8), "t2m": (295, 12),
                   "msl": (101300, 800), "tp": (0.0005, 0.0008)}
    for i, long in enumerate(cfg["variables"]["single_levels"]):
        short = SURFACE_SHORT.get(long, long)
        base, amp = surf_params.get(short, (0.0, 1.0))
        f = _smooth_field(rng, nt, nlat, nlon, base, amp, seed_phase=0.7 * i)
        if short == "tp":
            f = np.clip(f, 0, None)
        data_vars[short] = (("valid_time", "latitude", "longitude"), f)

    # pressure-level fields (with a pressure_level dim, as real CDS files have)
    levels = [int(x) for x in cfg["variables"]["pressure_levels"]["levels"]]
    nlev = len(levels)
    pl_base = {"z": 5e4, "t": 250, "u": 5, "v": 0, "q": 0.005}
    pl_amp = {"z": 3e3, "t": 25, "u": 12, "v": 12, "q": 0.004}
    for j, long in enumerate(cfg["variables"]["pressure_levels"]["variables"]):
        short = PL_SHORT.get(long, long)
        arr = np.empty((nt, nlev, nlat, nlon), dtype="float32")
        for k, lev in enumerate(levels):
            scale = lev / 1000.0
            base = pl_base.get(short, 0.0) * scale
            amp = pl_amp.get(short, 1.0)
            arr[:, k] = _smooth_field(rng, nt, nlat, nlon, base, amp,
                                      seed_phase=0.5 * j + 0.11 * k)
        data_vars[short] = (("valid_time", "pressure_level", "latitude", "longitude"), arr)

    ds = xr.Dataset(
        data_vars,
        coords=dict(
            valid_time=("valid_time", times),
            pressure_level=("pressure_level", np.array(levels, dtype="int32")),
            latitude=("latitude", lats),
            longitude=("longitude", lons),
        ),
        attrs={"title": "Synthetic ERA5-like data (Team 46 pipeline test)"},
    )

    out = Path(args.out) if args.out else (cfg.raw_dir / "era5_synthetic.nc")
    out.parent.mkdir(parents=True, exist_ok=True)
    comp = dict(zlib=True, complevel=1)
    ds.to_netcdf(out, encoding={v: comp for v in ds.data_vars})
    mb = out.stat().st_size / 1e6
    n_pl_vars = len(cfg["variables"]["pressure_levels"]["variables"])
    n_channels = len(cfg["variables"]["single_levels"]) + n_pl_vars * nlev
    print(f"Wrote {out}  ({mb:.1f} MB, {len(ds.data_vars)} base vars, "
          f"{n_channels} flattened channels expected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
