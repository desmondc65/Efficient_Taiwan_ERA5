"""Load raw ERA5 (NetCDF/zip) and write chunked+compressed Zarr.

Handles the awkward parts of the *new* CDS NetCDF output:
  * time coordinate may be called `valid_time`;
  * singleton `number` / `expver` dims and coords;
  * latitude/longitude vs lat/lon naming;
  * accumulated + instantaneous streams arriving as separate files;
  * pressure-level variables flattened to 3-D channels (z500, t850, ...).
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .chunking import resolve_chunks

# CDS long name -> NetCDF short name (used only for nicer flattened channel names).
LEVEL_DIM_CANDIDATES = ("pressure_level", "level", "plev", "isobaricInhPa")
DATA_DIMS = ("time", "lat", "lon")


def _normalize(ds: xr.Dataset) -> xr.Dataset:
    """Standardize coord/dim names and drop ensemble/expver singletons."""
    rename = {}
    if "valid_time" in ds.variables and "time" not in ds.dims:
        rename["valid_time"] = "time"
    if "latitude" in ds.variables:
        rename["latitude"] = "lat"
    if "longitude" in ds.variables:
        rename["longitude"] = "lon"
    ds = ds.rename(rename)

    # Drop singleton ensemble / version dims and stray coords.
    for d in ("number", "expver", "realization"):
        if d in ds.dims and ds.sizes.get(d, 1) == 1:
            ds = ds.isel({d: 0}, drop=True)
    for c in ("number", "expver", "realization"):
        if c in ds.coords:
            ds = ds.drop_vars(c, errors="ignore")
    return ds


def _collect_nc_files(raw_dir: Path, workdir: Path) -> list[Path]:
    """Return a list of .nc files, extracting any .zip archives into `workdir`."""
    raw_dir = Path(raw_dir)
    nc_files: list[Path] = sorted(raw_dir.rglob("*.nc"))
    for zf in sorted(raw_dir.rglob("*.zip")):
        with zipfile.ZipFile(zf) as z:
            dest = workdir / zf.stem
            dest.mkdir(parents=True, exist_ok=True)
            z.extractall(dest)
            nc_files.extend(sorted(dest.rglob("*.nc")))
    return nc_files


def load_raw_dataset(raw_dir: str | Path, flatten_levels: bool = True,
                     as_float32: bool = True) -> xr.Dataset:
    """Open every raw file under `raw_dir`, merge, and normalize to (time,lat,lon[,level])."""
    raw_dir = Path(raw_dir)
    workdir = Path(tempfile.mkdtemp(prefix="era5tw_unzip_"))
    try:
        nc_files = _collect_nc_files(raw_dir, workdir)
        if not nc_files:
            raise FileNotFoundError(
                f"No .nc or .zip files found under {raw_dir}. "
                "Run scripts/01_download_era5.py or scripts/02_make_synthetic.py first."
            )
        datasets = [_normalize(xr.open_dataset(p, decode_times=True)) for p in nc_files]
        ds = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
    finally:
        # combine loads lazily; realize into memory before cleaning temp files.
        ds = ds.load()
        shutil.rmtree(workdir, ignore_errors=True)

    # Sort time ascending and ensure lat descending (ERA5 convention) is preserved.
    if "time" in ds.dims:
        ds = ds.sortby("time")

    if flatten_levels:
        ds = _flatten_levels(ds)

    if as_float32:
        for v in ds.data_vars:
            if np.issubdtype(ds[v].dtype, np.floating):
                ds[v] = ds[v].astype("float32")
    return ds


def _flatten_levels(ds: xr.Dataset) -> xr.Dataset:
    """Turn (time, level, lat, lon) variables into per-level 3-D channels."""
    level_dim = next((d for d in LEVEL_DIM_CANDIDATES if d in ds.dims), None)
    if level_dim is None:
        return ds
    levels = [int(x) for x in ds[level_dim].values]
    out = xr.Dataset(attrs=ds.attrs)
    for name, da in ds.data_vars.items():
        if level_dim in da.dims:
            for lev in levels:
                chan = da.sel({level_dim: lev}).drop_vars(level_dim, errors="ignore")
                out[f"{name}{lev}"] = chan
        else:
            out[name] = da
    # carry through coords (time/lat/lon)
    for c in ("time", "lat", "lon"):
        if c in ds.coords:
            out = out.assign_coords({c: ds[c]})
    return out


def dataset_nbytes(ds: xr.Dataset) -> int:
    """Uncompressed in-memory size of all data variables (bytes)."""
    return int(sum(v.size * v.dtype.itemsize for v in ds.data_vars.values()))


def data_sizes(ds: xr.Dataset) -> dict[str, int]:
    return {d: int(ds.sizes[d]) for d in DATA_DIMS if d in ds.sizes}


def write_zarr(ds: xr.Dataset, out_path: str | Path, strategy: dict[str, Any],
               compressor, mode: str = "w", consolidated: bool = True) -> dict[str, Any]:
    """Write `ds` to Zarr with the given chunking strategy + compressor.

    Returns a metrics dict (does NOT time itself; wrap the call in a Timer).
    """
    out_path = Path(out_path)
    if out_path.exists() and mode == "w":
        shutil.rmtree(out_path)

    sizes = data_sizes(ds)
    resolved = resolve_chunks(strategy, sizes)              # {time,lat,lon}
    ds = ds.chunk(resolved)

    encoding = {}
    for v in ds.data_vars:
        ds[v].encoding.pop("chunks", None)
        ds[v].encoding.pop("preferred_chunks", None)
        encoding[v] = {"compressor": compressor}

    ds.to_zarr(out_path, mode=mode, encoding=encoding, consolidated=consolidated)
    return {
        "chunks": resolved,
        "n_data_vars": len(ds.data_vars),
        "sizes": sizes,
    }
