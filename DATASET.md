# Taiwan ERA5 → Zarr — open dataset

An analysis-ready, **training-optimized** Zarr store of ERA5 reanalysis over a
~2 km Taiwan domain, for regional weather-ML (StormCast / CorrDiff-style)
training. Sourced from ERA5 via the Copernicus CDS and regridded to the Taiwan
grid; shipped already chunked and losslessly compressed so you can train on it
directly — no CDS queue, no acquisition pipeline.

- **Store:** `data/zarr/taiwan_era5_full.zarr`
- **Coverage:** hourly, **2019-08-01 00:00 → 2022-12-31 23:00** (29,976 steps, 3.42 yr)
- **Grid:** `224 × 128` curvilinear ~2 km (lat 21.60–25.60 °N, lon 119.75–122.25 °E)
- **Channels:** 24 (4 surface + 5 vars × 4 pressure levels)
- **Layout:** chunks `(time=1, y=224, x=128)` (`full_field`), codec **blosc-zstd5 + shuffle (lossless)**, `float32`
- **Size:** 43.3 GB on disk / 719,507 chunk files (82.5 GB uncompressed; **1.9× lossless**)

> Built with `scripts/91_build_full_zarr.py`. The chunking is chosen for the
> training access pattern (one random full field at a time); see *Layout notes*.

**⬇ Download (Google Drive):**
<https://drive.google.com/drive/folders/1QEMg1qwU_Q34a9YN-VlyJhQIvSiUIZ3p> —
unzip into `data/zarr/`, then use the recipes below.

---

## What's in the store

```
Dimensions:   (time: 29976, y: 224, x: 128)
Coordinates:
    time       (time)   datetime64[ns]   hourly, 2019-08-01 .. 2022-12-31
    latitude   (y, x)   float32          2-D curvilinear latitude  (°N)
    longitude  (y, x)   float32          2-D curvilinear longitude (°E)
Data variables (24):  each (time, y, x) float32, native ERA5 units
```

`y`/`x` are grid indices (the grid is curvilinear, so latitude **and** longitude
each vary in 2-D). Values are **raw ERA5 units** (not normalized); per-channel
mean/std are given below for standardization.

| variable | meaning | units | mean | std |
|---|---|---|---:|---:|
| `mslp`  | mean sea-level pressure        | Pa     | 101325.77 | 536.66 |
| `t2m`   | 2 m temperature                | K      | 296.150 | 5.003 |
| `u10`   | 10 m eastward wind             | m s⁻¹  | -1.993 | 3.680 |
| `v10`   | 10 m northward wind            | m s⁻¹  | -2.451 | 5.138 |
| `q1000` | specific humidity @ 1000 hPa   | kg kg⁻¹| 0.0140 | 0.0040 |
| `q850`  | specific humidity @ 850 hPa    | kg kg⁻¹| 0.0100 | 0.0030 |
| `q500`  | specific humidity @ 500 hPa    | kg kg⁻¹| 0.0020 | 0.0020 |
| `q250`  | specific humidity @ 250 hPa    | kg kg⁻¹| 0.0002 | 0.0001 |
| `t1000` | temperature @ 1000 hPa         | K      | 296.687 | 4.676 |
| `t850`  | temperature @ 850 hPa          | K      | 289.078 | 4.453 |
| `t500`  | temperature @ 500 hPa          | K      | 267.312 | 2.593 |
| `t250`  | temperature @ 250 hPa          | K      | 233.480 | 1.855 |
| `u1000` | eastward wind @ 1000 hPa       | m s⁻¹  | -2.357 | 4.318 |
| `u850`  | eastward wind @ 850 hPa        | m s⁻¹  | -0.832 | 5.127 |
| `u500`  | eastward wind @ 500 hPa        | m s⁻¹  | 10.175 | 11.934 |
| `u250`  | eastward wind @ 250 hPa        | m s⁻¹  | 16.574 | 19.462 |
| `v1000` | northward wind @ 1000 hPa      | m s⁻¹  | -2.853 | 6.085 |
| `v850`  | northward wind @ 850 hPa       | m s⁻¹  | 0.344 | 4.645 |
| `v500`  | northward wind @ 500 hPa       | m s⁻¹  | 2.379 | 5.331 |
| `v250`  | northward wind @ 250 hPa       | m s⁻¹  | 1.641 | 8.317 |
| `z1000` | geopotential @ 1000 hPa        | m² s⁻² | 1116.94 | 437.53 |
| `z850`  | geopotential @ 850 hPa         | m² s⁻² | 14824.62 | 274.38 |
| `z500`  | geopotential @ 500 hPa         | m² s⁻² | 57386.11 | 423.72 |
| `z250`  | geopotential @ 250 hPa         | m² s⁻² | 107376.66 | 713.42 |

**Canonical channel order** (matches the mean/std arrays and a stacked `(C,H,W)` tensor):

```python
CHANNELS = ["mslp","t2m","u10","v10",
            "q1000","q850","q500","q250", "t1000","t850","t500","t250",
            "u1000","u850","u500","u250", "v1000","v850","v500","v250",
            "z1000","z850","z500","z250"]
```

Geopotential height (m) = `z<level> / 9.80665`.

---

## Quick start (Python / xarray)

```bash
pip install xarray zarr "numcodecs<0.13"     # zarr<3 stack used to write the store
```

```python
import xarray as xr

ds = xr.open_zarr("data/zarr/taiwan_era5_full.zarr", consolidated=True)
print(ds)                      # overview: dims, coords, 24 data variables
print(list(ds.data_vars))      # variable names
print(ds.time.values[[0, -1]]) # first / last timestamp
```

Opening is lazy — nothing is read until you index a variable.

### Extract a variable

```python
# one variable, lazily
z500 = ds["z500"]                         # (time, y, x)

# a time range (label-based) -> still lazy
jan2022 = ds["t2m"].sel(time="2022-01")   # all hourly steps in Jan 2022

# a single timestamp, materialized to a NumPy array
frame = ds["t2m"].sel(time="2022-01-01T00:00").values   # (224, 128)

# a spatial crop by grid index (curvilinear grid -> use isel on y/x)
box = ds["mslp"].isel(y=slice(80, 160), x=slice(40, 100))

# a full time series at one grid point (fast: pulls one column of frames)
series = ds["t2m"].isel(y=112, x=64).values               # (29976,)
```

### Get an ML frame (all 24 channels at one time) as `(C, H, W)`

```python
import numpy as np
CHANNELS = ["mslp","t2m","u10","v10","q1000","q850","q500","q250",
            "t1000","t850","t500","t250","u1000","u850","u500","u250",
            "v1000","v850","v500","v250","z1000","z850","z500","z250"]

t = 1000
frame = np.stack([ds[c].isel(time=t).values for c in CHANNELS])   # (24, 224, 128)
```

### Normalize with the provided statistics

The per-channel mean/std ship inside the store as `stats.npz` (arrays in the
canonical channel order above):

```python
st = np.load("data/zarr/taiwan_era5_full.zarr/stats.npz", allow_pickle=True)
MEAN, STD, chans = st["mean"], st["std"], list(st["channels"])
norm = (frame - MEAN[:, None, None]) / STD[:, None, None]   # frame: (24, 224, 128)
```

Or hard-code them:

```python
MEAN = np.array([101325.77,296.150,-1.993,-2.451,0.0140,0.0100,0.0020,0.0002,
                 296.687,289.078,267.312,233.480,-2.357,-0.832,10.175,16.574,
                 -2.853,0.344,2.379,1.641,1116.94,14824.62,57386.11,107376.66])
STD  = np.array([536.66,5.003,3.680,5.138,0.0040,0.0030,0.0020,0.0001,
                 4.676,4.453,2.593,1.855,4.318,5.127,11.934,19.462,
                 6.085,4.645,5.331,8.317,437.53,274.38,423.72,713.42])
norm = (frame - MEAN[:, None, None]) / STD[:, None, None]
```

### Find the nearest grid cell to a (lat, lon)

The grid is curvilinear, so use the 2-D coordinates:

```python
lat0, lon0 = 25.03, 121.56          # ~Taipei
lat = ds["latitude"].values; lon = ds["longitude"].values
j, i = np.unravel_index(np.argmin((lat-lat0)**2 + (lon-lon0)**2), lat.shape)
taipei_t2m = ds["t2m"].isel(y=j, x=i).values    # hourly series at that cell
```

---

## PyTorch training loader (next-frame, the layout's intended use)

The `full_field` chunking makes each random timestep one chunk per channel, so a
shuffled DataLoader reads independent chunks and scales across workers.

```python
import numpy as np, torch, xarray as xr
from torch.utils.data import Dataset, DataLoader

class TaiwanERA5(Dataset):
    def __init__(self, path, channels, window=2):
        self.path, self.channels, self.window = path, channels, window
        self._ds = None
        self._n = xr.open_zarr(path, consolidated=True).sizes["time"]
    def _open(self):                      # fork-safe: reopen per worker
        if self._ds is None:
            self._ds = xr.open_zarr(self.path, consolidated=True)
        return self._ds
    def __len__(self):  return self._n - (self.window - 1)
    def _frame(self, t):
        d = self._open()
        return np.stack([d[c].isel(time=t).values for c in self.channels])
    def __getitem__(self, t):
        x = self._frame(t); y = self._frame(t + self.window - 1)
        return torch.from_numpy(x), torch.from_numpy(y)

loader = DataLoader(TaiwanERA5("data/zarr/taiwan_era5_full.zarr", CHANNELS),
                    batch_size=32, shuffle=True, num_workers=8,
                    persistent_workers=True, drop_last=True)
for x, y in loader:        # x, y: (32, 24, 224, 128)
    ...
```

---

## Layout notes

- **Why `full_field`?** Training reads *one random full field at a time*. With
  `(1, 224, 128)` chunks that is exactly one chunk per channel, so reads are fast
  and parallelize with `num_workers` (≈ 6× from 0→8 workers in our benchmarks).
- **Smaller download instead?** If you only need bulk/archival storage, rechunk
  to bundle time (e.g. `spatial_block (24,32,32)`) for up to **8.5×** compression
  (~10 GB) at the cost of slower random-frame reads:
  ```python
  ds.chunk({"time": 24, "y": 32, "x": 32}).to_zarr("taiwan_era5_block.zarr")
  ```
- **Time-series workloads** (one point, long history) are slow on this layout;
  rechunk to `(time=-1, y=1, x=1)` if that is your access pattern.

## Provenance

ERA5 hourly reanalysis (Copernicus Climate Change Service / ECMWF), retrieved
via the CDS and regridded to the ~2 km Taiwan grid used by the StormCast "LowRes"
inputs. ERA5 is © ECMWF, licensed under the Copernicus license; cite Hersbach et
al. (2020) and acknowledge the Copernicus Climate Data Store when using this data.
```
