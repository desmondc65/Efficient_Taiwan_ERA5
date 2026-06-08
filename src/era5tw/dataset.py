"""Zarr-backed readers for the benchmarks.

Two layers:
  * ZarrFieldReader  -- numpy-only random access to the (time, lat, lon) channels.
                        Used by the chunking benchmark to probe access patterns.
  * ZarrForecastDataset -- a torch Dataset (next-frame forecasting) built on top,
                        used by the DataLoader and DDP benchmarks.

Both are fork-safe: the underlying Zarr handle is (re)opened per process so that
PyTorch DataLoader workers each get an independent store, which is exactly the
"each chunk is an independent object -> parallel reads" property we benchmark.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_COORD_NAMES = {"time", "lat", "lon", "latitude", "longitude", "valid_time"}


def _open_group(path: str | Path):
    import zarr
    path = str(path)
    try:
        return zarr.open_consolidated(path, mode="r")
    except Exception:
        return zarr.open_group(path, mode="r")


class ZarrFieldReader:
    """Random access to stacked (channel, lat, lon) fields from a Zarr store."""

    def __init__(self, zarr_path: str | Path, variables: list[str] | None = None):
        self.path = str(zarr_path)
        self._pid = -1
        self._z = None
        # Discover channel variables once (cheap metadata read).
        z = _open_group(self.path)
        if variables is None:
            variables = []
            for name, arr in z.arrays():
                dims = arr.attrs.get("_ARRAY_DIMENSIONS", [])
                if name in _COORD_NAMES:
                    continue
                if arr.ndim == 3 and list(dims) == ["time", "lat", "lon"]:
                    variables.append(name)
                elif arr.ndim == 3 and name not in _COORD_NAMES:
                    variables.append(name)
            variables = sorted(variables)
        if not variables:
            raise ValueError(f"No 3-D (time,lat,lon) variables found in {self.path}")
        self.variables = variables
        sample = z[variables[0]]
        self.n_time, self.n_lat, self.n_lon = sample.shape
        self.n_channels = len(variables)
        self.dtype = np.float32

    # -- fork-safe handle ------------------------------------------------
    def _store(self):
        pid = os.getpid()
        if self._z is None or pid != self._pid:
            self._z = _open_group(self.path)
            self._pid = pid
        return self._z

    # -- access patterns -------------------------------------------------
    def read_frame(self, t: int) -> np.ndarray:
        """One timestep, all channels -> (C, H, W). 'Full spatial field' pattern."""
        z = self._store()
        out = np.empty((self.n_channels, self.n_lat, self.n_lon), dtype=self.dtype)
        for c, v in enumerate(self.variables):
            out[c] = z[v][t]
        return out

    def read_point_series(self, y: int, x: int) -> np.ndarray:
        """Full time series at one grid point, all channels -> (C, T). 'LSTM' pattern."""
        z = self._store()
        out = np.empty((self.n_channels, self.n_time), dtype=self.dtype)
        for c, v in enumerate(self.variables):
            out[c] = z[v][:, y, x]
        return out

    def __len__(self) -> int:
        return self.n_time


class ZarrForecastDataset:
    """Next-frame forecasting samples: (input_frame_t, target_frame_t+window-1).

    Subclasses torch.utils.data.Dataset lazily so the module imports without torch.
    """

    def __init__(self, zarr_path, variables=None, window: int = 2):
        import torch  # noqa: F401  (validates torch presence early)
        self.reader = ZarrFieldReader(zarr_path, variables)
        self.window = int(window)

    def __len__(self) -> int:
        return self.reader.n_time - (self.window - 1)

    def __getitem__(self, idx: int):
        import torch
        x = self.reader.read_frame(idx)
        y = self.reader.read_frame(idx + self.window - 1)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_torch_dataset(zarr_path, variables=None, window: int = 2):
    """Factory that returns an object usable as a torch Dataset."""
    import torch.utils.data as tud

    class _DS(tud.Dataset):
        def __init__(self):
            self._inner = ZarrForecastDataset(zarr_path, variables, window)

        def __len__(self):
            return len(self._inner)

        def __getitem__(self, i):
            return self._inner[i]

    return _DS()
