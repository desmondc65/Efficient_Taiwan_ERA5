"""Canonical on-disk locations, shared so every script agrees on names."""
from __future__ import annotations

from pathlib import Path

from .config import Config


def production_zarr(cfg: Config) -> Path:
    return cfg.zarr_dir / f"{cfg['region']['name']}_era5.zarr"


def bench_zarr(cfg: Config, chunking: str, codec: str) -> Path:
    return cfg.zarr_dir / "bench" / f"{chunking}__{codec}.zarr"
