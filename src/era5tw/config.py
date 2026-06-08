"""Load and resolve config.yaml into plain Python structures.

Importing this module also makes `src/` importable when scripts are run directly
(see scripts/_bootstrap.py for the path injection used by the CLI scripts).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels up from this file (src/era5tw/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

_ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
_ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


class Config:
    """Thin dict wrapper with attribute-style access and path helpers."""

    def __init__(self, data: dict[str, Any], path: Path):
        self.data = data
        self.path = path

    # --- generic access --------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    # --- resolved, ready-to-use views -----------------------------------
    @property
    def area(self) -> list[float]:
        return list(self.data["region"]["area"])  # [N, W, S, E]

    @property
    def years(self) -> list[str]:
        return [str(y) for y in self.data["time"]["years"]]

    @property
    def months(self) -> list[str]:
        return [str(m).zfill(2) for m in self.data["time"]["months"]]

    @property
    def days(self) -> list[str]:
        d = self.data["time"]["days"]
        return list(_ALL_DAYS) if d == "all" else [str(x).zfill(2) for x in d]

    @property
    def hours(self) -> list[str]:
        h = self.data["time"]["hours"]
        return list(_ALL_HOURS) if h == "all" else list(h)

    def abspath(self, rel: str) -> Path:
        """Resolve a config-relative path against the repo root."""
        p = Path(rel)
        return p if p.is_absolute() else (REPO_ROOT / p)

    @property
    def raw_dir(self) -> Path:
        return self.abspath(self.data["download"]["raw_dir"])

    @property
    def zarr_dir(self) -> Path:
        return self.abspath(self.data["zarr"]["out_dir"])

    @property
    def results_dir(self) -> Path:
        return self.abspath(self.data["benchmark"]["results_dir"])


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg_path = cfg_path if cfg_path.is_absolute() else (REPO_ROOT / cfg_path)
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data, cfg_path)
