"""Timing + metric-recording utilities shared by every benchmark script.

Design goals:
  * one consistent results format (a JSONL row per measurement + a flat CSV);
  * every row carries run/system metadata so results are reproducible & comparable;
  * zero heavy dependencies (pandas only used at report time).
"""
from __future__ import annotations

import json
import platform
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# System / run metadata
# ---------------------------------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "n/a"


def system_info() -> dict[str, Any]:
    """Hardware + library fingerprint attached to every recorded row."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        info["cpu_count"] = None
    try:
        import psutil
        info["mem_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        info["mem_total_gb"] = None
    for pkg in ("xarray", "zarr", "numcodecs", "dask", "numpy", "torch"):
        info[f"ver_{pkg}"] = _pkg_version(pkg)
    # GPU info (best effort; only if torch present)
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return info


def run_id() -> str:
    """A monotonically-sortable run id based on wall-clock time."""
    t = time.localtime()
    return time.strftime("%Y%m%d-%H%M%S", t)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@dataclass
class Timer:
    """High-resolution wall-clock timer (context manager)."""
    label: str = ""
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0


@contextmanager
def timed(label: str = "") -> Iterator[Timer]:
    t = Timer(label)
    with t:
        yield t


# ---------------------------------------------------------------------------
# Disk accounting
# ---------------------------------------------------------------------------

def dir_size_bytes(path: str | Path) -> int:
    """Total size (bytes) of every regular file under `path` (recursive)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def count_files(path: str | Path) -> int:
    path = Path(path)
    return sum(1 for p in path.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Results logger
# ---------------------------------------------------------------------------

@dataclass
class MetricsLogger:
    """Append measurement rows to results/<name>.jsonl and results/<name>.csv."""
    results_dir: Path
    name: str
    rid: str = field(default_factory=run_id)
    _sys: dict[str, Any] = field(default_factory=system_info)

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.results_dir / f"{self.name}.jsonl"
        self.csv_path = self.results_dir / f"{self.name}.csv"
        self._rows: list[dict[str, Any]] = []

    def log(self, **fields: Any) -> dict[str, Any]:
        row = {"run_id": self.rid, **fields}
        # keep a compact system fingerprint in each row
        row.setdefault("hostname", self._sys.get("hostname"))
        self._rows.append(row)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps({**row, "_sys": self._sys}) + "\n")
        # human-friendly echo
        pretty = "  ".join(
            f"{k}={_fmt(v)}" for k, v in fields.items() if not k.startswith("_")
        )
        print(f"[{self.name}] {pretty}", flush=True)
        return row

    def _all_rows(self) -> list[dict[str, Any]]:
        """Every row ever written to this metric's JSONL (append-only history),
        with the bulky `_sys` fingerprint stripped. This makes the CSV accumulate
        across separate process invocations (e.g. a multi-world-size DDP sweep)."""
        rows: list[dict[str, Any]] = []
        if not self.jsonl_path.exists():
            return list(self._rows)
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r.pop("_sys", None)
                rows.append(r)
        return rows

    def flush_csv(self) -> Path:
        """(Re)build the flat CSV from the full JSONL history (union of columns)."""
        rows = self._all_rows()
        if not rows:
            return self.csv_path
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        import csv
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return self.csv_path


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.4g}"
    return str(v)
