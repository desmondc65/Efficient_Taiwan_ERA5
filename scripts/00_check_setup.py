#!/usr/bin/env python3
"""Pre-flight check: dependencies, config sanity, and CDS API credentials.

Usage:
    python scripts/00_check_setup.py
    python scripts/00_check_setup.py --write-key <PERSONAL_ACCESS_TOKEN>

The --write-key option creates ~/.cdsapirc for the *new* CDS endpoint so that
scripts/01_download_era5.py can authenticate. You only ever need the token.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from era5tw.config import load_config

CDS_URL = "https://cds.climate.copernicus.eu/api"


def _ver(mod, import_name):
    """Best-effort version string (many packages lack __version__, e.g. cdsapi)."""
    v = getattr(mod, "__version__", None)
    if v:
        return v
    try:
        import importlib.metadata as md
        return md.version(import_name)
    except Exception:
        return "?"


def write_cdsapirc(token: str) -> Path:
    rc = Path.home() / ".cdsapirc"
    rc.write_text(f"url: {CDS_URL}\nkey: {token.strip()}\n")
    rc.chmod(0o600)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--write-key", metavar="TOKEN",
                    help="Write ~/.cdsapirc with this CDS Personal Access Token.")
    args = ap.parse_args()

    if args.write_key:
        rc = write_cdsapirc(args.write_key)
        print(f"[ok] wrote {rc} (url={CDS_URL})")

    ok = True

    # 1) dependencies
    print("\n== dependencies ==")
    for mod in ("numpy", "xarray", "zarr", "numcodecs", "dask", "netCDF4",
                "yaml", "pandas", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"  [ok] {mod:<10} {_ver(m, mod)}")
        except Exception as e:
            ok = False
            print(f"  [MISSING] {mod}: {e}")
    for opt in ("cdsapi", "torch", "psutil", "tqdm"):
        try:
            m = __import__(opt)
            print(f"  [opt] {opt:<10} {_ver(m, opt)}")
        except Exception:
            print(f"  [opt] {opt:<10} not installed")

    # 2) config sanity
    print("\n== config ==")
    cfg = load_config(args.config)
    n, w, s, e = cfg.area
    nlat = round((n - s) / cfg["region"]["resolution"]) + 1
    nlon = round((e - w) / cfg["region"]["resolution"]) + 1
    exp = cfg["region"]["expected_shape"]
    flag = "ok" if [nlat, nlon] == list(exp) else "WARN"
    if flag == "WARN":
        ok = False
    print(f"  area=[N{n},W{w},S{s},E{e}] -> grid {nlat} x {nlon}  (expected {exp})  [{flag}]")
    npr = len(cfg["variables"]["pressure_levels"]["levels"]) * \
        len(cfg["variables"]["pressure_levels"]["variables"])
    nsl = len(cfg["variables"]["single_levels"])
    print(f"  channels: {nsl} surface + {npr} pressure-level = {nsl + npr}")
    print(f"  time:     years={cfg.years} months={cfg.months} "
          f"({len(cfg.days)} days x {len(cfg.hours)} hours)")
    print(f"  chunking strategies: {list(cfg['zarr']['chunking_strategies'])}")
    print(f"  codecs:              {list(cfg['zarr']['codecs'])}")

    # 3) credentials
    print("\n== CDS credentials ==")
    rc = Path.home() / ".cdsapirc"
    if rc.exists():
        txt = rc.read_text()
        has_url = "url:" in txt and "/api" in txt
        has_key = "key:" in txt and len(txt.split("key:")[-1].strip()) > 8
        print(f"  [ok] {rc} present (url={'ok' if has_url else 'MISSING'}, "
              f"key={'ok' if has_key else 'MISSING'})")
        if not (has_url and has_key):
            ok = False
    else:
        print(f"  [info] {rc} not found. Create it with:")
        print(f"         python scripts/00_check_setup.py --write-key <TOKEN>")
        print(f"         (token from https://cds.climate.copernicus.eu -> your profile)")
        print(f"         Not required for the synthetic-data path.")

    print("\n" + ("ALL CORE CHECKS PASSED" if ok else "SOME CHECKS FAILED (see above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
