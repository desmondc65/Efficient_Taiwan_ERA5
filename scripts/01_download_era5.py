#!/usr/bin/env python3
"""Download regional ERA5 from the CDS, one request per (year, month, stream),
running requests concurrently to hide CDS queue latency. Records full timing.

Streams: 'sl' = single (surface) levels, 'pl' = pressure levels.

Examples:
    python scripts/01_download_era5.py                 # uses config.yaml
    python scripts/01_download_era5.py --months 01 02  # override months
    python scripts/01_download_era5.py --concurrency 6
    python scripts/01_download_era5.py --dry-run       # print requests only

Outputs go to <raw_dir>/; metrics to results/download.{jsonl,csv}.
Requests are resumable: an existing, non-empty target file is skipped.
"""
import _bootstrap  # noqa: F401
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from era5tw.config import load_config
from era5tw.metrics import MetricsLogger, Timer, dir_size_bytes


def build_requests(cfg, years, months):
    """Yield (stream, year, month, dataset, request_dict, target_path)."""
    area = cfg.area
    days, hours = cfg.days, cfg.hours
    dl = cfg["download"]
    common = dict(
        product_type=[dl["product_type"]],
        day=days,
        time=hours,
        data_format=dl["data_format"],
        download_format=dl["download_format"],
        area=area,
    )
    raw = cfg.raw_dir
    sl_vars = cfg["variables"]["single_levels"]
    pl = cfg["variables"]["pressure_levels"]
    for y in years:
        for m in months:
            yield ("sl", y, m, "reanalysis-era5-single-levels",
                   {**common, "variable": sl_vars, "year": [y], "month": [m]},
                   raw / f"era5_sl_{y}{m}.nc")
            yield ("pl", y, m, "reanalysis-era5-pressure-levels",
                   {**common, "variable": pl["variables"],
                    "pressure_level": [str(l) for l in pl["levels"]],
                    "year": [y], "month": [m]},
                   raw / f"era5_pl_{y}{m}.nc")


def _fix_extension_if_zip(path: Path) -> Path:
    """CDS sometimes returns a zip even when unarchived was requested."""
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
        if magic == b"PK":  # zip signature
            newp = path.with_suffix(".zip")
            path.rename(newp)
            return newp
    except OSError:
        pass
    return path


def fetch(client, dataset, request, target: Path):
    """Run one CDS retrieve; return (bytes, seconds)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return ("skipped", target.stat().st_size, 0.0, target)
    with Timer() as t:
        client.retrieve(dataset, request, str(target))
    target = _fix_extension_if_zip(target)
    return ("downloaded", target.stat().st_size, t.elapsed, target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--years", nargs="+", default=None)
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    years = args.years or cfg.years
    months = [str(m).zfill(2) for m in (args.months or cfg.months)]
    conc = args.concurrency or cfg["download"]["max_concurrent_requests"]

    reqs = list(build_requests(cfg, years, months))
    print(f"Planned {len(reqs)} requests "
          f"({len(years)} years x {len(months)} months x 2 streams), "
          f"concurrency={conc}, area={cfg.area}")
    for stream, y, m, ds, req, tgt in reqs:
        print(f"  [{stream}] {ds}  {y}-{m}  vars={len(req['variable'])}"
              + (f" x{len(req['pressure_level'])} levels" if 'pressure_level' in req else "")
              + f"  -> {tgt.name}")
    if args.dry_run:
        return 0

    import cdsapi
    client = cdsapi.Client()  # reads ~/.cdsapirc

    log = MetricsLogger(cfg.results_dir, "download")
    total_bytes = 0

    with Timer() as wall:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {
                ex.submit(fetch, client, ds, req, tgt): (stream, y, m)
                for (stream, y, m, ds, req, tgt) in reqs
            }
            for fut in as_completed(futs):
                stream, y, m = futs[fut]
                status, nbytes, secs, tgt = fut.result()
                total_bytes += nbytes
                mbps = (nbytes / 1e6 / secs) if secs > 0 else 0.0
                log.log(stream=stream, year=y, month=m, status=status,
                        file=tgt.name, bytes=nbytes, mb=round(nbytes / 1e6, 2),
                        seconds=round(secs, 2), mb_per_s=round(mbps, 2))

    log.log(stage="TOTAL", requests=len(reqs),
            wall_seconds=round(wall.elapsed, 2),
            total_mb=round(total_bytes / 1e6, 2),
            aggregate_mb_per_s=round(total_bytes / 1e6 / wall.elapsed, 2) if wall.elapsed else 0,
            raw_dir_mb=round(dir_size_bytes(cfg.raw_dir) / 1e6, 2))
    log.flush_csv()
    print(f"\nDone. {total_bytes/1e6:.1f} MB in {wall.elapsed:.1f}s "
          f"({total_bytes/1e6/wall.elapsed:.1f} MB/s aggregate). "
          f"Metrics -> {log.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
