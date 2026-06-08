#!/usr/bin/env python3
"""Summarize all recorded metrics into console tables and (optionally) plots.

Reads results/*.csv produced by scripts 01,03,04,05,06,07 and prints compact
summaries: download throughput, storage footprint, codec trade-offs, chunking
access-pattern trade-offs, dataloader worker-scaling, and DDP strong-scaling.

Usage:
    python scripts/08_report.py
    python scripts/08_report.py --plots      # also write results/plots/*.png
"""
import _bootstrap  # noqa: F401
import argparse

from era5tw.config import load_config


def _load(results_dir, name, latest=True):
    """Load results/<name>.csv. With latest=True keep only the most recent run_id
    (the CSV accumulates history across runs)."""
    import pandas as pd
    p = results_dir / f"{name}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if latest and "run_id" in df.columns and len(df):
        df = df[df["run_id"] == df["run_id"].max()].reset_index(drop=True)
    return df


def _latest_per(df, key):
    """Keep the most recent row for each value of `key` (for the DDP sweep:
    one row per world_size, each from its own run)."""
    if df is None or key not in df.columns or "run_id" not in df.columns:
        return df
    idx = df.groupby(key)["run_id"].transform("max") == df["run_id"]
    return df[idx].drop_duplicates(subset=[key], keep="last").reset_index(drop=True)


def _section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rd = cfg.results_dir
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    dl = _load(rd, "download")
    if dl is not None:
        _section("DOWNLOAD")
        tot = dl[dl.get("stage") == "TOTAL"] if "stage" in dl else None
        cols = [c for c in ["stream", "year", "month", "mb", "seconds", "mb_per_s", "status"] if c in dl.columns]
        print(dl[cols].to_string(index=False))
        if tot is not None and len(tot):
            print("\n" + tot[[c for c in ["requests", "wall_seconds", "total_mb", "aggregate_mb_per_s"] if c in tot.columns]].to_string(index=False))

    bz = _load(rd, "build_zarr")
    if bz is not None:
        _section("PRODUCTION ZARR / STORAGE FOOTPRINT")
        print(bz[[c for c in ["chunking", "codec", "channels", "raw_netcdf_mb",
                              "uncompressed_mb", "zarr_mb", "compression_ratio",
                              "write_seconds", "write_mb_per_s"] if c in bz.columns]].to_string(index=False))

    cd = _load(rd, "codecs")
    if cd is not None:
        _section("CODECS  (fixed chunking) -- compression vs read/write speed")
        print(cd[[c for c in ["codec", "zarr_mb", "compression_ratio", "write_s",
                              "write_mb_per_s", "full_read_mb_per_s",
                              "rand_frame_ms"] if c in cd.columns]].to_string(index=False))

    ck = _load(rd, "chunking")
    if ck is not None:
        _section("CHUNKING  (fixed codec) -- full-field vs time-series access")
        print(ck[[c for c in ["chunking", "chunk_shape", "n_files", "zarr_mb",
                              "field_per_s", "field_mb_per_s",
                              "series_per_s", "series_mb_per_s"] if c in ck.columns]].to_string(index=False))
        print("\n  field_per_s = random full spatial fields/s (CorrDiff/StormCast);")
        print("  series_per_s = random point time-series/s (LSTM).")

    dlo = _load(rd, "dataloader")
    if dlo is not None:
        _section("DATALOADER  -- samples/s vs num_workers per chunking")
        piv = dlo.pivot_table(index="chunking", columns="num_workers",
                              values="samples_per_s", aggfunc="max")
        print(piv.to_string())

    dp = _latest_per(_load(rd, "ddp", latest=False), "world_size")
    if dp is not None:
        _section("DDP  -- strong scaling")
        cols = [c for c in ["world_size", "backend", "store", "num_workers",
                            "global_samples_per_s", "per_gpu_samples_per_s",
                            "data_time_frac"] if c in dp.columns]
        d = dp[cols].sort_values("world_size") if "world_size" in dp.columns else dp[cols]
        print(d.to_string(index=False))
        if "world_size" in dp.columns and len(dp) > 1:
            base = dp.sort_values("world_size").iloc[0]
            b_ws, b_tp = base["world_size"], base["global_samples_per_s"]
            print("\n  scaling efficiency vs smallest world_size:")
            for _, r in dp.sort_values("world_size").iterrows():
                ideal = b_tp / b_ws * r["world_size"]
                eff = r["global_samples_per_s"] / ideal * 100 if ideal else 0
                print(f"    world_size={int(r['world_size'])}: "
                      f"{r['global_samples_per_s']:.0f} samples/s  "
                      f"({eff:.0f}% of ideal)")

    if args.plots:
        _make_plots(rd, cd, ck, dlo, dp)
    return 0


def _make_plots(rd, cd, ck, dlo, dp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable: {e}")
        return
    out = rd / "plots"
    out.mkdir(parents=True, exist_ok=True)

    if cd is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(cd["compression_ratio"], cd["full_read_mb_per_s"])
        for _, r in cd.iterrows():
            ax.annotate(r["codec"], (r["compression_ratio"], r["full_read_mb_per_s"]),
                        fontsize=8)
        ax.set_xlabel("compression ratio (higher=smaller)")
        ax.set_ylabel("full read MB/s (higher=faster)")
        ax.set_title("Codec trade-off: size vs read speed")
        fig.tight_layout(); fig.savefig(out / "codecs.png", dpi=120); plt.close(fig)

    if ck is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(ck))
        ax.bar([i - 0.2 for i in x], ck["field_per_s"], width=0.4, label="full-field/s")
        ax.bar([i + 0.2 for i in x], ck["series_per_s"], width=0.4, label="point-series/s")
        ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(ck["chunking"], rotation=30)
        ax.set_ylabel("reads/s (log)"); ax.legend(); ax.set_title("Chunking: access-pattern trade-off")
        fig.tight_layout(); fig.savefig(out / "chunking.png", dpi=120); plt.close(fig)

    if dlo is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, g in dlo.groupby("chunking"):
            g = g.sort_values("num_workers")
            ax.plot(g["num_workers"], g["samples_per_s"], marker="o", label=name)
        ax.set_xlabel("num_workers"); ax.set_ylabel("samples/s")
        ax.set_title("DataLoader worker scaling"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "dataloader.png", dpi=120); plt.close(fig)

    if dp is not None and "world_size" in dp.columns and len(dp) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        d = dp.sort_values("world_size")
        ax.plot(d["world_size"], d["global_samples_per_s"], marker="o", label="measured")
        base = d.iloc[0]
        ideal = [base["global_samples_per_s"] / base["world_size"] * w for w in d["world_size"]]
        ax.plot(d["world_size"], ideal, "--", label="ideal linear")
        ax.set_xlabel("world_size (GPUs)"); ax.set_ylabel("global samples/s")
        ax.set_title("DDP strong scaling"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "ddp_scaling.png", dpi=120); plt.close(fig)

    print(f"[plots] wrote PNGs to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
