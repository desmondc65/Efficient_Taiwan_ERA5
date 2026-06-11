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

    ll = _load(rd, "lowres_loader", latest=False)
    if ll is not None:
        _section("RELEASED DATASET  -- native-layout DataLoader (LowRes Zarr)")
        cols = [c for c in ["access", "num_workers", "batch_size", "n_frames",
                            "samples_per_s", "eff_mb_per_s", "phys_mb_per_s",
                            "amp_factor"] if c in ll.columns]
        srt = [c for c in ["access", "num_workers"] if c in ll.columns]
        print(ll[cols].sort_values(srt).to_string(index=False))
        print("\n  Native chunking (1326,3,28,16): one random frame reads ~3.6 GB"
              " (amp ~1326x) -> random-access loading is I/O-bound.")
        print("  Contrast: the rechunked full_field store reaches 453 samples/s"
              " (8 workers) above -- a one-time rechunk is essential.")

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


# Publication style: IEEE single-column width (3.45 in), caption-carried titles,
# embedded TrueType fonts, and both raster (png) and vector (pdf) outputs.
_RC = {
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linestyle": ":",
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
}
_COL_W = 3.45  # inches


def _fmt(v):
    """Compact bar label: 170.6 -> '171', 7.4 -> '7.4', 0.032 -> '0.03'."""
    return f"{v:.0f}" if v >= 10 else (f"{v:.1f}" if v >= 1 else f"{v:.2f}")


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")


def _make_plots(rd, cd, ck, dlo, dp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable: {e}")
        return
    plt.rcParams.update(_RC)
    out = rd / "plots"
    out.mkdir(parents=True, exist_ok=True)

    if cd is not None:
        # Codec frontier: compression ratio vs random-frame latency (the
        # training-relevant read). Lower-right = better.
        fig, ax = plt.subplots(figsize=(_COL_W, 2.1))
        ax.scatter(cd["compression_ratio"], cd["rand_frame_ms"],
                   s=22, color="#1f77b4", zorder=3)
        best = cd[cd["codec"] == "blosc_zstd5"]
        if len(best):
            ax.scatter(best["compression_ratio"], best["rand_frame_ms"],
                       s=60, facecolors="none", edgecolors="#d62728",
                       linewidths=1.2, zorder=4)
        off = {"blosc_zstd3": (-2, 7), "blosc_zstd5": (5, -9), "gzip5": (5, -2),
               "blosc_zstd9": (-12, 7)}
        for _, r in cd.iterrows():
            ax.annotate(r["codec"], (r["compression_ratio"], r["rand_frame_ms"]),
                        textcoords="offset points",
                        xytext=off.get(r["codec"], (4, 3)), fontsize=6.5)
        ax.set_xlabel("compression ratio (higher = smaller store)")
        ax.set_ylabel("random-frame read (ms)")
        ax.set_xlim(0.93, 2.12)
        _save(fig, out, "codecs"); plt.close(fig)

    if ck is not None:
        # Two panels sharing x: (a) throughput of both access patterns (log),
        # (b) lossless compression ratio. CSV rounds tiny rates to 0 -- recover
        # them from the per-read latency so every bar is drawable.
        f_ps = [(r["field_per_s"] if r["field_per_s"] > 0 else 1e3 / r["field_read_ms"])
                for _, r in ck.iterrows()]
        s_ps = [(r["series_per_s"] if r["series_per_s"] > 0 else 1e3 / r["series_read_ms"])
                for _, r in ck.iterrows()]
        labels = [c.replace("_", "\n").replace("timeseries", "time\nseries")
                  for c in ck["chunking"]]
        x = range(len(ck))
        fig, (a, b) = plt.subplots(
            2, 1, figsize=(_COL_W, 3.3), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.12})
        a.bar([i - 0.2 for i in x], f_ps, width=0.4,
              label="random full field", color="#1f77b4")
        a.bar([i + 0.2 for i in x], s_ps, width=0.4,
              label="random point series", color="#ff7f0e")
        for i, (fv, sv) in enumerate(zip(f_ps, s_ps)):
            a.annotate(_fmt(fv), (i - 0.2, fv), ha="center", va="bottom", fontsize=6)
            a.annotate(_fmt(sv), (i + 0.2, sv), ha="center", va="bottom", fontsize=6)
        a.set_yscale("log"); a.set_ylim(0.01, 2e3)
        a.set_ylabel("reads/s (log)")
        a.legend(ncol=2, loc="upper center")
        b.bar(list(x), ck["compression_ratio"], width=0.55, color="#2ca02c")
        for i, v in enumerate(ck["compression_ratio"]):
            b.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=6)
        b.set_ylim(0, 10.4)
        b.set_ylabel("compression $\\times$")
        b.set_xticks(list(x)); b.set_xticklabels(labels)
        _save(fig, out, "chunking"); plt.close(fig)

    if dlo is not None:
        # Worker scaling per layout (log y so all layouts' slopes are visible),
        # with a linear-scaling reference anchored at full_field's 1-worker rate.
        fig, ax = plt.subplots(figsize=(_COL_W, 2.3))
        order = {"full_field": 0, "daily": 1, "weekly": 2}
        for name, g in sorted(dlo.groupby("chunking"),
                              key=lambda kv: order.get(kv[0], 9)):
            g = g.sort_values("num_workers")
            ax.plot(g["num_workers"], g["samples_per_s"], marker="o",
                    ms=3.5, label=name)
            w0 = g[g["num_workers"] == 0]["samples_per_s"]
            wmax = g.iloc[-1]
            if len(w0) and w0.iloc[0] > 0:
                ax.annotate(f"{wmax['samples_per_s'] / w0.iloc[0]:.1f}x",
                            (wmax["num_workers"], wmax["samples_per_s"]),
                            textcoords="offset points", xytext=(5, -2), fontsize=6.5)
        ff = dlo[dlo["chunking"] == "full_field"].sort_values("num_workers")
        ff1 = ff[ff["num_workers"] == 1]["samples_per_s"]
        if len(ff1):
            ws = [w for w in ff["num_workers"] if w >= 1]
            ax.plot(ws, [ff1.iloc[0] * w for w in ws], "--", lw=0.9,
                    color="0.5", label="linear from 1 worker")
        ax.set_yscale("log")
        ax.set_xticks(sorted(dlo["num_workers"].unique()))
        ax.set_xlabel("DataLoader workers")
        ax.set_ylabel("samples/s (log)")
        ax.legend(loc="lower right")
        _save(fig, out, "dataloader"); plt.close(fig)

    if dp is not None and "world_size" in dp.columns and len(dp) > 1:
        # (a) measured vs ideal throughput with per-point efficiency;
        # (b) per-step time split: blocked-on-data vs compute.
        d = dp.sort_values("world_size")
        base = d.iloc[0]
        fig, (a, b) = plt.subplots(
            2, 1, figsize=(_COL_W, 3.4),
            gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.42})
        ideal = [base["global_samples_per_s"] / base["world_size"] * w
                 for w in d["world_size"]]
        a.plot(d["world_size"], ideal, "--", color="0.5", lw=0.9, label="ideal linear")
        a.plot(d["world_size"], d["global_samples_per_s"], marker="o", ms=4,
               color="#1f77b4", label="measured")
        for w, s, i in zip(d["world_size"], d["global_samples_per_s"], ideal):
            a.annotate(f"{s / i * 100:.0f}%", (w, s),
                       textcoords="offset points", xytext=(6, -9), fontsize=6.5)
        a.set_xticks(list(d["world_size"]))
        a.set_xlabel("GPUs (world size)")
        a.set_ylabel("global samples/s")
        a.legend(loc="upper left")
        if {"data_s", "compute_s", "steps"}.issubset(d.columns):
            ms_data = d["data_s"] / d["steps"] * 1e3
            ms_comp = d["compute_s"] / d["steps"] * 1e3
            xs = list(d["world_size"])
            b.bar(xs, ms_data, width=0.5, label="blocked on data", color="#ff7f0e")
            b.bar(xs, ms_comp, width=0.5, bottom=ms_data, label="compute",
                  color="#1f77b4")
            for w, md, mc, fr in zip(xs, ms_data, ms_comp, d["data_time_frac"]):
                b.annotate(f"{fr * 100:.0f}%", (w, md / 2), ha="center",
                           va="center", fontsize=6.5, color="white")
            b.set_xticks(xs)
            b.set_ylim(0, (ms_data + ms_comp).max() * 1.35)
            b.set_xlabel("GPUs (world size)")
            b.set_ylabel("ms / step (rank 0)")
            b.legend(ncol=2, loc="upper left")
        _save(fig, out, "ddp_scaling"); plt.close(fig)

    print(f"[plots] wrote PNG+PDF figures to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
