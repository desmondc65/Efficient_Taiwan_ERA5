#!/usr/bin/env python3
"""Multi-node / multi-GPU DDP data-feeding + training benchmark over the Zarr store.

Launch with torchrun so RANK/WORLD_SIZE/LOCAL_RANK are set. A DistributedSampler
shards the time axis across ranks; every rank reads its own chunks from the shared
Zarr store (chunks are independent objects -> no read contention). We measure
data-loading vs compute time and the global samples/s, so you can plot strong-
scaling efficiency as you add GPUs/nodes.

Single-node, all GPUs:
    torchrun --standalone --nproc_per_node=<NGPU> scripts/07_bench_ddp.py

Single GPU (your RTX 3090) / CPU smoke test:
    torchrun --standalone --nproc_per_node=1 scripts/07_bench_ddp.py
    python scripts/07_bench_ddp.py            # also works, world_size=1

Multi-node: see slurm/ddp_multinode.sbatch.

Options:
    --chunking full_field   which store to read (default: config default)
    --zarr <path>           explicit Zarr store
    --steps 100 --batch-size 32 --num-workers 4
"""
import _bootstrap  # noqa: F401
import argparse
import os


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def setup_dist():
    import torch
    import torch.distributed as dist
    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 1)
    local = env_int("LOCAL_RANK", 0)
    distributed = world > 1 or "RANK" in os.environ
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if distributed:
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local % max(1, torch.cuda.device_count()))
        device = torch.device(f"cuda:{local % max(1, torch.cuda.device_count())}")
    else:
        device = torch.device("cpu")
    return rank, world, local, device, distributed, backend


class TinyForecastNet:
    """Lazily-built small CNN (C->C). Defined as a factory to defer torch import."""
    def __new__(cls, channels, hidden):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--chunking", default=None)
    ap.add_argument("--codec", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from era5tw.config import load_config
    from era5tw.dataset import make_torch_dataset
    from era5tw.metrics import MetricsLogger, Timer
    from era5tw.paths import bench_zarr, production_zarr

    cfg = load_config(args.config)
    dd = cfg["benchmark"]["ddp"]
    steps = args.steps or dd["steps"]
    warmup = args.warmup if args.warmup is not None else dd["warmup_steps"]
    bs = args.batch_size or dd["batch_size"]
    nw = args.num_workers if args.num_workers is not None else dd["num_workers"]
    hidden = dd["model_channels"]
    codec = args.codec or cfg["zarr"]["default_codec"]

    if args.zarr:
        zpath = cfg.abspath(args.zarr)
    elif args.chunking:
        zpath = bench_zarr(cfg, args.chunking, codec)
    else:
        zpath = production_zarr(cfg)

    rank, world, local, device, distributed, backend = setup_dist()
    is_main = rank == 0
    if not zpath.exists():
        if is_main:
            print(f"[error] Zarr store not found: {zpath}\n"
                  f"        Build it first: python scripts/03_build_zarr.py")
        return 2
    if is_main:
        print(f"DDP bench: world_size={world}, backend={backend}, device={device}, "
              f"store={zpath.name}, batch/rank={bs}, workers={nw}")

    dataset = make_torch_dataset(zpath, window=cfg["benchmark"]["dataloader"]["window"])
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 drop_last=True) if distributed else None
    loader = DataLoader(
        dataset, batch_size=bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=nw, pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=(nw > 0), prefetch_factor=(2 if nw > 0 else None),
    )

    channels = dataset[0][0].shape[0]
    model = TinyForecastNet(channels, hidden).to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def batches():
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for b in loader:
                yield b
            epoch += 1

    gen = batches()
    data_t = compute_t = 0.0
    local_samples = 0

    total_iters = warmup + steps
    for i in range(total_iters):
        t0 = Timer().__enter__()
        x, y = next(gen)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0.__exit__()

        t1 = Timer().__enter__()
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1.__exit__()

        if i >= warmup:                       # exclude warmup from timing
            data_t += t0.elapsed
            compute_t += t1.elapsed
            local_samples += x.shape[0]

    step_total = data_t + compute_t

    # Aggregate across ranks: global samples, and the slowest rank's wall time.
    metrics = torch.tensor([local_samples, step_total, data_t, compute_t],
                           dtype=torch.float64, device=device)
    if distributed:
        gsamples = metrics[0].clone(); dist.all_reduce(gsamples, op=dist.ReduceOp.SUM)
        gwall = metrics[1].clone();    dist.all_reduce(gwall, op=dist.ReduceOp.MAX)
        gdata = metrics[2].clone();    dist.all_reduce(gdata, op=dist.ReduceOp.SUM)
        gcomp = metrics[3].clone();    dist.all_reduce(gcomp, op=dist.ReduceOp.SUM)
        global_samples = int(gsamples.item())
        wall = gwall.item()
        data_frac = gdata.item() / (gdata.item() + gcomp.item())
    else:
        global_samples = local_samples
        wall = step_total
        data_frac = data_t / step_total if step_total else 0

    if is_main:
        log = MetricsLogger(cfg.results_dir, "ddp")
        log.log(
            world_size=world, backend=backend,
            device=device.type, gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            store=zpath.name, batch_per_rank=bs, num_workers=nw, channels=channels,
            steps=steps, global_samples=global_samples,
            wall_s=round(wall, 3),
            global_samples_per_s=round(global_samples / wall, 1) if wall else 0,
            per_gpu_samples_per_s=round(global_samples / wall / world, 1) if wall else 0,
            data_time_frac=round(data_frac, 3),
            data_s=round(data_t, 3), compute_s=round(compute_t, 3),
        )
        log.flush_csv()
        print(f"\nworld_size={world}  global={global_samples/wall:.0f} samples/s  "
              f"per-GPU={global_samples/wall/world:.0f}  "
              f"data-bound={data_frac*100:.0f}%  -> {log.csv_path}")
        print("Tip: run with --nproc_per_node 1,2,4 and compare global_samples_per_s "
              "for strong-scaling efficiency.")

    if distributed:
        dist.destroy_process_group()   # also synchronizes ranks on teardown
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
