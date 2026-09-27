from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .stage1 import RestormerMedium
from .stage1_data import MediumTrainingDataset


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout-fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--total-iters", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output-root")
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--reset-scheduler", action="store_true")
    return parser.parse_args()


def distributed_setup() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return rank, world, local_rank


def bare(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def main() -> None:
    args = arguments()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    section = config["medium"]
    total_iters = int(args.total_iters or section["total_iters"])
    rank, world, local_rank = distributed_setup()
    seed = int(config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda", local_rank)
    output = Path(args.output_root or section["output_root"]) / f"holdout_{args.heldout_fold}"
    if rank == 0:
        (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    logger = logging.getLogger(f"medium_fold_{args.heldout_fold}")
    logger.setLevel(logging.INFO)
    if rank == 0:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream, file_handler = logging.StreamHandler(), logging.FileHandler(output / "train.log")
        stream.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)

    dataset = MediumTrainingDataset(
        config["data"]["root"],
        args.heldout_fold,
        section["patch_size"],
        task_resampling=section.get("task_resampling"),
    )
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=int(config["seed"]))
    batch_size_per_gpu = int(args.batch_size_per_gpu or section["batch_size_per_gpu"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=int(section["workers_per_gpu"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(section["workers_per_gpu"]) > 0,
    )
    model = RestormerMedium(
        dim=config["model"]["dim"],
        blocks=config["model"]["blocks"],
        heads=config["model"]["heads"],
        refinement_blocks=config["model"]["refinement_blocks"],
    ).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(section["learning_rate"]), betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=float(section["minimum_learning_rate"])
    )
    precision = section.get("precision", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = precision in {"bf16", "fp16"}
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        bare(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["iteration"])
        if args.reset_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_iters - start),
                eta_min=float(section["minimum_learning_rate"]),
            )
        else:
            scheduler.load_state_dict(state["scheduler"])

    if rank == 0:
        logger.info(
            "name=perceiveIR_stage1_medium_holdout_%d world=%d raw_samples=%d effective_samples=%d "
            "task_resampling=%s effective_task_counts=%s params=%d batch_per_gpu=%d global_batch=%d",
            args.heldout_fold,
            world,
            len(dataset.raw_samples),
            len(dataset),
            json.dumps(dataset.task_resampling, sort_keys=True),
            json.dumps(dataset.effective_task_counts, sort_keys=True),
            sum(parameter.numel() for parameter in model.parameters()),
            batch_size_per_gpu,
            batch_size_per_gpu * world,
        )
    iterator = iter(loader)
    epoch = 0
    began = time.time()
    running = 0.0
    model.train()
    for iteration in range(start + 1, total_iters + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with autocast:
            prediction = model(lq)
            loss = F.l1_loss(prediction, gt)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        reduced = loss.detach()
        if world > 1:
            dist.all_reduce(reduced)
            reduced /= world
        running += float(reduced)
        log_every = int(section["log_every"])
        if rank == 0 and (iteration == 1 or iteration % log_every == 0):
            divisor = 1 if iteration == 1 else log_every
            logger.info(
                "iter=%d/%d lr=%.3e l1=%.6f iter_s=%.3f",
                iteration,
                total_iters,
                optimizer.param_groups[0]["lr"],
                running / divisor,
                iteration / max(time.time() - began, 1e-6),
            )
            running = 0.0
        save_every = int(section["save_every"])
        if rank == 0 and (iteration % save_every == 0 or iteration == total_iters):
            path = output / "checkpoints" / f"medium_{iteration:06d}.pth"
            temporary = path.with_suffix(".pth.tmp")
            torch.save(
                {
                    "model": bare(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iteration": iteration,
                    "heldout_fold": args.heldout_fold,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "world_size": world,
                    "task_resampling": dataset.task_resampling,
                },
                temporary,
            )
            temporary.replace(path)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
