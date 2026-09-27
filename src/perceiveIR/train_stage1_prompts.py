from __future__ import annotations

import argparse
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

from .stage1 import FrozenCLIPImageEncoder, FrozenCLIPTextEncoder, QualityPromptLearner
from .stage1_data import PromptTripletDataset, TaskBalancedDistributedSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-iters", type=int)
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output")
    parser.add_argument("--task-balanced", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    section = config["prompt"]
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    seed = int(config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda", local_rank)
    output = Path(args.output or section["output"])
    if rank == 0:
        (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    logger = logging.getLogger("quality_prompts")
    logger.setLevel(logging.INFO)
    if rank == 0:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream, file_handler = logging.StreamHandler(), logging.FileHandler(output / "train.log")
        stream.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)

    dataset = PromptTripletDataset(section["manifests"], crop_size=224)
    batch_size_per_gpu = int(args.batch_size_per_gpu or section["batch_size_per_gpu"])
    if args.task_balanced:
        sampler = TaskBalancedDistributedSampler(
            dataset, batch_size_per_gpu, world, rank, seed=int(config["seed"])
        )
    else:
        sampler = DistributedSampler(
            dataset, world, rank, shuffle=True, seed=int(config["seed"])
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=int(section["workers_per_gpu"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(section["workers_per_gpu"]) > 0,
    )

    import clip

    clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False, download_root=config["pretrained"]["clip_cache"])
    clip_model.float().requires_grad_(False).eval().to(device)
    prompt_learner = QualityPromptLearner(clip_model, context_length=int(section["context_length"])).to(device)
    text_encoder = FrozenCLIPTextEncoder(clip_model).to(device).eval()
    image_encoder = FrozenCLIPImageEncoder(clip_model).to(device).eval()
    logit_scale = clip_model.logit_scale.exp().detach()
    if world > 1:
        prompt_learner = DistributedDataParallel(prompt_learner, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=float(section["learning_rate"]), betas=(0.9, 0.999))
    total_iters = int(args.total_iters or section["total_iters"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=float(section["minimum_learning_rate"])
    )
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        target = prompt_learner.module if isinstance(prompt_learner, DistributedDataParallel) else prompt_learner
        target.load_state_dict(state["prompt_learner"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = int(state["iteration"])

    if rank == 0:
        logger.info(
            "name=perceiveIR_stage1_prompts world=%d triplets=%d context_tokens=%d total_batch=%d task_balanced=%s",
            world,
            len(dataset),
            section["context_length"],
            batch_size_per_gpu * world,
            args.task_balanced,
        )
    iterator = iter(loader)
    epoch = 0
    began = time.time()
    loss_sum = accuracy_sum = 0.0
    for iteration in range(start + 1, total_iters + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        low = batch["low"].to(device, non_blocking=True)
        medium = batch["medium"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        batch_size = low.shape[0]
        images = torch.cat((low, medium, high), dim=0)
        labels = torch.arange(3, device=device).repeat_interleave(batch_size)
        optimizer.zero_grad(set_to_none=True)
        image_features = image_encoder(images)
        prompts = prompt_learner(torch.empty(0, device=device))
        target = prompt_learner.module if isinstance(prompt_learner, DistributedDataParallel) else prompt_learner
        text_features = F.normalize(text_encoder(prompts, target.tokenized_prompts), dim=-1)
        logits = logit_scale * image_features @ text_features.t()
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        accuracy = (logits.argmax(dim=-1) == labels).float().mean().detach()
        reduced_loss = loss.detach()
        if world > 1:
            dist.all_reduce(reduced_loss)
            dist.all_reduce(accuracy)
            reduced_loss /= world
            accuracy /= world
        loss_sum += float(reduced_loss)
        accuracy_sum += float(accuracy)
        log_every = int(section["log_every"])
        if rank == 0 and (iteration == 1 or iteration % log_every == 0):
            divisor = 1 if iteration == 1 else log_every
            logger.info(
                "iter=%d/%d lr=%.3e ce=%.6f accuracy=%.4f iter_s=%.3f",
                iteration,
                total_iters,
                optimizer.param_groups[0]["lr"],
                loss_sum / divisor,
                accuracy_sum / divisor,
                iteration / max(time.time() - began, 1e-6),
            )
            loss_sum = accuracy_sum = 0.0
        save_every = int(section["save_every"])
        if rank == 0 and (iteration % save_every == 0 or iteration == total_iters):
            target = prompt_learner.module if isinstance(prompt_learner, DistributedDataParallel) else prompt_learner
            path = output / "checkpoints" / f"quality_prompts_{iteration:06d}.pth"
            temporary = path.with_suffix(".pth.tmp")
            torch.save(
                {
                    "prompt_learner": target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iteration": iteration,
                    "quality_names": target.quality_names,
                    "context_length": target.context_length,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "world_size": world,
                    "global_batch_size": batch_size_per_gpu * world,
                    "task_balanced": args.task_balanced,
                },
                temporary,
            )
            temporary.replace(path)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
