from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import PaperFiveDataset, TASK_TO_ID
from .epoch_probe import ensure_manifest, evaluate_model_psnr
from .losses import DifficultyAdaptivePerceptualLoss, QualityCLIPLoss, degradation_contrastive_loss
from .model import DinoSemanticEncoder, PerceiveIR
from .proxy import DenoiseProxyModels, FSNetProxy, InstructIRProxy, MambaIRProxy, PromptIRProxy
from .sampling import GlobalBatchDistributedSampler
from .validation import evaluate as evaluate_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-iters", type=int)
    parser.add_argument("--batch-sizes", help="comma-separated per-rank batch sizes")
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--output")
    parser.add_argument("--resume")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return rank, world_size, local_rank


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def make_logger(output: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger("perceiveIR")
    logger.setLevel(logging.INFO)
    if rank == 0 and not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(output / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    return logger


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def patch_psnr(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (image.float().clamp(0, 1) - target.float()).square().mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-8))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    iteration: int,
    config: dict,
    training_state: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iteration": iteration,
            "config": config,
            "training_state": training_state,
        },
        temporary,
    )
    temporary.replace(path)


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_last_link(output: Path, checkpoint: Path) -> None:
    link = output / "last.pth"
    temporary = output / ".last.pth.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(checkpoint, output))
    temporary.replace(link)


def save_best_model(path: Path, model: nn.Module, iteration: int, config: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": unwrap(model).state_dict(), "iteration": iteration, "config": config}, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.total_iters is not None:
        config["train"]["total_iters"] = args.total_iters
    if args.output is not None:
        config["output"] = args.output
    if args.batch_sizes is not None:
        config["data"]["batch_size_by_rank"] = [int(size) for size in args.batch_sizes.split(",")]
    if args.val_every is not None:
        config["validation"]["every"] = args.val_every
    if args.save_every is not None:
        config["train"]["save_every"] = args.save_every

    rank, world_size, local_rank = setup_distributed()
    seed_everything(int(config["seed"]), rank)
    device = torch.device("cuda", local_rank)
    output = Path(config["output"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
        with open(output / "resolved_config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    if world_size > 1:
        dist.barrier()
    logger = make_logger(output, rank)

    dataset = PaperFiveDataset(
        config["data"]["root"],
        patch_size=int(config["data"]["patch_size"]),
        seed=int(config["seed"]),
        tasks=config["data"].get("tasks", ("denoise", "dehaze", "derain", "deblur", "lowlight")),
        task_resampling=config["data"].get("task_resampling"),
        proxy_manifests=config["data"].get("proxy_manifests", ()),
        special_proxy_manifest=config["data"].get("special_proxy_manifest"),
        dynamic_denoise_proxy=config["data"].get("dynamic_denoise_proxy", False),
        shared_noise_positive=config["data"].get("shared_noise_positive", False),
    )
    batch_sizes = config["data"].get("batch_size_by_rank")
    if batch_sizes is None:
        batch_sizes = [int(config["data"]["batch_size_per_gpu"])] * world_size
    if len(batch_sizes) != world_size or any(int(size) < 1 for size in batch_sizes):
        raise ValueError("batch_size_by_rank must contain one positive size per process")
    batch_sizes = [int(size) for size in batch_sizes]
    local_batch_size = batch_sizes[rank]
    global_batch_size = sum(batch_sizes)
    sampler = GlobalBatchDistributedSampler(len(dataset), batch_sizes, rank, int(config["seed"]))
    steps_per_epoch = sampler.steps_per_epoch
    probe_path = output / "dpl_probe" / "manifest.json"
    probe_count = int(config["train"].get("dpl_probe_per_task", 24))
    if rank == 0:
        ensure_manifest(probe_path, dataset.samples, dataset.patch_size, probe_count, int(config["seed"]))
    if world_size > 1:
        dist.barrier()
    probe_manifest = ensure_manifest(probe_path, dataset.samples, dataset.patch_size, probe_count, int(config["seed"]))
    probe_hash = hashlib.sha1(json.dumps(probe_manifest["records"], sort_keys=True).encode()).hexdigest()
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=int(config["data"]["workers_per_gpu"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(config["data"]["workers_per_gpu"]) > 0,
    )

    model = PerceiveIR(
        dim=int(config["model"]["dim"]),
        blocks=tuple(config["model"]["blocks"]),
        heads=tuple(config["model"]["heads"]),
        refinement_blocks=int(config["model"]["refinement_blocks"]),
    ).to(device)
    semantic_encoder = DinoSemanticEncoder(config["pretrained"]["dinov2"]).to(device)
    quality_loss = QualityCLIPLoss(
        config["pretrained"]["clip_cache"], config["pretrained"]["quality_prompt_checkpoint"]
    ).to(device)
    perceptual_loss = DifficultyAdaptivePerceptualLoss(config["pretrained"]["vgg16"]).to(device)
    proxy_models = None
    if config["data"].get("dynamic_denoise_proxy", False):
        proxy_models = DenoiseProxyModels(
            config["pretrained"]["denoise_proxy_checkpoints"], config["model"], device,
        )
    promptir_proxy = None
    promptir_config = config["pretrained"].get("promptir_proxy")
    instructir_config = config["pretrained"].get("instructir_proxy")
    fsnet_config = config["pretrained"].get("fsnet_proxy")
    mambair_config = config["pretrained"].get("mambair_proxy")
    extra_proxy_configs = (promptir_config, instructir_config, fsnet_config, mambair_config)
    if any(extra_proxy_configs) and not dataset.proxy_lookup:
        raise RuntimeError("multi-proxy training requires first-stage proxy manifests")
    if promptir_config:
        promptir_proxy = PromptIRProxy(
            promptir_config["source"], promptir_config["checkpoint"], device,
        )
    instructir_proxy = InstructIRProxy(
        instructir_config["source"], instructir_config["checkpoint"],
        instructir_config["embeddings"], device,
    ) if instructir_config else None
    fsnet_proxy = FSNetProxy(
        fsnet_config["source"], fsnet_config["checkpoint"], device,
    ) if fsnet_config else None
    mambair_proxy = MambaIRProxy(
        mambair_config["source"], mambair_config["checkpoints"], device,
    ) if mambair_config else None
    proxy_names = (["restormer"] if dataset.proxy_lookup else []) + [
        name for name, candidate in (("promptir", promptir_proxy), ("instructir", instructir_proxy),
                                     ("fsnet", fsnet_proxy), ("mambair", mambair_proxy))
        if candidate is not None
    ]
    if dataset.special_proxy_lookup:
        proxy_names.append("task_specific")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    total_iters = int(config["train"]["total_iters"])
    schedule_iters = int(config["train"].get("schedule_iters", total_iters))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=schedule_iters, eta_min=float(config["train"]["minimum_learning_rate"])
    )
    precision = config["train"].get("precision", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = precision in {"bf16", "fp16"}
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    weights = config["train"]["loss_weights"]

    start_iteration = 0
    epoch_reference_psnr = None
    epoch_origin_iteration = 0
    epoch_index_offset = 0
    batch_changed_on_resume = False
    resume = args.resume or config["train"].get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        previous = checkpoint["config"]
        if bool(previous["data"].get("shared_noise_positive", False)) != dataset.shared_noise_positive:
            raise RuntimeError("cannot resume across a CFE positive-pair definition change")
        for key in ("promptir_proxy", "instructir_proxy", "fsnet_proxy", "mambair_proxy"):
            if previous["pretrained"].get(key) != config["pretrained"].get(key):
                raise RuntimeError(f"cannot resume across a DPL {key} change")
        unwrap(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_iteration = int(checkpoint["iteration"])
        training_state = checkpoint.get("training_state", {})
        if training_state.get("probe_hash") != probe_hash:
            raise RuntimeError("resume checkpoint has a different DPL training probe")
        epoch_reference_psnr = training_state.get("epoch_reference_psnr")
        epoch_origin_iteration = int(training_state.get("epoch_origin_iteration", 0))
        epoch_index_offset = int(training_state.get("epoch_index_offset", 0))
        old_batch_sizes = checkpoint["config"]["data"].get("batch_size_by_rank")
        if old_batch_sizes is None:
            old_batch_sizes = [int(checkpoint["config"]["data"]["batch_size_per_gpu"])] * world_size
        if len(old_batch_sizes) != world_size:
            raise RuntimeError("resume checkpoint world size differs from the current run")
        batch_changed_on_resume = tuple(old_batch_sizes) != tuple(batch_sizes)
        if batch_changed_on_resume:
            old_steps_per_epoch = (len(dataset) + sum(old_batch_sizes) - 1) // sum(old_batch_sizes)
            completed_old_epochs = (start_iteration - epoch_origin_iteration) // old_steps_per_epoch
            epoch_index_offset += completed_old_epochs + 1
            epoch_origin_iteration = start_iteration
            epoch_reference_psnr = None
            logger.info(
                "batch transition %s -> %s at iteration %d; starting a fresh DPL/sampler epoch",
                old_batch_sizes, batch_sizes, start_iteration,
            )
        elif (start_iteration - epoch_origin_iteration) % steps_per_epoch and epoch_reference_psnr is None:
            raise RuntimeError("resume checkpoint lacks the current epoch's DPL PSNR threshold")
        logger.info("resumed from %s at iteration %d", resume, start_iteration)

    retain_best = bool(config["train"].get("retain_best_model", False))
    retain_last = bool(config["train"].get("retain_last", False))
    if rank == 0 and resume:
        if retain_last and not (output / "last.pth").exists():
            update_last_link(output, Path(resume).resolve())
        initial_summary_path = config.get("validation", {}).get("initial_summary")
        best_path = output / "validation" / "best.json"
        if retain_best and initial_summary_path and not best_path.exists():
            initial = json.loads(Path(initial_summary_path).read_text(encoding="utf-8"))["summary"]
            if initial["iteration"] != start_iteration or not math.isfinite(initial["macro_psnr"]):
                raise RuntimeError("initial validation summary does not match the resume checkpoint")
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_model_path = output / "best_model.pth"
            save_best_model(best_model_path, model, start_iteration, config)
            write_json_atomic(best_path, {**initial, "checkpoint": str(best_model_path)})
            logger.info("seeded validation best from iteration %d: macro_psnr=%.4f", start_iteration, initial["macro_psnr"])
    if world_size > 1:
        dist.barrier()

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if rank == 0:
        logger.info("experiment=%s world_size=%d trainable_parameters=%d", config["name"], world_size, trainable)
        logger.info("optimizer_steps=%d schedule_steps=%d", total_iters, schedule_iters)
        logger.info("dataset_size=%d task_counts=%s", len(dataset), dict(dataset.task_counts))
        logger.info("proxy_restorations=%d", len(dataset.proxy_lookup))
        logger.info(
            "dpl_proxy_models=%s shared_noise_positive=%s",
            proxy_names,
            dataset.shared_noise_positive,
        )
        logger.info("dpl_probe=%s samples=%d steps_per_epoch=%d", probe_path, len(probe_manifest["records"]), steps_per_epoch)
        logger.info(
            "patch=%d batch_by_rank=%s effective_batch=%d precision=%s prompt=%s",
            config["data"]["patch_size"],
            batch_sizes,
            global_batch_size,
            precision,
            config["pretrained"]["quality_prompt_checkpoint"],
        )

    model.train()
    resumed_step_in_epoch = (start_iteration - epoch_origin_iteration) % steps_per_epoch
    sampler.set_epoch(
        epoch_index_offset + (start_iteration - epoch_origin_iteration) // steps_per_epoch,
        resumed_step_in_epoch,
    )
    data_iterator = iter(loader)
    running: dict[str, float] = {}
    proxy_diagnostics = torch.zeros(3 * len(proxy_names) + 1, device=device) if any(extra_proxy_configs) else None
    diagnostic_steps = 0
    started = time.time()
    log_every = int(config["train"]["log_every"])
    save_every = int(config["train"]["save_every"])
    val_config = config.get("validation")
    val_every = int(val_config["every"]) if val_config else 0
    if val_config and not Path(val_config["manifest"]).is_file():
        raise FileNotFoundError(f"fixed validation manifest is missing: {val_config['manifest']}")
    optimizer.zero_grad(set_to_none=True)

    for iteration in range(start_iteration + 1, total_iters + 1):
        relative_iteration = iteration - 1 - epoch_origin_iteration
        if relative_iteration % steps_per_epoch == 0:
            epoch = epoch_index_offset + relative_iteration // steps_per_epoch
            sampler.set_epoch(epoch)
            data_iterator = iter(loader)
            epoch_reference_psnr = evaluate_model_psnr(
                probe_manifest, unwrap(model), semantic_encoder, device, rank, world_size, amp_dtype,
            )
            if rank == 0:
                logger.info("epoch=%d start_iter=%d model_probe_psnr=%.4f", epoch + 1, iteration, epoch_reference_psnr)
        try:
            batch = next(data_iterator)
        except StopIteration:
            raise RuntimeError("global-batch sampler exhausted before its synchronized epoch boundary")
        lq = batch["lq"].to(device, non_blocking=True)
        positive_lq = batch["positive_lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        proxy = batch["proxy"].to(device, non_blocking=True) if "proxy" in batch else None
        labels = batch["task"].to(device, non_blocking=True)
        noise_sigmas = batch["noise_sigma"].to(device, non_blocking=True)
        if proxy is not None and proxy_models is not None:
            folds = batch["proxy_fold"].to(device, non_blocking=True)
            proxy = proxy_models.render(lq, proxy, labels, folds, amp_dtype)
        proxies = [proxy] if proxy is not None else []
        proxy_masks = [torch.ones_like(labels, dtype=torch.bool)] if proxy is not None else []
        if promptir_proxy is not None:
            proxies.append(promptir_proxy.render(lq, amp_dtype))
            # The official PromptIR checkpoint covers only N/H/R.  Other tasks
            # still receive the stage-one and InstructIR proxy negatives.
            proxy_masks.append(labels < TASK_TO_ID["deblur"])
        if instructir_proxy is not None:
            proxies.append(instructir_proxy.render(lq, labels, amp_dtype))
            proxy_masks.append(torch.ones_like(labels, dtype=torch.bool))
        if fsnet_proxy is not None:
            candidate, valid = fsnet_proxy.render(lq, labels, amp_dtype)
            proxies.append(candidate)
            proxy_masks.append(valid)
        if mambair_proxy is not None:
            candidate, valid = mambair_proxy.render(lq, labels, noise_sigmas, amp_dtype)
            proxies.append(candidate)
            proxy_masks.append(valid)
        if dataset.special_proxy_lookup:
            special_proxy = batch["special_proxy"].to(device, non_blocking=True)
            valid = batch["special_proxy_valid"].to(device, non_blocking=True)
            if not torch.equal(valid, labels >= TASK_TO_ID["deblur"]):
                raise RuntimeError("task-specific proxy mask does not match deblur/lowlight labels")
            proxies.append(special_proxy)
            proxy_masks.append(valid)
        proxy_weights = []
        proxy_psnrs = []
        if epoch_reference_psnr is not None:
            for candidate in proxies:
                candidate_psnr = patch_psnr(candidate, gt)
                proxy_psnrs.append(candidate_psnr)
                proxy_weights.append(torch.where(
                    candidate_psnr <= epoch_reference_psnr,
                    torch.full_like(candidate_psnr, 1.25),
                    torch.full_like(candidate_psnr, 0.75),
                ))

        autocast = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with autocast:
            semantics = semantic_encoder(lq)
            restored, degradation, positive = model(
                lq, semantics, return_positive=True, positive_image=positive_lq
            )
            reconstruction = F.l1_loss(restored, gt)
            contrastive, cl_valid_fraction = degradation_contrastive_loss(
                degradation, positive, labels, temperature=0.1, return_valid_fraction=True,
            )
            clip_term = quality_loss(restored)
            perceptual = perceptual_loss(
                restored, gt, lq,
                proxies=proxies,
                proxy_weights=proxy_weights,
                proxy_masks=proxy_masks,
            )
            total = (
                reconstruction
                + float(weights["contrastive"]) * contrastive
                + float(weights["clip"]) * clip_term
                + float(weights["perceptual"]) * perceptual
            )

        # DDP averages gradients across ranks; compensate when rank batches differ.
        scaler.scale(total * (world_size * local_batch_size / global_batch_size)).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        values = {
            "total": total.detach(),
            "rec": reconstruction.detach(),
            "cl": contrastive.detach(),
            "clip": clip_term.detach(),
            "dpl": perceptual.detach(),
        }
        for key, value in values.items():
            value = value * local_batch_size
            if world_size > 1:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            running[key] = running.get(key, 0.0) + float(value / global_batch_size)

        if proxy_diagnostics is not None:
            if len(proxy_weights) != len(proxy_names):
                raise RuntimeError("one weighted DPL negative is required per named proxy")
            for index, valid in enumerate(proxy_masks):
                proxy_diagnostics[3 * index] += valid.float().sum()
                proxy_diagnostics[3 * index + 1] += ((proxy_weights[index] > 1) & valid).float().sum()
                proxy_diagnostics[3 * index + 2] += proxy_psnrs[index][valid].sum()
            proxy_diagnostics[-1] += cl_valid_fraction * local_batch_size
            diagnostic_steps += 1

        if proxy_diagnostics is not None and (iteration == 1 or iteration % log_every == 0):
            if world_size > 1:
                dist.all_reduce(proxy_diagnostics, op=dist.ReduceOp.SUM)
            if rank == 0:
                report = {}
                for index, name in enumerate(proxy_names):
                    count = float(proxy_diagnostics[3 * index])
                    report[name] = {
                        "count": int(count),
                        "hard_rate": round(float(proxy_diagnostics[3 * index + 1]) / max(count, 1), 4),
                        "psnr": round(float(proxy_diagnostics[3 * index + 2]) / max(count, 1), 4),
                    }
                logger.info("proxy_diagnostics iter=%d proxies=%s cl_valid=%.3f",
                            iteration, report, float(proxy_diagnostics[-1]) / (diagnostic_steps * global_batch_size))
            proxy_diagnostics.zero_()
            diagnostic_steps = 0

        if rank == 0 and (iteration == 1 or iteration % log_every == 0):
            divisor = 1 if iteration == 1 else log_every
            elapsed = time.time() - started
            rate = iteration * world_size / max(elapsed, 1e-6)
            logger.info(
                "iter=%d/%d lr=%.3e total=%.5f rec=%.5f cl=%.5f clip=%.5f dpl=%.5f rank_steps_s=%.3f",
                iteration,
                total_iters,
                optimizer.param_groups[0]["lr"],
                *(running[key] / divisor for key in ("total", "rec", "cl", "clip", "dpl")),
                rate,
            )
            running.clear()

        checkpoint_path = output / "checkpoints" / f"perceiveIR_{iteration:06d}.pth"
        saved_this_step = iteration % save_every == 0 or iteration == total_iters
        if rank == 0 and saved_this_step:
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, iteration, config,
                {
                    "epoch_reference_psnr": epoch_reference_psnr,
                    "probe_hash": probe_hash,
                    "dpl_probe_per_task": probe_count,
                    "epoch_origin_iteration": epoch_origin_iteration,
                    "epoch_index_offset": epoch_index_offset,
                },
            )
            state = {"iteration": iteration, "checkpoint": str(checkpoint_path), "elapsed_seconds": time.time() - started}
            write_json_atomic(output / "state.json", state)
            if retain_last:
                update_last_link(output, checkpoint_path)

        if val_every and iteration % val_every == 0:
            summary = evaluate_validation(
                val_config["manifest"], output, iteration, checkpoint_path if saved_this_step else None,
                unwrap(model), semantic_encoder, device, rank, world_size, update_best=not retain_best,
            )
            if rank == 0:
                if retain_best:
                    best_path = output / "validation" / "best.json"
                    previous = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
                    if previous is None or summary["macro_psnr"] > previous["macro_psnr"]:
                        best_model_path = output / "best_model.pth"
                        save_best_model(best_model_path, model, iteration, config)
                        write_json_atomic(best_path, {**summary, "checkpoint": str(best_model_path)})
                        logger.info("new validation best iter=%d macro_psnr=%.4f", iteration, summary["macro_psnr"])
                logger.info(
                    "val iter=%d crops=%d macro_psnr=%.4f macro_ssim=%.5f task_psnr=%s",
                    iteration, summary["crop_count"], summary["macro_psnr"],
                    summary["macro_ssim"],
                    {task: round(values["psnr"], 4) for task, values in summary["by_task"].items()},
                )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
