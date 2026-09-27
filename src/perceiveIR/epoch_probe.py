from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from .data import Sample
from .validation import make_noisy, open_rgb, stable_seed


def _selection_key(sample: Sample, seed: int) -> str:
    identity = f"{seed}|{sample.task}|{sample.lq_path}|{sample.gt_path}|{sample.noise_sigma}"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def _record(sample: Sample, patch: int, seed: int) -> dict:
    target = open_rgb(sample.gt_path)
    width, height = target.size
    if min(width, height) < patch:
        scale = patch / min(width, height)
        width, height = max(patch, round(width * scale)), max(patch, round(height * scale))
    identity = _selection_key(sample, seed)
    return {
        "task": sample.task,
        "input": str(sample.lq_path) if sample.lq_path else None,
        "target": str(sample.gt_path),
        "sigma": sample.noise_sigma,
        "resize": [width, height],
        "crop": [stable_seed(identity + "|x") % (width - patch + 1),
                 stable_seed(identity + "|y") % (height - patch + 1), patch],
        "noise_seed": stable_seed(identity + "|noise"),
    }


def load_pair(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    high_image = open_rgb(record["target"])
    size = tuple(record["resize"])
    if high_image.size != size:
        high_image = high_image.resize(size, Image.Resampling.BICUBIC)
    x, y, patch = record["crop"]
    box = (x, y, x + patch, y + patch)
    high = pil_to_tensor(high_image.crop(box)).float().div_(255.0)
    if record["input"] is None:
        low = make_noisy(high, int(record["sigma"]), int(record["noise_seed"]))
    else:
        low_image = open_rgb(record["input"])
        if low_image.size != size:
            low_image = low_image.resize(size, Image.Resampling.BICUBIC)
        low = pil_to_tensor(low_image.crop(box)).float().div_(255.0)
    return low, high


def build_manifest(samples: Sequence[Sample], patch: int, per_task: int, seed: int) -> dict:
    if per_task < 3 or per_task % 3:
        raise ValueError("dpl_probe_per_task must be a positive multiple of three")
    unique = {(sample.task, sample.lq_path, sample.gt_path, sample.noise_sigma): sample for sample in samples}
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in unique.values():
        groups[sample.task].append(sample)
    records: list[dict] = []
    for task in sorted(groups):
        candidates = sorted(groups[task], key=lambda sample: _selection_key(sample, seed))
        if task == "denoise":
            for sigma in (15, 25, 50):
                chosen = [sample for sample in candidates if sample.noise_sigma == sigma][:per_task // 3]
                if len(chosen) != per_task // 3:
                    raise RuntimeError(f"insufficient denoise sigma={sigma} training probe samples")
                records.extend(_record(sample, patch, seed) for sample in chosen)
            continue
        distinct = []
        seen_targets: set[Path] = set()
        for sample in candidates:
            if sample.gt_path not in seen_targets:
                distinct.append(sample)
                seen_targets.add(sample.gt_path)
            if len(distinct) >= per_task * 4:
                break
        if len(distinct) < per_task:
            raise RuntimeError(f"insufficient distinct {task} training probe scenes")
        scored = []
        for sample in distinct:
            record = _record(sample, patch, seed)
            low, high = load_pair(record)
            mse = float((low - high).square().mean())
            scored.append((-10 * math.log10(max(mse, 1e-8)), record))
        scored.sort(key=lambda item: item[0])
        records.extend(scored[round(float(index))][1]
                       for index in np.linspace(0, len(scored) - 1, per_task))
    return {
        "purpose": "fixed training-only model-PSNR probe for DPL epoch weighting; not validation",
        "patch_size": patch,
        "per_task": per_task,
        "seed": seed,
        "records": records,
    }


def ensure_manifest(path: Path, samples: Sequence[Sample], patch: int, per_task: int, seed: int) -> dict:
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest["patch_size"], manifest["per_task"], manifest["seed"]) != (patch, per_task, seed):
            raise RuntimeError(f"training PSNR probe configuration differs: {path}")
        return manifest
    manifest = build_manifest(samples, patch, per_task, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
    return manifest


def evaluate_model_psnr(
    manifest: dict, model: torch.nn.Module, semantic_encoder: torch.nn.Module,
    device: torch.device, rank: int, world_size: int, amp_dtype: torch.dtype,
) -> float:
    was_training = model.training
    model.eval()
    score = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        for record in manifest["records"][rank::world_size]:
            low_cpu, high_cpu = load_pair(record)
            low = low_cpu.unsqueeze(0).to(device)
            semantics = semantic_encoder(low)
            restored, _, _ = model(low, semantics)
            mse = (restored[0].float().clamp(0, 1).cpu() - high_cpu).square().mean().item()
            score[0] += -10.0 * math.log10(max(mse, 1e-8))
            score[1] += 1
    if world_size > 1:
        dist.all_reduce(score, op=dist.ReduceOp.SUM)
    model.train(was_training)
    return float(score[0] / score[1])
