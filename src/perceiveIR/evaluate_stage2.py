from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.metrics import structural_similarity
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from .data import index_by_stem, list_images
from .model import DinoSemanticEncoder, PerceiveIR
from .validation import stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-image three-task perceiveIR test evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="first N samples, only for a smoke run")
    return parser.parse_args()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div_(255.0)


def make_noisy(clean: torch.Tensor, sigma: int, identity: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(stable_seed("full_test|" + identity))
    noise = torch.randn(clean.shape, generator=generator)
    return (clean * 255.0 + noise * sigma).clamp(0, 255).div(255.0)


def build_samples(test_root: Path, validation_manifest: Path) -> list[dict]:
    selected = json.loads(validation_manifest.read_text(encoding="utf-8"))["records"]
    validation_targets = {
        (item["task"], item["target"]) for item in selected
    }
    samples = []
    for benchmark in ("bsd68", "kodak24", "urban100"):
        for target in list_images(test_root / "denoise" / benchmark / "target"):
            for sigma in (15, 25, 50):
                samples.append({
                    "task": "denoise", "benchmark": benchmark, "sigma": sigma,
                    "input": None, "target": str(target),
                    "group": f"{benchmark}|sigma{sigma}",
                    "used_for_validation": ("denoise", str(target)) in validation_targets,
                })
    for task, benchmark, input_root, target_root in (
        ("dehaze", "SOTS", test_root / "dehaze" / "input", test_root / "dehaze" / "target"),
        ("derain", "Rain100L", test_root / "derain" / "Rain100L" / "input",
         test_root / "derain" / "Rain100L" / "target"),
    ):
        targets = index_by_stem(target_root)
        for input_path in list_images(input_root):
            target_stem = input_path.stem.split("_", 1)[0] if task == "dehaze" else input_path.stem
            target = targets.get(target_stem)
            if target is None:
                raise RuntimeError(f"missing target for {input_path}")
            samples.append({
                "task": task, "benchmark": benchmark, "sigma": None,
                "input": str(input_path), "target": str(target), "group": benchmark,
                "used_for_validation": (task, str(target)) in validation_targets,
            })
    for sample in samples:
        sample["id"] = hashlib.sha1(
            f"{sample['task']}|{sample['input']}|{sample['target']}|{sample['sigma']}".encode()
        ).hexdigest()[:20]
    # Spread the first few reports across all benchmarks and noise levels.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["group"]].append(sample)
    for group in grouped:
        grouped[group].sort(key=lambda item: (item["used_for_validation"], item["id"]))
    ordered = []
    for index in range(max(map(len, grouped.values()))):
        ordered.extend(rows[index] for rows in grouped.values() if index < len(rows))
    if len({sample["id"] for sample in ordered}) != len(ordered):
        raise RuntimeError("duplicate test sample identities")
    return ordered


@torch.inference_mode()
def restore_tiled(
    model: PerceiveIR,
    semantic_encoder: DinoSemanticEncoder,
    low: torch.Tensor,
    device: torch.device,
    tile_size: int,
    overlap: int,
) -> torch.Tensor:
    height, width = low.shape[-2:]
    stride = tile_size - overlap
    output = torch.zeros_like(low)
    weight = torch.zeros((1, height, width), dtype=low.dtype)

    def positions(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        result = list(range(0, length - tile_size + 1, stride))
        if result[-1] != length - tile_size:
            result.append(length - tile_size)
        return result

    for top in positions(height):
        for left in positions(width):
            tile = low[:, top : top + tile_size, left : left + tile_size].unsqueeze(0).to(device)
            tile_height, tile_width = tile.shape[-2:]
            pad_h, pad_w = (-tile_height) % 8, (-tile_width) % 8
            if pad_h or pad_w:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                semantics = semantic_encoder(tile)
                restored, _, _ = model(tile, semantics)
            prediction = restored[0, :, :tile_height, :tile_width].float().clamp(0, 1).cpu()
            output[:, top : top + tile_height, left : left + tile_width] += prediction
            weight[:, top : top + tile_height, left : left + tile_width] += 1
    return output / weight.clamp_min(1)


def metrics(image: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = float((image.float().clamp(0, 1) - target.float()).square().mean())
    first = image.permute(1, 2, 0).numpy()
    second = target.permute(1, 2, 0).numpy()
    return {
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "ssim": float(structural_similarity(first, second, channel_axis=-1, data_range=1.0)),
    }


def summarize(rows: list[dict], expected: int, iteration: int, checkpoint: str) -> dict:
    result = {
        "iteration": iteration, "checkpoint": checkpoint,
        "count": len(rows), "expected": expected, "complete": len(rows) == expected,
        "metric": "mean per full RGB image; unclipped border; float PSNR/SSIM; BF16 tiled inference",
        "by_group": {}, "by_task": {}, "unseen_by_group": {}, "unseen_by_task": {},
    }

    def aggregate(items: list[dict]) -> dict:
        return {
            "count": len(items),
            "psnr": float(np.mean([item["psnr"] for item in items])),
            "ssim": float(np.mean([item["ssim"] for item in items])),
            "input_psnr": float(np.mean([item["input_psnr"] for item in items])),
            "input_ssim": float(np.mean([item["input_ssim"] for item in items])),
        }

    for scope, candidates in (("", rows), ("unseen_", [row for row in rows if not row["used_for_validation"]])):
        for group in sorted({row["group"] for row in candidates}):
            result[scope + "by_group"][group] = aggregate([row for row in candidates if row["group"] == group])
        for task in ("denoise", "dehaze", "derain"):
            chosen = [row for row in candidates if row["task"] == task]
            if chosen:
                result[scope + "by_task"][task] = aggregate(chosen)
        if len(result[scope + "by_task"]) == 3:
            result[scope + "macro_psnr"] = float(np.mean(
                [value["psnr"] for value in result[scope + "by_task"].values()]
            ))
    return result


def main() -> None:
    args = parse_args()
    if args.tile_size < 16 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("tile size must be >=16 and overlap must be smaller")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    iteration = int(checkpoint["iteration"])
    model = PerceiveIR(
        dim=int(config["model"]["dim"]),
        blocks=tuple(config["model"]["blocks"]),
        heads=tuple(config["model"]["heads"]),
        refinement_blocks=int(config["model"]["refinement_blocks"]),
    ).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    semantic_encoder = DinoSemanticEncoder(config["pretrained"]["dinov2"]).cuda().eval()
    device = torch.device("cuda:0")
    root = Path(config["data"]["root"])
    if (root / "AiO").is_dir():
        root /= "AiO"
    samples = build_samples(root / "test", Path(args.validation_manifest))
    if args.limit:
        samples = samples[:args.limit]
    rows_path = output / ("smoke_records.jsonl" if args.limit else "records.jsonl")
    summary_path = output / ("smoke_summary.json" if args.limit else "summary.json")
    existing = []
    if rows_path.exists():
        existing = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
        if any(row["iteration"] != iteration for row in existing):
            raise RuntimeError("existing records belong to a different checkpoint")
    done = {row["id"] for row in existing}
    if len(done) != len(existing):
        raise RuntimeError("duplicate recorded test sample")
    print(f"iteration={iteration} test_samples={len(samples)} already_done={len(done)} tile={args.tile_size}", flush=True)
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            if sample["id"] in done:
                continue
            target = image_tensor(Path(sample["target"]))
            if sample["input"] is None:
                low = make_noisy(target, int(sample["sigma"]), sample["target"] + "|" + str(sample["sigma"]))
            else:
                low = image_tensor(Path(sample["input"]))
            if low.shape != target.shape:
                raise RuntimeError(f"test pair sizes differ: {sample['input']} and {sample['target']}")
            restored = restore_tiled(model, semantic_encoder, low, device, args.tile_size, args.overlap)
            result = {**sample, "iteration": iteration, "height": target.shape[1], "width": target.shape[2]}
            result.update(metrics(restored, target))
            before = metrics(low, target)
            result["input_psnr"], result["input_ssim"] = before["psnr"], before["ssim"]
            if not all(math.isfinite(result[key]) for key in ("psnr", "ssim", "input_psnr", "input_ssim")):
                raise RuntimeError(f"non-finite test metric: {sample['id']}")
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            existing.append(result)
            if len(existing) % 10 == 0 or len(existing) == len(samples):
                write_json_atomic(summary_path, summarize(existing, len(samples), iteration, str(checkpoint_path)))
                print(f"done={len(existing)}/{len(samples)} group={sample['group']} "
                      f"psnr={result['psnr']:.3f} elapsed_s={time.time()-started:.1f}", flush=True)
    write_json_atomic(summary_path, summarize(existing, len(samples), iteration, str(checkpoint_path)))
    print(f"complete={len(existing) == len(samples)} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
