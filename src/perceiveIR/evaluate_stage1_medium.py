from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import lpips
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .render_stage1_medium import tiled_forward
from .stage1 import RestormerMedium
from .stage1_data import build_source_samples, load_source_pair, sample_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit held-out medium-quality candidates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout-fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--preview-per-group", type=int, default=2)
    parser.add_argument("--lpips-max-side", type=int, default=512)
    return parser.parse_args()


def identity(sample) -> str:
    return f"{sample.task}|{sample.lq_path or sample.gt_path}"


def select_samples(samples, count: int, seed: int):
    grouped = defaultdict(list)
    for sample in samples:
        key = hashlib.sha1(f"{seed}|{identity(sample)}".encode("utf-8")).hexdigest()
        grouped[sample.task].append((key, sample))
    selected = []
    for task in sorted(grouped):
        ordered = [sample for _, sample in sorted(grouped[task], key=lambda item: item[0])]
        selected.extend(ordered if count <= 0 else ordered[:count])
    return selected


def deterministic_pair(sample):
    digest = hashlib.sha1(identity(sample).encode("utf-8")).hexdigest()[:16]
    noise_seed = int(digest, 16) % (2**31) if sample.lq_path is None else None
    return load_source_pair(sample, deterministic_noise_seed=noise_seed)


def image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return pil_to_tensor(image).float().div_(255.0).unsqueeze(0).to(device)


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-12))


def ssim(image: torch.Tensor, target: torch.Tensor) -> float:
    image_np = image[0].permute(1, 2, 0).float().cpu().numpy()
    target_np = target[0].permute(1, 2, 0).float().cpu().numpy()
    return float(structural_similarity(image_np, target_np, channel_axis=-1, data_range=1.0))


def resize_for_lpips(tensor: torch.Tensor, max_side: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    scale = min(1.0, max_side / max(height, width))
    scale = max(scale, 64.0 / min(height, width))
    if abs(scale - 1.0) < 1e-6:
        return tensor
    size = (max(64, round(height * scale)), max(64, round(width * scale)))
    return torch.nn.functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


@torch.inference_mode()
def lpips_distance(metric, first: torch.Tensor, second: torch.Tensor, max_side: int) -> float:
    first = resize_for_lpips(first, max_side).mul(2).sub(1)
    second = resize_for_lpips(second, max_side).mul(2).sub(1)
    return float(metric(first, second).mean().item())


@torch.inference_mode()
def restore(model, low: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=low.is_cuda):
        return tiled_forward(model, low, args.tile_size, args.tile_overlap).clamp(0, 1)


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (28, 28, 28))
    panel.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return panel


def save_contact_sheet(rows, output: Path, cell_size: int = 256) -> None:
    if not rows:
        return
    header = 26
    row_label = 22
    canvas = Image.new("RGB", (cell_size * 3, header + len(rows) * (cell_size + row_label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Low / degraded", "Medium candidate", "High / GT")):
        draw.text((column * cell_size + 6, 6), label, fill="black")
    for row_index, (label, low, medium, high) in enumerate(rows):
        y = header + row_index * (cell_size + row_label)
        draw.text((5, y + 3), label, fill="black")
        image_y = y + row_label
        for column, image in enumerate((low, medium, high)):
            canvas.paste(fit_panel(image, (cell_size, cell_size)), (column * cell_size, image_y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, subsampling=0)


def summarize(records: list[dict]) -> dict:
    result = {}
    for task in sorted({record["task"] for record in records}) + ["all"]:
        rows = records if task == "all" else [record for record in records if record["task"] == task]
        result[task] = {
            "count": len(rows),
            "psnr_improved_rate": float(np.mean([row["delta_psnr"] > 0 for row in rows])),
            "ssim_improved_rate": float(np.mean([row["delta_ssim"] > 0 for row in rows])),
            "lpips_improved_rate": float(np.mean([row["lpips_gain"] > 0 for row in rows])),
            "delta_psnr_mean": float(np.mean([row["delta_psnr"] for row in rows])),
            "delta_ssim_mean": float(np.mean([row["delta_ssim"] for row in rows])),
            "lpips_gain_mean": float(np.mean([row["lpips_gain"] for row in rows])),
            "progress_median": float(np.median([row["progress"] for row in rows])),
            "progress_q10": float(np.quantile([row["progress"] for row in rows], 0.1)),
            "progress_q90": float(np.quantile([row["progress"] for row in rows], 0.9)),
        }
    return result


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = int(state["iteration"])
    output = Path(args.output_root) / f"fold_{args.heldout_fold}" / f"step_{step:06d}"
    output.mkdir(parents=True, exist_ok=True)

    model = RestormerMedium(
        dim=config["model"]["dim"],
        blocks=config["model"]["blocks"],
        heads=config["model"]["heads"],
        refinement_blocks=config["model"]["refinement_blocks"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    perceptual.requires_grad_(False)

    heldout = [
        sample
        for sample in build_source_samples(config["data"]["root"])
        if sample_fold(sample) == args.heldout_fold
    ]
    samples = select_samples(heldout, args.samples_per_task, args.seed)
    sample_by_id = {identity(sample): sample for sample in samples}
    records = []
    metrics_path = output / "metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            low_image, high_image = deterministic_pair(sample)
            if low_image.size != high_image.size:
                raise RuntimeError(f"image-size mismatch for {identity(sample)}")
            low = image_tensor(low_image, device)
            high = image_tensor(high_image, device)
            medium = restore(model, low, args)
            mse_low = float(torch.mean((low - high) ** 2).item())
            mse_medium = float(torch.mean((medium - high) ** 2).item())
            low_lpips = lpips_distance(perceptual, low, high, args.lpips_max_side)
            medium_lpips = lpips_distance(perceptual, medium, high, args.lpips_max_side)
            record = {
                "identity": identity(sample),
                "task": sample.task,
                "fold": args.heldout_fold,
                "step": step,
                "width": low_image.width,
                "height": low_image.height,
                "low": str(sample.lq_path) if sample.lq_path is not None else None,
                "high": str(sample.gt_path),
                "mse_low": mse_low,
                "mse_medium": mse_medium,
                "psnr_low": psnr_from_mse(mse_low),
                "psnr_medium": psnr_from_mse(mse_medium),
                "ssim_low": ssim(low, high),
                "ssim_medium": ssim(medium, high),
                "lpips_low": low_lpips,
                "lpips_medium": medium_lpips,
                "lpips_low_medium": lpips_distance(perceptual, low, medium, args.lpips_max_side),
                "progress": 1.0 - mse_medium / max(mse_low, 1e-12),
            }
            record["delta_psnr"] = record["psnr_medium"] - record["psnr_low"]
            record["delta_ssim"] = record["ssim_medium"] - record["ssim_low"]
            record["lpips_gain"] = record["lpips_low"] - record["lpips_medium"]
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index == 1 or index % 25 == 0 or index == len(samples):
                print(
                    f"fold={args.heldout_fold} step={step} audited={index}/{len(samples)} "
                    f"task={sample.task} progress={record['progress']:.4f}",
                    flush=True,
                )

    preview_root = output / "visuals"
    per_group = max(0, args.preview_per_group)
    for task in sorted({record["task"] for record in records}):
        task_records = [record for record in records if record["task"] == task]
        worst = sorted(task_records, key=lambda row: row["progress"])[:per_group]
        best = sorted(task_records, key=lambda row: row["progress"], reverse=True)[:per_group]
        random_rows = sorted(
            task_records,
            key=lambda row: hashlib.sha1(f"preview|{args.seed}|{row['identity']}".encode()).hexdigest(),
        )[:per_group]
        chosen = [("worst", row) for row in worst]
        chosen += [("random", row) for row in random_rows]
        chosen += [("closest-GT", row) for row in best]
        panels = []
        for category, record in chosen:
            sample = sample_by_id[record["identity"]]
            low_image, high_image = deterministic_pair(sample)
            medium = restore(model, image_tensor(low_image, device), args)
            label = (
                f"{category} r={record['progress']:.3f} "
                f"dPSNR={record['delta_psnr']:.2f}dB"
            )
            panels.append((label, low_image, to_pil_image(medium[0].float().cpu()), high_image))
        save_contact_sheet(panels, preview_root / f"{task}.jpg")

    report = {
        "fold": args.heldout_fold,
        "step": step,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "samples_per_task_requested": args.samples_per_task,
        "summary": summarize(records),
    }
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"completed fold={args.heldout_fold} step={step} output={output}", flush=True)


if __name__ == "__main__":
    main()
