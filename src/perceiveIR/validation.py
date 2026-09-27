from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from skimage.metrics import structural_similarity
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from .data import TASKS, index_by_stem, list_images


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha1(value.encode("utf-8")).digest()[:8], "big") % (2**31)


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def positions(size: tuple[int, int], patch: int, identity: str) -> list[tuple[int, int]]:
    width, height = size
    if min(width, height) < patch:
        raise ValueError(f"validation image {identity} is smaller than {patch}: {size}")
    center = ((width - patch) // 2, (height - patch) // 2)
    offset = (
        stable_seed(identity + "|x") % (width - patch + 1),
        stable_seed(identity + "|y") % (height - patch + 1),
    )
    return [center, offset]


def image_tensor(image: Image.Image, x: int, y: int, patch: int) -> torch.Tensor:
    return pil_to_tensor(image.crop((x, y, x + patch, y + patch))).float().div_(255.0)


def make_noisy(clean: torch.Tensor, sigma: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator)
    return (clean * 255.0 + noise * sigma).clamp(0, 255).div(255.0)


def patch_psnr(first: torch.Tensor, second: torch.Tensor) -> float:
    mse = float((first.float().clamp(0, 1) - second.float()).square().mean())
    return -10.0 * math.log10(max(mse, 1e-8))


def patch_ssim(first: torch.Tensor, second: torch.Tensor) -> float:
    first_np = first.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    second_np = second.float().permute(1, 2, 0).cpu().numpy()
    return float(structural_similarity(first_np, second_np, channel_axis=-1, data_range=1.0))


def paired_candidates(input_root: Path, target_root: Path, task: str, patch: int) -> list[dict]:
    targets = index_by_stem(target_root)
    candidates = []
    for input_path in list_images(input_root):
        target_stem = input_path.stem.split("_", 1)[0] if task == "dehaze" else input_path.stem
        target_path = targets.get(target_stem)
        if target_path is None:
            raise RuntimeError(f"missing validation target for {input_path}")
        low, high = open_rgb(input_path), open_rgb(target_path)
        if low.size != high.size:
            raise RuntimeError(f"validation dimensions differ: {input_path} and {target_path}")
        x, y = positions(low.size, patch, str(input_path))[0]
        score = patch_psnr(image_tensor(low, x, y, patch), image_tensor(high, x, y, patch))
        candidates.append({"task": task, "input": str(input_path), "target": str(target_path), "score": score})
    return candidates


def quantile_select(candidates: list[dict], count: int) -> list[dict]:
    ordered = sorted(candidates, key=lambda item: (item["score"], item["input"]))
    chosen = []
    used_targets: set[str] = set()
    used_inputs: set[str] = set()
    for position in np.linspace(0, len(ordered) - 1, count):
        indices = sorted(range(len(ordered)), key=lambda index: (abs(index - position), index))
        for index in indices:
            item = ordered[index]
            if item["target"] not in used_targets and item["input"] not in used_inputs:
                chosen.append(item)
                used_targets.add(item["target"])
                used_inputs.add(item["input"])
                break
        else:
            raise RuntimeError("not enough distinct validation scenes")
    return chosen


def denoise_candidates(test_root: Path, patch: int) -> list[dict]:
    selected = []
    for benchmark in ("bsd68", "kodak24", "urban100"):
        candidates = []
        for path in list_images(test_root / "denoise" / benchmark / "target"):
            image = open_rgb(path)
            x, y = positions(image.size, patch, str(path))[0]
            crop = image_tensor(image, x, y, patch)
            texture = float((crop[:, :, 1:] - crop[:, :, :-1]).abs().mean())
            texture += float((crop[:, 1:, :] - crop[:, :-1, :]).abs().mean())
            candidates.append((texture, str(path)))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) < 4:
            raise RuntimeError(f"not enough {benchmark} denoising images")
        for position in np.linspace(0, len(candidates) - 1, 4):
            _, path = candidates[round(float(position))]
            for sigma in (15, 25, 50):
                selected.append({
                    "task": "denoise", "benchmark": benchmark, "input": None,
                    "target": path, "sigma": sigma, "stratum": f"{benchmark}|sigma{sigma}",
                })
    return selected


def build_manifest(config: dict) -> dict:
    patch = int(config["data"]["patch_size"])
    tasks = tuple(config["data"].get("tasks", TASKS))
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError(f"invalid validation tasks: {tasks}")
    root = Path(config["data"]["root"])
    if (root / "AiO").is_dir():
        root /= "AiO"
    test_root = root / "test"
    selected = denoise_candidates(test_root, patch) if "denoise" in tasks else []
    for task, benchmark, count, input_root, target_root in (
        ("dehaze", "SOTS", 24, test_root / "dehaze" / "input", test_root / "dehaze" / "target"),
        ("derain", "Rain100L", 24, test_root / "derain" / "Rain100L" / "input", test_root / "derain" / "Rain100L" / "target"),
        ("deblur", "GoPro", 24, test_root / "deblur" / "gopro" / "input", test_root / "deblur" / "gopro" / "target"),
        ("lowlight", "LOLv1", 5, test_root / "enhance" / "lol" / "input", test_root / "enhance" / "lol" / "target"),
    ):
        if task not in tasks:
            continue
        candidates = paired_candidates(input_root, target_root, task, patch)
        for index, item in enumerate(quantile_select(candidates, count)):
            selected.append({
                "task": task, "benchmark": benchmark,
                "input": item["input"], "target": item["target"],
                "stratum": f"severity_q{index * 4 // count + 1}", "selection_psnr": item["score"],
            })
    records = []
    for image_index, item in enumerate(selected):
        image = open_rgb(item["target"])
        for crop_index, (x, y) in enumerate(positions(image.size, patch, item["input"] or item["target"])):
            record = dict(item)
            record.update({
                "id": f"{item['task']}_{image_index:03d}_{crop_index}",
                "crop": [x, y, patch],
                "noise_seed": stable_seed(f"{item['target']}|{item.get('sigma')}|{crop_index}"),
            })
            records.append(record)
    return {
        "selection": "deterministic texture quantiles for three denoise benchmarks and all three noise levels; distinct-scene degradation-PSNR quantiles for paired tasks; center and hash-position 256 crops",
        "purpose": "checkpoint validation only; remaining test images are reserved for later evaluation",
        "patch_size": patch,
        "image_count": len(selected),
        "crop_count": len(records),
        "task_counts": dict(Counter(item["task"] for item in selected)),
        "records": records,
    }


def load_pair(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, patch = record["crop"]
    high = image_tensor(open_rgb(record["target"]), x, y, patch)
    if record["input"] is None:
        low = make_noisy(high, int(record["sigma"]), int(record["noise_seed"]))
    else:
        low = image_tensor(open_rgb(record["input"]), x, y, patch)
    return low, high


def evaluate(
    manifest_path: str | Path,
    output: Path,
    iteration: int,
    checkpoint: Path | None,
    model: torch.nn.Module,
    semantic_encoder: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    update_best: bool = True,
) -> dict | None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["patch_size"] != 256:
        raise RuntimeError("the fixed validation manifest does not match the 256-pixel experiment")
    local_records = []
    model.eval()
    semantic_encoder.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for record in manifest["records"][rank::world_size]:
            low_cpu, high_cpu = load_pair(record)
            low = low_cpu.unsqueeze(0).to(device)
            semantics = semantic_encoder(low)
            restored, _, _ = model(low, semantics)
            restored_cpu = restored[0].float().clamp(0, 1).cpu()
            local_records.append({
                "id": record["id"], "task": record["task"], "benchmark": record["benchmark"],
                "stratum": record["stratum"],
                "psnr": patch_psnr(restored_cpu, high_cpu),
                "ssim": patch_ssim(restored_cpu, high_cpu),
                "input_psnr": patch_psnr(low_cpu, high_cpu),
            })
    model.train()
    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, local_records)
    else:
        gathered = [local_records]
    if rank != 0:
        return None
    rows = sorted((row for part in gathered for row in part), key=lambda row: row["id"])
    if len(rows) != manifest["crop_count"]:
        raise RuntimeError("validation did not cover every fixed crop")
    by_task = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        by_task[task] = {
            "count": len(task_rows),
            "psnr": float(np.mean([row["psnr"] for row in task_rows])),
            "ssim": float(np.mean([row["ssim"] for row in task_rows])),
            "input_psnr": float(np.mean([row["input_psnr"] for row in task_rows])),
        }
    by_group = {}
    for group in sorted({(row["task"], row["stratum"]) for row in rows}):
        group_rows = [row for row in rows if (row["task"], row["stratum"]) == group]
        by_group["|".join(group)] = {
            "count": len(group_rows),
            "psnr": float(np.mean([row["psnr"] for row in group_rows])),
            "ssim": float(np.mean([row["ssim"] for row in group_rows])),
        }
    summary = {
        "iteration": iteration, "checkpoint": str(checkpoint) if checkpoint is not None else None, "crop_count": len(rows),
        "macro_psnr": float(np.mean([item["psnr"] for item in by_task.values()])),
        "macro_ssim": float(np.mean([item["ssim"] for item in by_task.values()])),
        "by_task": by_task, "by_group": by_group,
    }
    val_dir = output / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / f"iter_{iteration:06d}.json").write_text(
        json.dumps({"summary": summary, "records": rows}, indent=2), encoding="utf-8"
    )
    with (val_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")
    if update_best:
        best_path = val_dir / "best.json"
        previous = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
        if previous is None or summary["macro_psnr"] > previous["macro_psnr"]:
            best_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed three-task restoration validation subset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="run a single-process validation of an existing checkpoint")
    parser.add_argument("--output", help="output directory for standalone validation")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    path = Path(config["validation"]["manifest"])
    if args.checkpoint:
        from .model import DinoSemanticEncoder, PerceiveIR

        device = torch.device("cuda:0")
        model = PerceiveIR(
            dim=int(config["model"]["dim"]),
            blocks=tuple(config["model"]["blocks"]),
            heads=tuple(config["model"]["heads"]),
            refinement_blocks=int(config["model"]["refinement_blocks"]),
        ).to(device)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        semantic_encoder = DinoSemanticEncoder(config["pretrained"]["dinov2"]).to(device)
        output = Path(args.output or config["output"])
        result = evaluate(path, output, int(state["iteration"]), Path(args.checkpoint),
                          model, semantic_encoder, device, 0, 1)
        print(json.dumps(result))
        return
    if path.exists():
        raise FileExistsError(f"validation manifest already exists: {path}")
    manifest = build_manifest(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({"manifest": str(path), "image_count": manifest["image_count"],
                      "crop_count": manifest["crop_count"], "task_counts": manifest["task_counts"]}))


if __name__ == "__main__":
    main()
