"""Evaluate three- or five-task restoration with AdaIR-aligned whole-image protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from .data import index_by_stem, list_images
from .evaluate_stage2 import metrics, write_json_atomic
from .model import DinoSemanticEncoder, PerceiveIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="optional YAML; defaults to the configuration embedded in the checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", help="override the dataset path embedded in the checkpoint")
    parser.add_argument("--dinov2", help="override the local DINOv2 path embedded in the checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-manifest", help="optional; marks test scenes used for checkpoint selection")
    parser.add_argument("--protocol", choices=("adair3", "adair5-common", "adair5"), required=True)
    parser.add_argument("--limit", type=int, default=0, help="smoke run only")
    return parser.parse_args()


def build_samples(root: Path, validation_manifest: Path | None, protocol: str = "adair3") -> list[dict]:
    if protocol not in ("adair3", "adair5-common", "adair5"):
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    test_root = root / "AiO" / "test" if (root / "AiO").is_dir() else root / "test"

    def test_relative_target(path: str | Path) -> str:
        # Validation manifests may have been created under a different AiOIR
        # root. Keep the path below the benchmark's test/ directory stable.
        parts = Path(path).parts
        positions = [index for index, part in enumerate(parts) if part == "test"]
        return "/".join(parts[positions[-1]:]) if positions else str(path)

    validation_targets = set() if validation_manifest is None else {
        (row["task"], test_relative_target(row["target"]))
        for row in json.loads(validation_manifest.read_text(encoding="utf-8"))["records"]
    }
    samples = []
    denoise_targets = list_images(test_root / "denoise" / "bsd68" / "target")
    for sigma in (15, 25, 50):
        for index, target in enumerate(denoise_targets):
            samples.append({
                "task": "denoise", "dataset": f"bsd68_sigma{sigma}", "sigma": sigma,
                "input": None, "target": str(target), "dataset_index": index,
                "used_for_validation": ("denoise", test_relative_target(target)) in validation_targets,
            })
    paired_tasks = [
        ("dehaze", "SOTS-Outdoor", test_root / "dehaze" / "input", test_root / "dehaze" / "target"),
        ("derain", "Rain100L", test_root / "derain" / "Rain100L" / "input",
         test_root / "derain" / "Rain100L" / "target"),
    ]
    if protocol == "adair5":
        paired_tasks.extend((
            ("deblur", "GoPro", test_root / "deblur" / "gopro" / "input",
             test_root / "deblur" / "gopro" / "target"),
            ("lowlight", "LOLv1", test_root / "enhance" / "lol" / "input",
             test_root / "enhance" / "lol" / "target"),
        ))
    for task, name, input_dir, target_dir in paired_tasks:
        targets = index_by_stem(target_dir)
        for input_path in list_images(input_dir):
            stem = input_path.stem.split("_", 1)[0] if task == "dehaze" else input_path.stem
            target = targets.get(stem)
            if target is None:
                raise RuntimeError(f"missing {task} target for {input_path}")
            samples.append({
                "task": task, "dataset": name, "sigma": None,
                "input": str(input_path), "target": str(target), "dataset_index": None,
                "used_for_validation": (task, test_relative_target(target)) in validation_targets,
            })
    for sample in samples:
        sample["id"] = hashlib.sha1(
            f"{sample['task']}|{sample['input']}|{sample['target']}|{sample['sigma']}".encode()
        ).hexdigest()[:20]
    expected = 1930 if protocol == "adair5" else 804
    if len(samples) != expected or len({row["id"] for row in samples}) != expected:
        raise RuntimeError(f"{protocol} expected {expected} unique cases, got {len(samples)}")
    return samples


def crop_to_16(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    crop_h, crop_w = height % 16, width % 16
    top, left = crop_h // 2, crop_w // 2
    return array[top : height - crop_h + top, left : width - crop_w + left, :]


def as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array).copy()).permute(2, 0, 1).float().div_(255.0)


def load_pair(sample: dict, protocol: str, rng: np.random.RandomState) -> tuple[torch.Tensor, torch.Tensor]:
    if protocol not in ("adair3", "adair5-common", "adair5"):
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    with Image.open(sample["target"]) as image:
        target_array = np.asarray(image.convert("RGB"))
    if protocol == "adair3":
        target_array = crop_to_16(target_array)
    target = as_tensor(target_array)
    if sample["input"] is None:
        sigma = int(sample["sigma"])
        if protocol == "adair3":
            # ClearAIR's AdaIR mode seeds NumPy once and advances its RNG in
            # sorted BSD68 order for each of the three noise levels.
            noise = rng.randn(*target_array.shape)
            degraded = np.clip(target_array + noise * sigma, 0, 255).astype(np.uint8)
            low = as_tensor(degraded)
        else:
            # The saved five-task direct-test configuration uses the normal
            # dataset path: seed 3407 + BSD68 index, then uint8 quantization.
            generator = torch.Generator().manual_seed(3407 + int(sample["dataset_index"]))
            noise = torch.randn(target.shape, generator=generator)
            low = (target * 255.0 + noise * sigma).clamp(0, 255).to(torch.uint8).float().div_(255.0)
    else:
        with Image.open(sample["input"]) as image:
            low_array = np.asarray(image.convert("RGB"))
        if protocol == "adair3":
            low_array = crop_to_16(low_array)
        low = as_tensor(low_array)
    if low.shape != target.shape:
        raise RuntimeError(f"test pair dimensions differ: {sample['input']} and {sample['target']}")
    return low, target


@torch.inference_mode()
def restore_direct(model: PerceiveIR, semantics: DinoSemanticEncoder, low: torch.Tensor,
                   device: torch.device = torch.device("cuda:0")) -> torch.Tensor:
    height, width = low.shape[-2:]
    image = low.unsqueeze(0).to(device)
    pad_h, pad_w = (-height) % 8, (-width) % 8
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    features = semantics(image)
    result, _, _ = model(image, features)
    return result[0, :, :height, :width].float().clamp(0, 1).cpu()


def summarize(rows: list[dict], expected: int, iteration: int, protocol: str,
              checkpoint: str, validation_known: bool = True) -> dict:
    def aggregate(items: list[dict]) -> dict:
        return {
            "count": len(items),
            "psnr": float(np.mean([item["psnr"] for item in items])),
            "ssim": float(np.mean([item["ssim"] for item in items])),
            "input_psnr": float(np.mean([item["input_psnr"] for item in items])),
        }

    summary = {}
    unseen = {}
    groups = ("bsd68_sigma15", "bsd68_sigma25", "bsd68_sigma50", "SOTS-Outdoor", "Rain100L")
    if protocol == "adair5":
        groups += ("GoPro", "LOLv1")
    for name in groups:
        group = [row for row in rows if row["dataset"] == name]
        if group:
            summary[name] = aggregate(group)
        if validation_known:
            group_unseen = [row for row in group if not row["used_for_validation"]]
            if group_unseen:
                unseen[name] = aggregate(group_unseen)
    if rows:
        summary["Average"] = aggregate(rows)
        if validation_known:
            unseen_rows = [row for row in rows if not row["used_for_validation"]]
            if unseen_rows:
                unseen["Average"] = aggregate(unseen_rows)
    column_mean = {
        metric: float(np.mean([summary[name][metric] for name in groups]))
        for metric in ("psnr", "ssim")
    } if all(name in summary for name in groups) else None
    return {
        "protocol": protocol, "iteration": iteration, "checkpoint": checkpoint,
        "complete": len(rows) == expected, "count": len(rows), "expected": expected,
        "inference": "FP32, direct single whole-image forward, no tiling; pad right/bottom only when needed",
        "metrics": "mean per-image RGB PSNR/SSIM, no border crop",
        "column_mean": column_mean,
        "summary": summary, "excluding_validation_scenes": unseen if validation_known else None,
        "validation_exclusion_available": validation_known,
    }


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = (yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
              if args.config else state.get("config"))
    if config is None:
        raise RuntimeError("checkpoint contains no config; provide --config")
    iteration = int(state["iteration"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu or install CUDA PyTorch")
    model = PerceiveIR(
        dim=int(config["model"]["dim"]),
        blocks=tuple(config["model"]["blocks"]),
        heads=tuple(config["model"]["heads"]),
        refinement_blocks=int(config["model"]["refinement_blocks"]),
    ).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    semantic_encoder = DinoSemanticEncoder(args.dinov2 or config["pretrained"]["dinov2"]).to(device).eval()
    manifest = Path(args.validation_manifest) if args.validation_manifest else None
    samples = build_samples(Path(args.data_root or config["data"]["root"]), manifest, args.protocol)
    if args.limit:
        samples = samples[:args.limit]
    records_path = output / ("smoke_records.jsonl" if args.limit else "records.jsonl")
    summary_path = output / ("smoke_summary.json" if args.limit else "summary.json")
    existing = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line] if records_path.exists() else []
    if len({row["id"] for row in existing}) != len(existing):
        raise RuntimeError("duplicate test record")
    if any(row["iteration"] != iteration or row["protocol"] != args.protocol for row in existing):
        raise RuntimeError("existing records use a different checkpoint or protocol")
    samples_by_id = {row["id"]: row for row in samples}
    if not {row["id"] for row in existing}.issubset(samples_by_id):
        raise RuntimeError("existing records are not from this test enumeration")
    if any(row["used_for_validation"] != samples_by_id[row["id"]]["used_for_validation"]
           for row in existing):
        raise RuntimeError("existing records used a different validation manifest; choose a new output directory")
    done = {row["id"] for row in existing}
    rng = np.random.RandomState(0)
    started = time.time()
    print(f"protocol={args.protocol} iteration={iteration} expected={len(samples)} resumed={len(done)}", flush=True)
    with records_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            low, target = load_pair(sample, args.protocol, rng)
            if sample["id"] in done:
                continue
            restored = restore_direct(model, semantic_encoder, low, device)
            after, before = metrics(restored, target), metrics(low, target)
            row = {**sample, "iteration": iteration, "protocol": args.protocol,
                   "height": target.shape[1], "width": target.shape[2],
                   "psnr": after["psnr"], "ssim": after["ssim"],
                   "input_psnr": before["psnr"], "input_ssim": before["ssim"]}
            if not all(math.isfinite(row[key]) for key in ("psnr", "ssim", "input_psnr", "input_ssim")):
                raise RuntimeError(f"non-finite metrics on {sample['id']}")
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            existing.append(row)
            if len(existing) % 10 == 0 or len(existing) == len(samples):
                write_json_atomic(summary_path, summarize(existing, len(samples), iteration, args.protocol,
                                                          str(Path(args.checkpoint).resolve()), manifest is not None))
                print(f"done={len(existing)}/{len(samples)} group={sample['dataset']} elapsed_s={time.time()-started:.1f}", flush=True)
    write_json_atomic(summary_path, summarize(existing, len(samples), iteration, args.protocol,
                                              str(Path(args.checkpoint).resolve()), manifest is not None))
    print(f"complete={len(existing) == len(samples)} summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
