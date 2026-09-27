from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from .evaluate_stage1_prompts import (
    QUALITY_NAMES,
    TASKS,
    crop_positions,
    make_summary,
    prepare_common_images,
)
from .stage1 import FrozenCLIPImageEncoder, FrozenCLIPTextEncoder, QualityPromptLearner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen quality-prompt checkpoints using cached CLIP image features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-crops", type=int, choices=(1, 5), default=5)
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or any(record["task"] not in TASKS for record in records):
        raise RuntimeError("audit records are empty or contain an unknown task")
    identities = [record["identity"] for record in records]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate test identities")
    return records


def feature_cache_key(records: list[dict], num_crops: int) -> str:
    identities = "\n".join(record["identity"] for record in records)
    return hashlib.sha256(f"CLIP-ViT-B32|{num_crops}|{identities}".encode()).hexdigest()


@torch.inference_mode()
def encode_records(
    records: list[dict],
    image_encoder: FrozenCLIPImageEncoder,
    device: torch.device,
    num_crops: int,
    cache_path: Path,
) -> torch.Tensor:
    cache_key = feature_cache_key(records, num_crops)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("key") != cache_key or tuple(cached["features"].shape) != (len(records), 3, 512):
            raise RuntimeError(f"stale or malformed feature cache: {cache_path}")
        print(f"reused_features={cache_path} shape={tuple(cached['features'].shape)}", flush=True)
        return cached["features"].float()
    features = []
    for index, record in enumerate(records, start=1):
        images = [
            Image.open(record[key]).convert("RGB")
            for key in ("low_path", "medium_path", "high_path")
        ]
        images = prepare_common_images(images)
        width, height = images[0].size
        positions = crop_positions(width, height, 224, num_crops)
        crops = [
            pil_to_tensor(image.crop((left, top, left + 224, top + 224))).float().div_(255.0)
            for image in images for left, top in positions
        ]
        batch = torch.stack(crops).to(device)
        feature = image_encoder(batch).reshape(3, len(positions), -1).mean(dim=1)
        features.append(feature.cpu())
        if index == 1 or index % 100 == 0 or index == len(records):
            print(f"encoded={index}/{len(records)}", flush=True)
    result = torch.stack(features).float()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save({"key": cache_key, "features": result}, temporary)
    temporary.replace(cache_path)
    return result


def split_records(records: list[dict]) -> tuple[list[int], list[int]]:
    by_task = {task: [] for task in TASKS}
    for index, record in enumerate(records):
        by_task[record["task"]].append(index)
    screen, holdout = [], []
    for task in TASKS:
        ranked = sorted(
            by_task[task],
            key=lambda index: hashlib.sha256(
                f"perceiveIR-checkpoint-screen|{records[index]['identity']}".encode()
            ).hexdigest(),
        )
        midpoint = (len(ranked) + 1) // 2
        screen.extend(ranked[:midpoint])
        holdout.extend(ranked[midpoint:])
    return sorted(screen), sorted(holdout)


def summary_for(
    source_records: list[dict],
    features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    indices: list[int],
) -> dict:
    selected = features[indices]
    logits = logit_scale * selected @ text_features.t()
    probs = logits.softmax(dim=-1)
    predictions = probs.argmax(dim=-1).numpy()
    scores = (probs @ torch.arange(3, dtype=probs.dtype)).numpy()
    assessed = [
        {
            "task": source_records[index]["task"],
            "predictions": predictions[position].tolist(),
            "scores": scores[position].tolist(),
        }
        for position, index in enumerate(indices)
    ]
    return make_summary(assessed, Counter(record["task"] for record in assessed))


def rank_key(summary: dict) -> tuple:
    macro = summary["task_macro"]
    minimum = summary["task_minimum"]
    return (
        int(summary["suggested_gate"]["passed"]),
        minimum["monotonic_ordering_rate"],
        macro["strict_triplet_rate"],
        macro["image_accuracy"],
        minimum["image_accuracy"],
    )


def short_metrics(summary: dict) -> dict:
    return {
        "pass": summary["suggested_gate"]["passed"],
        "macro_image": summary["task_macro"]["image_accuracy"],
        "min_image": summary["task_minimum"]["image_accuracy"],
        "macro_strict": summary["task_macro"]["strict_triplet_rate"],
        "overall_monotonic": summary["overall"]["monotonic_ordering_rate"],
        "min_monotonic": summary["task_minimum"]["monotonic_ordering_rate"],
        "derain_strict": summary["by_task"]["derain"]["strict_triplet_rate"],
        "derain_monotonic": summary["by_task"]["derain"]["monotonic_ordering_rate"],
        "deblur_medium_recall": summary["by_task"]["deblur"]["per_quality_accuracy"]["mediocre"],
    }


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = read_records(Path(args.records))
    checkpoints = sorted(Path(args.checkpoint_dir).glob("quality_prompts_*.pth"))
    if not checkpoints:
        raise RuntimeError("no prompt checkpoints")
    device = torch.device(args.device)

    import clip

    clip_model, _ = clip.load(
        "ViT-B/32", device="cpu", jit=False, download_root=config["pretrained"]["clip_cache"]
    )
    clip_model.float().requires_grad_(False).eval().to(device)
    image_encoder = FrozenCLIPImageEncoder(clip_model).to(device).eval()
    feature_path = Path(args.feature_cache) if args.feature_cache else output_root / f"image_features_{args.num_crops}crops.pt"
    features = encode_records(records, image_encoder, device, args.num_crops, feature_path)
    image_encoder.cpu()
    del image_encoder

    learner = QualityPromptLearner(clip_model, context_length=int(config["prompt"]["context_length"])).to(device).eval()
    text_encoder = FrozenCLIPTextEncoder(clip_model).to(device).eval()
    logit_scale = clip_model.logit_scale.exp().detach().cpu().float()
    screen_indices, holdout_indices = split_records(records)
    all_indices = list(range(len(records)))
    rows = []
    summaries = {}
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if tuple(state["quality_names"]) != QUALITY_NAMES:
            raise RuntimeError(f"unexpected quality names in {checkpoint}")
        learner.load_state_dict(state["prompt_learner"])
        with torch.inference_mode():
            text_features = F.normalize(text_encoder(learner(), learner.tokenized_prompts), dim=-1).cpu().float()
        screen = summary_for(records, features, text_features, logit_scale, screen_indices)
        full = summary_for(records, features, text_features, logit_scale, all_indices)
        step = int(state["iteration"])
        row = {"step": step, "checkpoint": str(checkpoint.resolve())}
        row.update({f"screen_{key}": value for key, value in short_metrics(screen).items()})
        row.update({f"full_{key}": value for key, value in short_metrics(full).items()})
        rows.append(row)
        summaries[step] = {"screen": screen, "full": full}
        print(
            f"step={step} screen={short_metrics(screen)} full={short_metrics(full)}",
            flush=True,
        )
    rows.sort(key=lambda row: row["step"])
    selected = max(rows, key=lambda row: rank_key(summaries[row["step"]]["screen"]))
    selected_step = selected["step"]
    selected_state = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    learner.load_state_dict(selected_state["prompt_learner"])
    with torch.inference_mode():
        selected_text = F.normalize(text_encoder(learner(), learner.tokenized_prompts), dim=-1).cpu().float()
    holdout = summary_for(records, features, selected_text, logit_scale, holdout_indices)
    report = {
        "selected_checkpoint": selected["checkpoint"],
        "selected_step": selected_step,
        "selection_basis": "deterministic half of each task, selected by gate then minimum monotonic and task-macro strict triplet rate",
        "screen_counts": dict(Counter(records[index]["task"] for index in screen_indices)),
        "holdout_counts": dict(Counter(records[index]["task"] for index in holdout_indices)),
        "screen": summaries[selected_step]["screen"],
        "holdout": holdout,
        "full": summaries[selected_step]["full"],
        "any_full_gate_pass": any(row["full_pass"] for row in rows),
        "caveat": "The 100K model was previously evaluated on all AiO test images; this holdout is only for comparing checkpoints, not a pristine final benchmark.",
        "num_crops": args.num_crops,
    }
    with (output_root / "checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary = output_root / "selection.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output_root / "selection.json")
    print(
        f"selected_step={selected_step} screen={short_metrics(report['screen'])} "
        f"holdout={short_metrics(holdout)} any_full_gate_pass={report['any_full_gate_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
