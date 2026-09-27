from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .data import NOISE_SIGMAS, Sample, index_by_stem, list_images
from .evaluate_stage1_medium import save_contact_sheet
from .render_stage1_medium import save_png_atomic, tiled_forward
from .stage1 import FrozenCLIPImageEncoder, FrozenCLIPTextEncoder, QualityPromptLearner, RestormerMedium
from .stage1_data import load_source_pair, sample_fold


QUALITY_NAMES = ("terrible", "mediocre", "excellent")
TASKS = ("deblur", "dehaze", "denoise", "derain", "lowlight")


@dataclass(frozen=True)
class TestTriplet:
    task: str
    low_path: Path | None
    high_path: Path
    noise_sigma: int | None = None

    @property
    def identity(self) -> str:
        return f"{self.task}|{self.low_path or self.high_path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned quality prompts on unseen AiO test images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-checkpoint", required=True)
    parser.add_argument("--medium-checkpoint-fold0", required=True)
    parser.add_argument("--medium-checkpoint-fold1", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--medium-cache-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--num-crops", type=int, choices=(1, 5), default=5)
    parser.add_argument("--max-per-task", type=int, default=0)
    return parser.parse_args()


def paired_same_stem(input_root: Path, target_root: Path, task: str) -> list[TestTriplet]:
    targets = index_by_stem(target_root)
    result = []
    for low_path in list_images(input_root):
        high_path = targets.get(low_path.stem)
        if high_path is None:
            raise RuntimeError(f"missing target for {low_path}")
        result.append(TestTriplet(task, low_path, high_path))
    return result


def build_test_triplets(data_root: str | Path) -> list[TestTriplet]:
    root = Path(data_root)
    if (root / "AiO").is_dir():
        root = root / "AiO"
    test_root = root / "test"

    result = paired_same_stem(
        test_root / "deblur" / "gopro" / "input",
        test_root / "deblur" / "gopro" / "target",
        "deblur",
    )

    dehaze_targets = index_by_stem(test_root / "dehaze" / "target")
    for low_path in list_images(test_root / "dehaze" / "input"):
        target_stem = low_path.stem.split("_", 1)[0]
        high_path = dehaze_targets.get(target_stem)
        if high_path is None:
            raise RuntimeError(f"missing dehaze target for {low_path}")
        result.append(TestTriplet("dehaze", low_path, high_path))

    denoise_targets = []
    for benchmark in ("bsd68", "kodak24", "urban100"):
        denoise_targets.extend(list_images(test_root / "denoise" / benchmark / "target"))
    for index, high_path in enumerate(sorted(denoise_targets)):
        result.append(TestTriplet("denoise", None, high_path, NOISE_SIGMAS[index % len(NOISE_SIGMAS)]))

    result.extend(
        paired_same_stem(
            test_root / "derain" / "Rain100L" / "input",
            test_root / "derain" / "Rain100L" / "target",
            "derain",
        )
    )
    result.extend(
        paired_same_stem(
            test_root / "enhance" / "lol" / "input",
            test_root / "enhance" / "lol" / "target",
            "lowlight",
        )
    )
    counts = Counter(sample.task for sample in result)
    missing = [task for task in TASKS if counts[task] == 0]
    if missing:
        raise RuntimeError(f"empty test tasks: {missing}")
    return result


def deterministic_subset(samples: list[TestTriplet], maximum: int) -> list[TestTriplet]:
    if maximum <= 0:
        return samples
    grouped: dict[str, list[TestTriplet]] = {task: [] for task in TASKS}
    for sample in samples:
        grouped[sample.task].append(sample)
    result = []
    for task in TASKS:
        ordered = sorted(
            grouped[task],
            key=lambda item: hashlib.sha1(f"prompt-audit|{item.identity}".encode()).hexdigest(),
        )
        result.extend(ordered[:maximum])
    return result


def load_medium_model(config: dict, checkpoint: Path, device: torch.device) -> RestormerMedium:
    model = RestormerMedium(
        dim=config["model"]["dim"],
        blocks=config["model"]["blocks"],
        heads=config["model"]["heads"],
        refinement_blocks=config["model"]["refinement_blocks"],
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["iteration"]) != 20000:
        raise RuntimeError(f"expected a 20K medium checkpoint, found iteration={state['iteration']}")
    model.load_state_dict(state["model"])
    return model.eval()


def load_prompt_model(config: dict, checkpoint: Path, device: torch.device):
    import clip

    clip_model, _ = clip.load(
        "ViT-B/32",
        device="cpu",
        jit=False,
        download_root=config["pretrained"]["clip_cache"],
    )
    clip_model.float().requires_grad_(False).eval().to(device)
    learner = QualityPromptLearner(clip_model, context_length=int(config["prompt"]["context_length"])).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state["iteration"]) <= 0:
        raise RuntimeError(f"invalid prompt checkpoint iteration={state['iteration']}")
    learner.load_state_dict(state["prompt_learner"])
    learner.eval()
    text_encoder = FrozenCLIPTextEncoder(clip_model).to(device).eval()
    image_encoder = FrozenCLIPImageEncoder(clip_model).to(device).eval()
    with torch.inference_mode():
        text_features = F.normalize(text_encoder(learner(), learner.tokenized_prompts), dim=-1)
    return learner, image_encoder, text_features, clip_model.logit_scale.exp().detach()


def prepare_common_images(images: list[Image.Image], crop: int = 224) -> list[Image.Image]:
    width = min(image.width for image in images)
    height = min(image.height for image in images)
    result = [image.convert("RGB").crop((0, 0, width, height)) for image in images]
    if min(width, height) < crop:
        scale = crop / min(width, height)
        size = (max(crop, round(width * scale)), max(crop, round(height * scale)))
        result = [image.resize(size, Image.Resampling.BICUBIC) for image in result]
    return result


def crop_positions(width: int, height: int, crop: int, count: int) -> list[tuple[int, int]]:
    if count == 1:
        return [((width - crop) // 2, (height - crop) // 2)]
    return [
        (0, 0),
        (width - crop, 0),
        (0, height - crop),
        (width - crop, height - crop),
        ((width - crop) // 2, (height - crop) // 2),
    ]


@torch.inference_mode()
def classify_triplet(
    images: list[Image.Image],
    image_encoder: FrozenCLIPImageEncoder,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    device: torch.device,
    num_crops: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = prepare_common_images(images)
    width, height = images[0].size
    positions = crop_positions(width, height, 224, num_crops)
    tensors = []
    for image in images:
        for left, top in positions:
            crop = image.crop((left, top, left + 224, top + 224))
            tensors.append(pil_to_tensor(crop).float().div_(255.0))
    batch = torch.stack(tensors).to(device, non_blocking=True)
    features = image_encoder(batch)
    logits = (logit_scale * features @ text_features.t()).reshape(3, len(positions), 3).mean(dim=1)
    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    quality_axis = torch.arange(3, device=device, dtype=probabilities.dtype)
    scores = probabilities @ quality_axis
    return (
        logits.float().cpu().numpy(),
        probabilities.float().cpu().numpy(),
        scores.float().cpu().numpy(),
    )


def load_low_high(sample: TestTriplet) -> tuple[Image.Image, Image.Image]:
    source = Sample(sample.low_path, sample.high_path, sample.task, sample.noise_sigma)
    digest = hashlib.sha1(sample.identity.encode()).hexdigest()[:16]
    noise_seed = int(digest, 16) % (2**31) if sample.low_path is None else None
    return load_source_pair(source, deterministic_noise_seed=noise_seed)


def atomic_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def confusion_matrix(records: list[dict]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for record in records:
        for truth, prediction in enumerate(record["predictions"]):
            matrix[truth, prediction] += 1
    return matrix


def metrics_for(records: list[dict]) -> dict:
    matrix = confusion_matrix(records)
    diagonal = np.diag(matrix)
    per_quality = diagonal / np.maximum(matrix.sum(axis=1), 1)
    strict = [all(prediction == truth for truth, prediction in enumerate(row["predictions"])) for row in records]
    monotonic = [row["scores"][0] < row["scores"][1] < row["scores"][2] for row in records]
    low_medium = [row["scores"][1] - row["scores"][0] for row in records]
    medium_high = [row["scores"][2] - row["scores"][1] for row in records]
    return {
        "triplets": len(records),
        "images": len(records) * 3,
        "image_accuracy": float(diagonal.sum() / max(matrix.sum(), 1)),
        "per_quality_accuracy": {
            quality: float(value) for quality, value in zip(QUALITY_NAMES, per_quality, strict=True)
        },
        "strict_triplet_rate": float(np.mean(strict)),
        "monotonic_ordering_rate": float(np.mean(monotonic)),
        "low_to_medium_positive_margin_rate": float(np.mean(np.asarray(low_medium) > 0)),
        "medium_to_high_positive_margin_rate": float(np.mean(np.asarray(medium_high) > 0)),
        "low_to_medium_margin_mean": float(np.mean(low_medium)),
        "medium_to_high_margin_mean": float(np.mean(medium_high)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": (
            matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ).tolist(),
    }


def make_summary(records: list[dict], source_counts: Counter) -> dict:
    overall = metrics_for(records)
    by_task = {}
    for task in TASKS:
        by_task[task] = metrics_for([record for record in records if record["task"] == task])
    macro = {
        key: float(np.mean([by_task[task][key] for task in TASKS]))
        for key in ("image_accuracy", "strict_triplet_rate", "monotonic_ordering_rate")
    }
    minimum = {
        key: float(min(by_task[task][key] for task in TASKS))
        for key in ("image_accuracy", "strict_triplet_rate", "monotonic_ordering_rate")
    }
    gate = bool(
        macro["image_accuracy"] >= 0.85
        and minimum["image_accuracy"] >= 0.75
        and macro["strict_triplet_rate"] >= 0.70
        and overall["monotonic_ordering_rate"] >= 0.90
        and minimum["monotonic_ordering_rate"] >= 0.80
    )
    return {
        "test_source_counts": dict(sorted(source_counts.items())),
        "overall": overall,
        "by_task": by_task,
        "task_macro": macro,
        "task_minimum": minimum,
        "suggested_gate": {
            "passed": gate,
            "criteria": {
                "task_macro_image_accuracy": ">= 0.85",
                "minimum_task_image_accuracy": ">= 0.75",
                "task_macro_strict_triplet_rate": ">= 0.70",
                "overall_monotonic_ordering_rate": ">= 0.90",
                "minimum_task_monotonic_ordering_rate": ">= 0.80",
            },
        },
    }


def save_task_csv(summary: dict, path: Path) -> None:
    fields = [
        "task",
        "triplets",
        "image_accuracy",
        "terrible_accuracy",
        "mediocre_accuracy",
        "excellent_accuracy",
        "strict_triplet_rate",
        "monotonic_ordering_rate",
        "low_to_medium_margin_mean",
        "medium_to_high_margin_mean",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            metrics = summary["by_task"][task]
            writer.writerow(
                {
                    "task": task,
                    "triplets": metrics["triplets"],
                    "image_accuracy": metrics["image_accuracy"],
                    "terrible_accuracy": metrics["per_quality_accuracy"]["terrible"],
                    "mediocre_accuracy": metrics["per_quality_accuracy"]["mediocre"],
                    "excellent_accuracy": metrics["per_quality_accuracy"]["excellent"],
                    "strict_triplet_rate": metrics["strict_triplet_rate"],
                    "monotonic_ordering_rate": metrics["monotonic_ordering_rate"],
                    "low_to_medium_margin_mean": metrics["low_to_medium_margin_mean"],
                    "medium_to_high_margin_mean": metrics["medium_to_high_margin_mean"],
                }
            )


def plot_confusions(summary: dict, output: Path) -> None:
    names = ("overall",) + TASKS
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, name in zip(axes.flat, names, strict=True):
        metrics = summary["overall"] if name == "overall" else summary["by_task"][name]
        matrix = np.asarray(metrics["confusion_matrix_normalized"])
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
        axis.set_title(name)
        axis.set_xticks(range(3), QUALITY_NAMES, rotation=30, ha="right")
        axis.set_yticks(range(3), QUALITY_NAMES)
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def representative_records(records: list[dict]) -> list[tuple[str, dict]]:
    ordered = sorted(records, key=lambda row: row["scores"][0])
    quantiles = (0.10, 0.30, 0.50, 0.70, 0.90)
    result = []
    for quantile in quantiles:
        index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
        result.append((f"q{int(quantile * 100):02d}", ordered[index]))
    return result


def save_representative_panels(records: list[dict], output_root: Path) -> None:
    for task in TASKS:
        rows = []
        task_records = [record for record in records if record["task"] == task]
        for quantile, record in representative_records(task_records):
            low = Image.open(record["low_path"]).convert("RGB")
            medium = Image.open(record["medium_path"]).convert("RGB")
            high = Image.open(record["high_path"]).convert("RGB")
            predictions = "/".join(QUALITY_NAMES[index][0].upper() for index in record["predictions"])
            scores = record["scores"]
            label = (
                f"{quantile} pred={predictions} "
                f"score={scores[0]:.2f}/{scores[1]:.2f}/{scores[2]:.2f}"
            )
            rows.append((label, low, medium, high))
        save_contact_sheet(rows, output_root / f"{task}.jpg", cell_size=256)


def write_readme(summary: dict, output: Path) -> None:
    lines = [
        "# perceiveIR held-out prompt audit",
        "",
        "This audit uses only AiO test images that were not used to train the Restormer medium generators or the quality prompts.",
        "",
        "## Aggregate",
        "",
        f"- Triplets: {summary['overall']['triplets']}",
        f"- Image accuracy: {summary['overall']['image_accuracy']:.2%}",
        f"- Task-macro image accuracy: {summary['task_macro']['image_accuracy']:.2%}",
        f"- Strict triplet rate: {summary['overall']['strict_triplet_rate']:.2%}",
        f"- Task-macro strict triplet rate: {summary['task_macro']['strict_triplet_rate']:.2%}",
        f"- Monotonic ordering rate: {summary['overall']['monotonic_ordering_rate']:.2%}",
        f"- Suggested engineering gate: {'PASS' if summary['suggested_gate']['passed'] else 'FAIL'}",
        "",
        "## By degradation",
        "",
        "| Task | N | Image accuracy | Terrible | Mediocre | Excellent | Strict triplet | Monotonic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        metrics = summary["by_task"][task]
        quality = metrics["per_quality_accuracy"]
        lines.append(
            f"| {task} | {metrics['triplets']} | {metrics['image_accuracy']:.2%} | "
            f"{quality['terrible']:.2%} | {quality['mediocre']:.2%} | "
            f"{quality['excellent']:.2%} | {metrics['strict_triplet_rate']:.2%} | "
            f"{metrics['monotonic_ordering_rate']:.2%} |"
        )
    lines += [
        "",
        "Representative panels use the 10th, 30th, 50th, 70th, and 90th percentiles of predicted low-image quality score within each task; they are not worst/best cherry-picks.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    samples = deterministic_subset(build_test_triplets(config["data"]["root"]), args.max_per_task)
    source_counts = Counter(sample.task for sample in samples)
    print(f"test_counts={dict(sorted(source_counts.items()))}", flush=True)

    models = {} if args.medium_cache_root else {
        0: load_medium_model(config, Path(args.medium_checkpoint_fold0), device),
        1: load_medium_model(config, Path(args.medium_checkpoint_fold1), device),
    }
    _, image_encoder, text_features, logit_scale = load_prompt_model(
        config, Path(args.prompt_checkpoint), device
    )

    records = []
    metrics_path = output_root / "records.jsonl"
    with open(metrics_path.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            digest = hashlib.sha1(sample.identity.encode()).hexdigest()[:16]
            base_cache_root = Path(args.medium_cache_root) if args.medium_cache_root else output_root / "cache"
            cache_root = base_cache_root / sample.task
            medium_path = cache_root / "medium" / f"{digest}.png"
            low_path = sample.low_path or (cache_root / "low" / f"{digest}.png")
            low_path = Path(low_path)
            low, high = load_low_high(sample)
            if sample.low_path is None and not low_path.is_file():
                low_path.parent.mkdir(parents=True, exist_ok=True)
                save_png_atomic(low, low_path)
            medium_path.parent.mkdir(parents=True, exist_ok=True)
            if medium_path.is_file() and medium_path.stat().st_size > 0:
                medium = Image.open(medium_path).convert("RGB")
            else:
                if args.medium_cache_root:
                    raise RuntimeError(f"missing cached medium image: {medium_path}")
                tensor = pil_to_tensor(low).float().div_(255.0).unsqueeze(0).to(device)
                fold = sample_fold(Sample(sample.low_path, sample.high_path, sample.task, sample.noise_sigma))
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    restored = tiled_forward(
                        models[fold], tensor, args.tile_size, args.tile_overlap
                    ).clamp(0, 1)
                medium = to_pil_image(restored[0].float().cpu())
                save_png_atomic(medium, medium_path)
            fold = sample_fold(Sample(sample.low_path, sample.high_path, sample.task, sample.noise_sigma))
            logits, probabilities, scores = classify_triplet(
                [low, medium, high],
                image_encoder,
                text_features,
                logit_scale,
                device,
                args.num_crops,
            )
            predictions = probabilities.argmax(axis=1)
            record = {
                "identity": sample.identity,
                "task": sample.task,
                "medium_fold": fold,
                "noise_sigma": sample.noise_sigma,
                "low_path": str(low_path),
                "medium_path": str(medium_path),
                "high_path": str(sample.high_path),
                "predictions": predictions.tolist(),
                "scores": scores.tolist(),
                "logits": logits.tolist(),
                "probabilities": probabilities.tolist(),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index == 1 or index % 25 == 0 or index == len(samples):
                strict = all(prediction == truth for truth, prediction in enumerate(predictions))
                monotonic = bool(scores[0] < scores[1] < scores[2])
                print(
                    f"audited={index}/{len(samples)} task={sample.task} "
                    f"strict={strict} monotonic={monotonic}",
                    flush=True,
                )
    metrics_path.with_suffix(".jsonl.tmp").replace(metrics_path)

    summary = make_summary(records, source_counts)
    summary["configuration"] = {
        "prompt_checkpoint": str(Path(args.prompt_checkpoint).resolve()),
        "medium_cache_root": str(Path(args.medium_cache_root).resolve()) if args.medium_cache_root else None,
        "medium_checkpoint_fold0": str(Path(args.medium_checkpoint_fold0).resolve()),
        "medium_checkpoint_fold1": str(Path(args.medium_checkpoint_fold1).resolve()),
        "num_crops": args.num_crops,
        "max_per_task": args.max_per_task,
    }
    atomic_json(summary, output_root / "summary.json")
    save_task_csv(summary, output_root / "summary_by_task.csv")
    plot_confusions(summary, output_root / "confusion_matrices.png")
    save_representative_panels(records, output_root / "representative_visuals")
    write_readme(summary, output_root / "README.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
