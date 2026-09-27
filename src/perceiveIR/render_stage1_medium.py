from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .stage1 import RestormerMedium
from .stage1_data import build_source_samples, load_source_pair, sample_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--heldout-fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=32)
    return parser.parse_args()


def tiled_forward(model, image: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
    def forward_with_padding(tile: torch.Tensor) -> torch.Tensor:
        tile_height, tile_width = tile.shape[-2:]
        pad_h = (-tile_height) % 8
        pad_w = (-tile_width) % 8
        padded = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")
        return model(padded)[..., :tile_height, :tile_width]

    _, _, height, width = image.shape
    if height <= tile_size and width <= tile_size:
        return forward_with_padding(image)
    stride = tile_size - overlap
    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)
    y_positions = list(range(0, max(height - tile_size, 0) + 1, stride))
    x_positions = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not y_positions or y_positions[-1] != max(height - tile_size, 0):
        y_positions.append(max(height - tile_size, 0))
    if not x_positions or x_positions[-1] != max(width - tile_size, 0):
        x_positions.append(max(width - tile_size, 0))
    for top in y_positions:
        for left in x_positions:
            tile = image[..., top : top + tile_size, left : left + tile_size]
            prediction = forward_with_padding(tile)
            output[..., top : top + tile_size, left : left + tile_size] += prediction
            weight[..., top : top + tile_size, left : left + tile_size] += 1
    return output / weight.clamp_min(1)


def save_png_atomic(image: Image.Image, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    model = RestormerMedium(
        dim=config["model"]["dim"],
        blocks=config["model"]["blocks"],
        heads=config["model"]["heads"],
        refinement_blocks=config["model"]["refinement_blocks"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    output_root = Path(config["medium"]["triplet_root"]) / f"fold_{args.heldout_fold}"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.jsonl"
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    samples = [
        sample for sample in build_source_samples(config["data"]["root"])
        if sample_fold(sample) == args.heldout_fold
    ]
    rendered = reused = 0
    with open(temporary_manifest, "w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            identity = f"{sample.task}|{sample.lq_path or sample.gt_path}"
            digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
            task_root = output_root / sample.task
            medium_path = task_root / "medium" / f"{digest}.png"
            if sample.lq_path is None:
                low_path = task_root / "low" / f"{digest}.png"
            else:
                low_path = sample.lq_path
            medium_path.parent.mkdir(parents=True, exist_ok=True)
            if sample.lq_path is None:
                Path(low_path).parent.mkdir(parents=True, exist_ok=True)
            medium_ready = medium_path.is_file() and medium_path.stat().st_size > 0
            low_ready = Path(low_path).is_file() and Path(low_path).stat().st_size > 0
            if not medium_ready or not low_ready:
                noise_seed = int(digest, 16) % (2**31) if sample.lq_path is None else None
                low_image, _ = load_source_pair(sample, deterministic_noise_seed=noise_seed)
                if not medium_ready:
                    tensor = pil_to_tensor(low_image).float().div_(255.0).unsqueeze(0).to(device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        restored = tiled_forward(model, tensor, args.tile_size, args.tile_overlap).clamp(0, 1)
                    save_png_atomic(to_pil_image(restored[0].float().cpu()), medium_path)
                    rendered += 1
                if sample.lq_path is None and not low_ready:
                    save_png_atomic(low_image, Path(low_path))
            else:
                reused += 1
            record = {
                "low": str(low_path),
                "medium": str(medium_path),
                "high": str(sample.gt_path),
                "task": sample.task,
                "fold": args.heldout_fold,
            }
            handle.write(json.dumps(record) + "\n")
            if (index + 1) % 100 == 0:
                print(
                    f"fold={args.heldout_fold} completed={index + 1}/{len(samples)} "
                    f"new={rendered} reused={reused}",
                    flush=True,
                )
    temporary_manifest.replace(manifest)
    print(
        f"completed fold={args.heldout_fold} samples={len(samples)} new={rendered} "
        f"reused={reused} manifest={manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
