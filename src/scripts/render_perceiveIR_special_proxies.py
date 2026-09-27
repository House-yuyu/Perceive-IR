"""Render frozen task-specific proxies from training inputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from perceiveIR.data import build_paper_samples
from perceiveIR.special_proxy import load_nafnet, load_retinexformer, render_tiled


def psnr(first: torch.Tensor, second: torch.Tensor) -> float:
    mse = float((first - second).square().mean())
    return -10.0 * math.log10(max(mse, 1e-8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("deblur", "lowlight"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid renderer shard specification")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    samples = build_paper_samples(args.data_root, tasks=[args.task],
                                  task_resampling={args.task: 1})
    expected = {"deblur": 2103, "lowlight": 485}[args.task]
    if len(samples) != expected:
        raise RuntimeError(f"unexpected {args.task} source count: {len(samples)} != {expected}")
    samples = samples[args.shard_index::args.num_shards]
    if args.limit is not None:
        samples = samples[:args.limit]
    task_root = args.output_root / args.task
    task_root.mkdir(parents=True, exist_ok=True)
    if args.task == "deblur":
        model = load_nafnet(args.source, args.checkpoint, device)
        model_name = "NAFNet-GoPro-width32"
    else:
        model = load_retinexformer(args.source, args.checkpoint, device)
        model_name = "RetinexFormer-LOL_v1"
    records = []
    for index, sample in enumerate(samples, start=1):
        source = sample.lq_path
        assert source is not None
        name = hashlib.sha1(str(source).encode()).hexdigest() + ".png"
        output = task_root / name
        with Image.open(source) as opened:
            low_image = opened.convert("RGB")
        if not output.is_file():
            low = pil_to_tensor(low_image).float().div_(255.0).unsqueeze(0).to(device)
            restored = render_tiled(model, low, tile=args.tile, overlap=args.overlap)[0].cpu()
            temporary = output.with_suffix(".tmp")
            to_pil_image(restored).save(temporary, format="PNG")
            temporary.replace(output)
        with Image.open(output) as opened:
            restored_image = opened.convert("RGB")
        if restored_image.size != low_image.size:
            raise RuntimeError(f"proxy image size mismatch: {output}")
        with Image.open(sample.gt_path) as opened:
            target_image = opened.convert("RGB")
        if target_image.size != low_image.size:
            raise RuntimeError(f"training input/GT size mismatch: {source}")
        low_cpu = pil_to_tensor(low_image).float().div_(255.0)
        target_cpu = pil_to_tensor(target_image).float().div_(255.0)
        restored_cpu = pil_to_tensor(restored_image).float().div_(255.0)
        records.append({"task": args.task, "input": str(source), "output": str(output),
                        "model": model_name, "input_psnr": psnr(low_cpu, target_cpu),
                        "output_psnr": psnr(restored_cpu, target_cpu)})
        if index % 25 == 0 or index == len(samples):
            print(f"{args.task}: {index}/{len(samples)}", flush=True)
    if args.limit is not None:
        manifest_name = f"manifest_smoke_{args.shard_index}-of-{args.num_shards}.jsonl"
    elif args.num_shards > 1:
        manifest_name = f"manifest_shard_{args.shard_index}-of-{args.num_shards}.jsonl"
    else:
        manifest_name = "manifest.jsonl"
    manifest = task_root / manifest_name
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    temporary.replace(manifest)
    summary = {"task": args.task, "model": model_name, "count": len(records),
               "mean_input_psnr": sum(row["input_psnr"] for row in records) / len(records),
               "mean_output_psnr": sum(row["output_psnr"] for row in records) / len(records),
               "positive_gain_count": sum(row["output_psnr"] > row["input_psnr"] for row in records),
               "manifest": str(manifest)}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
