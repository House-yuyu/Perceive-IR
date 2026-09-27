"""Audit five-task proxy quality on the fixed, training-only DPL probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from perceiveIR.data import TASK_TO_ID
from perceiveIR.epoch_probe import load_pair
from perceiveIR.proxy import InstructIRProxy
from perceiveIR.special_proxy import load_nafnet, load_retinexformer, render_tiled


def psnr(image: torch.Tensor, target: torch.Tensor) -> float:
    mse = float((image.float().clamp(0, 1) - target).square().mean())
    return -10.0 * math.log10(max(mse, 1e-8))


def medium_lookup(manifests: list[Path], tasks: set[str]) -> dict[tuple[str, str], Path]:
    result = {}
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item["task"] in tasks:
                    key = (item["task"], item["low"])
                    if key in result:
                        raise RuntimeError(f"duplicate medium proxy: {key}")
                    result[key] = Path(item["medium"])
    return result


def load_medium(path: Path, record: dict) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        size = tuple(record["resize"])
        if image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        x, y, patch = record["crop"]
        image = image.crop((x, y, x + patch, y + patch))
        return pil_to_tensor(image).float().div_(255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=24)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--naf-source", type=Path)
    parser.add_argument("--naf-checkpoint", type=Path)
    parser.add_argument("--retinex-source", type=Path)
    parser.add_argument("--retinex-checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    tasks = {"deblur", "lowlight"}
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    paths = medium_lookup(args.manifest, tasks)
    model = InstructIRProxy(args.source, args.checkpoint, args.embeddings, torch.device(args.device))
    if any((args.naf_source, args.naf_checkpoint, args.retinex_source, args.retinex_checkpoint)):
        if not all((args.naf_source, args.naf_checkpoint, args.retinex_source, args.retinex_checkpoint)):
            raise ValueError("both complete specialist model specifications are required")
        specialists = {
            "deblur": load_nafnet(args.naf_source, args.naf_checkpoint, torch.device(args.device)),
            "lowlight": load_retinexformer(args.retinex_source, args.retinex_checkpoint,
                                            torch.device(args.device)),
        }
    else:
        specialists = {}
    records = []
    counts = {task: 0 for task in tasks}
    for item in probe["records"]:
        task = item["task"]
        if task not in tasks or counts[task] >= args.per_task:
            continue
        counts[task] += 1
        low, high = load_pair(item)
        medium = load_medium(paths[(task, item["input"])], item)
        if low.shape != high.shape or low.shape != medium.shape:
            raise RuntimeError(f"unaligned sample: {item['input']}")
        label = torch.tensor([TASK_TO_ID[task]], device=args.device)
        with torch.inference_mode():
            restored = model.render(low.unsqueeze(0).to(args.device), label, torch.bfloat16)[0].cpu()
        scores = {name: psnr(image, high) for name, image in
                  (("input", low), ("restormer", medium), ("instructir", restored))}
        if specialists:
            with torch.inference_mode():
                candidate = render_tiled(specialists[task], low.unsqueeze(0).to(args.device),
                                         tile=256, overlap=64)[0].cpu()
            scores["task_specific"] = psnr(candidate, high)
        records.append({"task": task, "input": item["input"], "psnr": scores})
    if any(counts[task] != args.per_task for task in tasks):
        raise RuntimeError(f"probe coverage is incomplete: {counts}")
    summary = {}
    for task in sorted(tasks):
        rows = [row["psnr"] for row in records if row["task"] == task]
        summary[task] = {
            "count": len(rows),
            "mean_psnr": {name: sum(row[name] for row in rows) / len(rows)
                          for name in rows[0]},
            "positive_gain_count": {name: sum(row[name] > row["input"] for row in rows)
                                    for name in rows[0] if name != "input"},
        }
    report = {"purpose": "train-only 256-pixel proxy audit; not held-out validation",
              "probe": str(args.probe), "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
