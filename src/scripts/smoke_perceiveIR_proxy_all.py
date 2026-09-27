"""Exercise every official proxy family and all MambaIR sigmas on one free GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from perceiveIR.data import TASK_TO_ID
from perceiveIR.proxy import FSNetProxy, InstructIRProxy, MambaIRProxy, PromptIRProxy


def check(image: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    if image.shape != reference.shape or not torch.isfinite(image).all():
        raise RuntimeError(f"{name} returned a wrong-shaped or non-finite image")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["pretrained"]
    device = torch.device("cuda:0")
    amp_dtype = torch.bfloat16
    started = time.time()
    image = torch.rand(1, 3, 32, 32, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    report = {}
    prompt = config["promptir_proxy"]
    model = PromptIRProxy(prompt["source"], prompt["checkpoint"], device)
    check(model.render(image, amp_dtype), image, "PromptIR")
    report["promptir"] = "passed"
    del model
    torch.cuda.empty_cache()

    instruct = config["instructir_proxy"]
    model = InstructIRProxy(instruct["source"], instruct["checkpoint"], instruct["embeddings"], device)
    for task in ("denoise", "dehaze", "derain"):
        label = torch.tensor([TASK_TO_ID[task]], device=device)
        check(model.render(image, label, amp_dtype), image, f"InstructIR/{task}")
    report["instructir"] = "passed: denoise, dehaze, derain"
    del model
    torch.cuda.empty_cache()

    fsnet = config["fsnet_proxy"]
    model = FSNetProxy(fsnet["source"], fsnet["checkpoint"], device)
    label = torch.tensor([TASK_TO_ID["dehaze"]], device=device)
    output, mask = model.render(image, label, amp_dtype)
    check(output, image, "FSNet/dehaze")
    if mask.tolist() != [True]:
        raise RuntimeError("FSNet haze mask is incorrect")
    report["fsnet"] = "passed: dehaze"
    del model
    torch.cuda.empty_cache()

    mamba = config["mambair_proxy"]
    model = MambaIRProxy(mamba["source"], mamba["checkpoints"], device)
    label = torch.tensor([TASK_TO_ID["denoise"]], device=device)
    for sigma in (15, 25, 50):
        output, mask = model.render(image, label, torch.tensor([sigma], device=device), amp_dtype)
        check(output, image, f"MambaIR/sigma{sigma}")
        if mask.tolist() != [True]:
            raise RuntimeError(f"MambaIR sigma{sigma} mask is incorrect")
    report["mambair"] = "passed: sigma 15, 25, 50"
    report["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
    report["seconds"] = round(time.time() - started, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
