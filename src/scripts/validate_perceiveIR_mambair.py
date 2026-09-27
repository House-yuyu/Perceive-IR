"""Validate official MambaIR denoisers on one real 256-pixel training crop."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from perceiveIR.data import TASK_TO_ID
from perceiveIR.proxy import MambaIRProxy


def psnr(image: torch.Tensor, target: torch.Tensor) -> float:
    mse = (image.float() - target.float()).square().mean().clamp_min(1e-8)
    return float(-10 * torch.log10(mse))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["data"]["root"])
    if (root / "AiO").is_dir():
        root /= "AiO"
    image_path = sorted((root / "train" / "Denosie" / "gt").glob("*.bmp"))[0]
    clean = pil_to_tensor(Image.open(image_path).convert("RGB")).float().div_(255)
    height, width = clean.shape[-2:]
    if min(height, width) < 256:
        raise RuntimeError(f"chosen clean image is too small: {image_path}")
    clean = clean[:, (height - 256) // 2 : (height + 256) // 2,
                  (width - 256) // 2 : (width + 256) // 2].unsqueeze(0).cuda()
    settings = config["pretrained"]["mambair_proxy"]
    proxy = MambaIRProxy(settings["source"], settings["checkpoints"], clean.device)
    report = {"image": str(image_path), "patch": 256, "device": torch.cuda.get_device_name(0), "sigmas": {}}
    label = torch.tensor([TASK_TO_ID["denoise"]], device=clean.device)
    for sigma in (15, 25, 50):
        torch.manual_seed(1000 + sigma)
        degraded = (clean * 255 + torch.randn_like(clean) * sigma).clamp(0, 255) / 255
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        restored, mask = proxy.render(degraded, label, torch.tensor([sigma], device=clean.device), torch.bfloat16)
        torch.cuda.synchronize()
        if mask.tolist() != [True] or restored.shape != clean.shape or not torch.isfinite(restored).all():
            raise RuntimeError(f"MambaIR sigma {sigma} returned an invalid image or task mask")
        input_psnr = psnr(degraded, clean)
        restored_psnr = psnr(restored, clean)
        report["sigmas"][str(sigma)] = {
            "input_psnr": input_psnr,
            "restored_psnr": restored_psnr,
            "gain_db": restored_psnr - input_psnr,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        print(f"sigma={sigma}: {input_psnr:.3f} -> {restored_psnr:.3f} dB "
              f"({restored_psnr - input_psnr:+.3f}), peak={report['sigmas'][str(sigma)]['peak_allocated_gib']:.2f} GiB", flush=True)
    torch.manual_seed(123)
    noisy_15 = (clean * 255 + torch.randn_like(clean) * 15).clamp(0, 255) / 255
    noisy_50 = (clean * 255 + torch.randn_like(clean) * 50).clamp(0, 255) / 255
    mixed_input = torch.cat((noisy_15, clean, noisy_50))
    mixed_labels = torch.tensor([TASK_TO_ID["denoise"], TASK_TO_ID["dehaze"], TASK_TO_ID["denoise"]], device=clean.device)
    mixed_sigmas = torch.tensor([15, 0, 50], device=clean.device)
    mixed_output, mixed_mask = proxy.render(mixed_input, mixed_labels, mixed_sigmas, torch.bfloat16)
    if mixed_mask.tolist() != [True, False, True] or not torch.equal(mixed_output[1], clean[0]):
        raise RuntimeError("MambaIR mixed-task batch routing changed the ineligible haze sample")
    if not torch.isfinite(mixed_output).all():
        raise RuntimeError("MambaIR mixed-task batch returned a non-finite image")
    report["mixed_batch"] = {"batch_size": 3, "valid_mask": mixed_mask.tolist(), "ineligible_unchanged": True}
    print("mixed batch: denoise/haze/denoise routing passed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
