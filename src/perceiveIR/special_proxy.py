"""Offline task-specific proxy models for GoPro blur and LOL-v1 low light."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_nafnet(source: str | Path, checkpoint: str | Path, device: torch.device) -> nn.Module:
    source = Path(source)
    if "basicsr" in sys.modules:
        raise RuntimeError("NAFNet must load in an isolated renderer process")
    sys.path.insert(0, str(source))
    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    finally:
        sys.path.remove(str(source))
    model = NAFNet(img_channel=3, width=32, enc_blk_nums=[1, 1, 1, 28],
                   middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["params"], strict=True)
    return model.to(device).requires_grad_(False).eval()


def load_retinexformer(source: str | Path, checkpoint: str | Path,
                       device: torch.device) -> nn.Module:
    model_file = Path(source) / "basicsr/models/archs/RetinexFormer_arch.py"
    RetinexFormer = _load_source_module("perceiveIR_offline_retinexformer", model_file).RetinexFormer
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40,
                         stage=1, num_blocks=[1, 2, 2])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["params"], strict=True)
    return model.to(device).requires_grad_(False).eval()


@torch.inference_mode()
def render_tiled(model: nn.Module, image: torch.Tensor, *, tile: int = 512,
                 overlap: int = 64, multiple: int = 16,
                 amp_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Memory-bounded full-image inference with blended overlapping tiles."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("render_tiled expects a single RGB image")
    if tile <= overlap or overlap < 0 or tile % multiple:
        raise ValueError("tile/overlap/multiple are incompatible")
    height, width = image.shape[-2:]
    output = torch.zeros_like(image, dtype=torch.float32)
    count = torch.zeros((1, 1, height, width), dtype=torch.float32, device=image.device)
    stride = tile - overlap
    ys = list(range(0, max(height - tile, 0) + 1, stride))
    xs = list(range(0, max(width - tile, 0) + 1, stride))
    ys.append(max(height - tile, 0))
    xs.append(max(width - tile, 0))
    for top in sorted(set(ys)):
        for left in sorted(set(xs)):
            patch = image[..., top:min(top + tile, height), left:min(left + tile, width)]
            ph, pw = patch.shape[-2:]
            pad_h = (-ph) % multiple
            pad_w = (-pw) % multiple
            patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")
            with torch.autocast("cuda", dtype=amp_dtype, enabled=image.is_cuda):
                restored = model(patch)
            restored = restored[..., :ph, :pw].float().clamp_(0, 1)
            weight = torch.ones_like(restored[:, :1])
            # Ramp only where another tile overlaps, avoiding edge darkening.
            fade = min(overlap, ph // 2, pw // 2)
            if fade:
                ramp = torch.linspace(0, 1, fade + 2, device=image.device)[1:-1]
                if top > 0:
                    weight[..., :fade, :] *= ramp.view(1, 1, -1, 1)
                if top + ph < height:
                    weight[..., -fade:, :] *= ramp.flip(0).view(1, 1, -1, 1)
                if left > 0:
                    weight[..., :, :fade] *= ramp.view(1, 1, 1, -1)
                if left + pw < width:
                    weight[..., :, -fade:] *= ramp.flip(0).view(1, 1, 1, -1)
            output[..., top:top + ph, left:left + pw] += restored * weight
            count[..., top:top + ph, left:left + pw] += weight
    return (output / count.clamp_min(1e-8)).clamp_(0, 1)
