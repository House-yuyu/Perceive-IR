from __future__ import annotations

import importlib.util
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from .data import TASKS, TASK_TO_ID
from .stage1 import RestormerMedium


class DenoiseProxyModels:
    def __init__(self, paths: list[str | Path], model_config: dict, device: torch.device):
        if len(paths) != 2:
            raise ValueError("denoise_proxy_checkpoints must list fold 0 then fold 1")
        self.models: list[nn.Module] = []
        for fold, path in enumerate(paths):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if int(checkpoint.get("heldout_fold", -1)) != fold:
                raise RuntimeError(f"denoise proxy checkpoint fold mismatch: {path}")
            model = RestormerMedium(
                dim=int(model_config["dim"]),
                blocks=tuple(model_config["blocks"]),
                heads=tuple(model_config["heads"]),
                refinement_blocks=int(model_config["refinement_blocks"]),
            )
            model.load_state_dict(checkpoint["model"])
            self.models.append(model.to(device).requires_grad_(False).eval())

    @torch.no_grad()
    def render(
        self, degraded: torch.Tensor, proxies: torch.Tensor,
        task_labels: torch.Tensor, folds: torch.Tensor, amp_dtype: torch.dtype,
    ) -> torch.Tensor:
        proxies = proxies.clone()
        with torch.autocast("cuda", dtype=amp_dtype):
            for fold, model in enumerate(self.models):
                indices = torch.nonzero((task_labels == TASK_TO_ID["denoise"]) & (folds == fold)).flatten()
                if indices.numel():
                    restored = model(degraded.index_select(0, indices)).float().clamp(0, 1)
                    proxies.index_copy_(0, indices, restored)
        return proxies


class PromptIRProxy:
    """Frozen, official all-in-one PromptIR checkpoint used as a DPL negative."""

    def __init__(self, source: str | Path, checkpoint_path: str | Path, device: torch.device):
        model_file = Path(source) / "net" / "model.py"
        if not model_file.is_file():
            raise FileNotFoundError(f"PromptIR model source is missing: {model_file}")
        spec = importlib.util.spec_from_file_location("perceiveIR_promptir_proxy_model", model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load PromptIR model source: {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.PromptIR(decoder=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" not in checkpoint:
            raise RuntimeError("PromptIR checkpoint does not contain a Lightning state_dict")
        state_dict = checkpoint["state_dict"]
        if not state_dict or not all(key.startswith("net.") for key in state_dict):
            raise RuntimeError("PromptIR checkpoint has unexpected parameter names")
        model.load_state_dict({key.removeprefix("net."): value for key, value in state_dict.items()}, strict=True)
        self.model = model.to(device).requires_grad_(False).eval()

    @torch.no_grad()
    def render(self, degraded: torch.Tensor, amp_dtype: torch.dtype) -> torch.Tensor:
        with torch.autocast("cuda", dtype=amp_dtype, enabled=degraded.is_cuda):
            return self.model(degraded).float().clamp(0, 1)


@contextmanager
def _external_models(source: Path):
    """Import an official repository's `models` package without leaking its generic name."""
    if "models" in sys.modules or any(name.startswith("models.") for name in sys.modules):
        raise RuntimeError("another top-level models package is already imported")
    sys.path.insert(0, str(source))
    try:
        yield
    finally:
        sys.path.remove(str(source))
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]


class InstructIRProxy:
    """Official seven-degradation InstructIR image model with frozen task text vectors."""

    def __init__(self, source: str | Path, checkpoint_path: str | Path,
                 embeddings_path: str | Path, device: torch.device):
        with _external_models(Path(source)):
            create_model = importlib.import_module("models.instructir").create_model
        model = create_model(input_channels=3, width=32, enc_blks=[2, 2, 4, 8],
                             middle_blk_num=4, dec_blks=[2, 2, 2, 2], txtdim=256)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False), strict=True)
        embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        names = TASKS if set(embeddings) == set(TASKS) else TASKS[:3]
        if set(embeddings) != set(names):
            raise ValueError(f"InstructIR task embeddings have unexpected keys: {sorted(embeddings)}")
        self.embeddings = torch.stack([embeddings[name] for name in names]).to(device)
        if self.embeddings.shape != (len(names), 256):
            raise ValueError("InstructIR task embeddings must contain 256-dimensional task vectors")
        self.model = model.to(device).requires_grad_(False).eval()

    @torch.no_grad()
    def render(self, degraded: torch.Tensor, labels: torch.Tensor, amp_dtype: torch.dtype) -> torch.Tensor:
        if labels.numel() and int(labels.max()) >= len(self.embeddings):
            raise ValueError("InstructIR task embedding missing for a training task")
        with torch.autocast("cuda", dtype=amp_dtype, enabled=degraded.is_cuda):
            return self.model(degraded, self.embeddings.index_select(0, labels)).float().clamp(0, 1)


class FSNetProxy:
    """Official OTS dehazing checkpoint; only haze samples are eligible."""

    def __init__(self, source: str | Path, checkpoint_path: str | Path, device: torch.device):
        with _external_models(Path(source) / "Dehazing" / "OTS"):
            fsnet = importlib.import_module("models.FSNet").FSNet
        model = fsnet(num_res=16)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        self.model = model.to(device).requires_grad_(False).eval()

    @torch.no_grad()
    def render(self, degraded: torch.Tensor, labels: torch.Tensor,
               amp_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        valid = labels == TASK_TO_ID["dehaze"]
        result = degraded.clone()
        indices = valid.nonzero().flatten()
        if indices.numel():
            with torch.autocast("cuda", dtype=amp_dtype, enabled=degraded.is_cuda):
                restored = self.model(degraded.index_select(0, indices))[2].float().clamp(0, 1)
            result.index_copy_(0, indices, restored)
        return result, valid


class MambaIRProxy:
    """Official color-denoising models, selected by the sample's Gaussian sigma."""

    def __init__(self, source: str | Path, checkpoints: dict[int, str | Path], device: torch.device):
        source = Path(source)
        sys.path.insert(0, str(source))
        try:
            mambair = importlib.import_module("basicsr.archs.mambair_arch").MambaIR
        finally:
            sys.path.remove(str(source))
        if set(int(sigma) for sigma in checkpoints) != {15, 25, 50}:
            raise ValueError("MambaIR requires official sigma-15/25/50 checkpoints")
        self.models: dict[int, nn.Module] = {}
        for sigma, path in checkpoints.items():
            model = mambair(upscale=1, in_chans=3, img_size=128, img_range=1.,
                            d_state=16, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, mlp_ratio=1.2)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["params"], strict=True)
            self.models[int(sigma)] = model.to(device).requires_grad_(False).eval()

    @torch.no_grad()
    def render(self, degraded: torch.Tensor, labels: torch.Tensor, sigmas: torch.Tensor,
               amp_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        valid = labels == TASK_TO_ID["denoise"]
        if (valid & ~torch.isin(sigmas, torch.tensor((15, 25, 50), device=sigmas.device))).any():
            raise ValueError("a denoising sample has no matching MambaIR sigma checkpoint")
        result = degraded.clone()
        for sigma, model in self.models.items():
            indices = (valid & (sigmas == sigma)).nonzero().flatten()
            if indices.numel():
                with torch.autocast("cuda", dtype=amp_dtype, enabled=degraded.is_cuda):
                    restored = model(degraded.index_select(0, indices)).float().clamp(0, 1)
                result.index_copy_(0, indices, restored)
        return result, valid
