from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def degradation_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    return_valid_fraction: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    positive_logit = (anchor * positive).sum(dim=-1, keepdim=True) / temperature
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        local_size = torch.tensor([anchor.shape[0]], device=anchor.device)
        sizes = [torch.empty_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        max_size = max(int(size.item()) for size in sizes)
        padded_features = F.pad(anchor.detach(), (0, 0, 0, max_size - anchor.shape[0]))
        padded_labels = F.pad(labels.detach(), (0, max_size - labels.shape[0]), value=-1)
        gathered_features = [torch.empty_like(padded_features) for _ in range(world_size)]
        gathered_labels = [torch.empty_like(padded_labels) for _ in range(world_size)]
        dist.all_gather(gathered_features, padded_features)
        dist.all_gather(gathered_labels, padded_labels)
        negative_bank = torch.cat(
            [features[:int(size.item())] for features, size in zip(gathered_features, sizes)], dim=0
        )
        negative_labels = torch.cat(
            [rank_labels[:int(size.item())] for rank_labels, size in zip(gathered_labels, sizes)], dim=0
        )
    else:
        negative_bank = anchor.detach()
        negative_labels = labels.detach()

    all_losses: list[torch.Tensor] = []
    for index in range(anchor.shape[0]):
        mask = negative_labels != labels[index]
        negatives = negative_bank[mask]
        if negatives.numel() == 0:
            continue
        negative_logits = anchor[index : index + 1] @ negatives.t() / temperature
        logits = torch.cat((positive_logit[index : index + 1], negative_logits), dim=1)
        all_losses.append(F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=anchor.device)))
    loss = torch.stack(all_losses).mean() if all_losses else anchor.sum() * 0.0
    if return_valid_fraction:
        return loss, len(all_losses) / anchor.shape[0]
    return loss


class QualityCLIPLoss(nn.Module):
    def __init__(self, cache_dir: str | Path, prompt_checkpoint: str | Path):
        super().__init__()
        import clip
        from .stage1 import FrozenCLIPTextEncoder, QualityPromptLearner

        model, _ = clip.load("ViT-B/32", device="cpu", jit=False, download_root=str(cache_dir))
        model.float().requires_grad_(False).eval()
        self.model = model
        checkpoint = torch.load(prompt_checkpoint, map_location="cpu", weights_only=False)
        if int(checkpoint["iteration"]) != 100000 or not checkpoint.get("task_balanced", False):
            raise RuntimeError("restoration requires the accepted 100K task-balanced quality prompts")
        learner = QualityPromptLearner(model, context_length=16)
        learner.load_state_dict(checkpoint["prompt_learner"])
        learner.requires_grad_(False).eval()
        text_encoder = FrozenCLIPTextEncoder(model).eval()
        with torch.no_grad():
            text_features = F.normalize(text_encoder(learner(), learner.tokenized_prompts), dim=-1)
        self.register_buffer("quality_features", text_features)
        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(image.clamp(0, 1), size=(224, 224), mode="bicubic", align_corners=False)
        normalized = (resized - self.mean) / self.std
        image_features = F.normalize(self.model.encode_image(normalized), dim=-1)
        logits = image_features @ self.quality_features.t()
        excellent_probability = logits.softmax(dim=-1)[:, 2]
        return (1.0 - excellent_probability).mean()


class VGGFeatures(nn.Module):
    def __init__(self, weights_path: str | Path):
        super().__init__()
        from torchvision.models import vgg16

        model = vgg16(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.features = model.features[:16].requires_grad_(False).eval()
        self.indices = {3, 7, 11, 15}
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(False)
        self.features.eval()
        return self

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = (image.clamp(0, 1) - self.mean) / self.std
        outputs: list[torch.Tensor] = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.indices:
                outputs.append(x)
        return outputs


class DifficultyAdaptivePerceptualLoss(nn.Module):
    def __init__(self, weights_path: str | Path):
        super().__init__()
        self.vgg = VGGFeatures(weights_path)
        self.layer_weights = (1 / 12, 1 / 6, 1 / 3, 1.0)

    def forward(
        self,
        restored: torch.Tensor,
        ground_truth: torch.Tensor,
        degraded: torch.Tensor,
        proxies: Sequence[torch.Tensor] = (),
        proxy_weights: Sequence[float | torch.Tensor] = (),
        proxy_masks: Sequence[torch.Tensor] = (),
    ) -> torch.Tensor:
        if proxy_weights and len(proxy_weights) != len(proxies):
            raise ValueError("one DPL weight is required per proxy")
        if proxy_masks and len(proxy_masks) != len(proxies):
            raise ValueError("one DPL validity mask is required per proxy")
        restored_features = self.vgg(restored)
        with torch.no_grad():
            gt_features = self.vgg(ground_truth)
            degraded_features = self.vgg(degraded)
            proxy_features = [self.vgg(proxy) for proxy in proxies]

        total = restored.new_zeros((restored.shape[0],))
        def feature_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            return (first - second).abs().mean(dim=(1, 2, 3))

        for level, weight in enumerate(self.layer_weights):
            positive = feature_distance(restored_features[level], gt_features[level])
            denominator = 2.0 * feature_distance(restored_features[level], degraded_features[level])
            for proxy_index, features in enumerate(proxy_features):
                proxy_weight = proxy_weights[proxy_index] if proxy_weights else 1.0
                proxy_mask = proxy_masks[proxy_index] if proxy_masks else 1.0
                denominator = denominator + proxy_mask * proxy_weight * feature_distance(
                    restored_features[level], features[level]
                )
            total = total + weight * positive / denominator.clamp_min(1e-6)
        return total.mean()
