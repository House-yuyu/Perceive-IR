from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .model import Downsample, TransformerBlock, Upsample


class RestormerMedium(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        blocks: Sequence[int] = (4, 6, 6, 8),
        heads: Sequence[int] = (1, 2, 4, 8),
        refinement_blocks: int = 4,
    ):
        super().__init__()
        if len(blocks) != 4 or len(heads) != 4:
            raise ValueError("blocks and heads must each contain four values")
        channels = (dim, dim * 2, dim * 4, dim * 8)
        self.embed = nn.Conv2d(3, channels[0], 3, padding=1)
        self.encoder1 = self._stage(channels[0], heads[0], blocks[0])
        self.down1 = Downsample(channels[0])
        self.encoder2 = self._stage(channels[1], heads[1], blocks[1])
        self.down2 = Downsample(channels[1])
        self.encoder3 = self._stage(channels[2], heads[2], blocks[2])
        self.down3 = Downsample(channels[2])
        self.latent = self._stage(channels[3], heads[3], blocks[3])

        self.up3 = Upsample(channels[3])
        self.reduce3 = nn.Conv2d(channels[2] * 2, channels[2], 1)
        self.decoder3 = self._stage(channels[2], heads[2], blocks[2])
        self.up2 = Upsample(channels[2])
        self.reduce2 = nn.Conv2d(channels[1] * 2, channels[1], 1)
        self.decoder2 = self._stage(channels[1], heads[1], blocks[1])
        self.up1 = Upsample(channels[1])
        self.reduce1 = nn.Conv2d(channels[0] * 2, channels[0], 1)
        self.decoder1 = self._stage(channels[0], heads[0], blocks[0])
        self.refinement = self._stage(channels[0], heads[0], refinement_blocks)
        self.output = nn.Conv2d(channels[0], 3, 3, padding=1)

    @staticmethod
    def _stage(channels: int, heads: int, depth: int) -> nn.Sequential:
        return nn.Sequential(*(TransformerBlock(channels, heads) for _ in range(depth)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(self.embed(image))
        encoder2 = self.encoder2(self.down1(encoder1))
        encoder3 = self.encoder3(self.down2(encoder2))
        latent = self.latent(self.down3(encoder3))
        decoder3 = self.decoder3(self.reduce3(torch.cat((self.up3(latent), encoder3), dim=1)))
        decoder2 = self.decoder2(self.reduce2(torch.cat((self.up2(decoder3), encoder2), dim=1)))
        decoder1 = self.decoder1(self.reduce1(torch.cat((self.up1(decoder2), encoder1), dim=1)))
        return self.output(self.refinement(decoder1)) + image


class QualityPromptLearner(nn.Module):
    quality_names = ("terrible", "mediocre", "excellent")

    def __init__(self, clip_model: nn.Module, context_length: int = 16):
        super().__init__()
        import clip

        if context_length < 1 or context_length > 75:
            raise ValueError("context_length must be in [1, 75]")
        self.context_length = int(context_length)
        width = clip_model.ln_final.weight.shape[0]
        context = torch.empty(len(self.quality_names), context_length, width)
        nn.init.normal_(context, std=0.02)
        quality_tokens = clip.tokenize(list(self.quality_names)).to(
            clip_model.token_embedding.weight.device
        )
        end_of_text = quality_tokens.argmax(dim=-1)
        if not torch.all(end_of_text == 2):
            raise RuntimeError("quality names must each tokenize to exactly one CLIP token")
        with torch.no_grad():
            quality_embeddings = clip_model.token_embedding(quality_tokens)[:, 1]
        context[:, -1] = quality_embeddings.to(context)
        self.context = nn.Parameter(context)

        placeholder = " ".join("X" for _ in range(context_length))
        tokenized = clip.tokenize([placeholder] * len(self.quality_names)).to(
            clip_model.token_embedding.weight.device
        )
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized)
        self.register_buffer("token_prefix", embedding[:, :1])
        self.register_buffer("token_suffix", embedding[:, 1 + context_length :])
        self.register_buffer("tokenized_prompts", tokenized)

    def forward(self, _device_anchor: torch.Tensor | None = None) -> torch.Tensor:
        return torch.cat((self.token_prefix, self.context, self.token_suffix), dim=1)


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.requires_grad_(False)

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts.to(self.dtype) + self.positional_embedding.to(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).to(self.dtype)
        end_of_text = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), end_of_text] @ self.text_projection


class FrozenCLIPImageEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.visual = clip_model.visual
        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1))
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
        images = (images - self.mean) / self.std
        return F.normalize(self.visual(images), dim=-1)
