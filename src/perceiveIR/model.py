from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.body = WithBiasLayerNorm(channels) if bias else BiasFreeLayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.body(x)
        return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class FeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.66):
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden * 2, 1)
        self.depthwise = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(left) * right)


class MDTA(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (h c) x y -> b h c (x y)", h=self.heads)
        k = rearrange(k, "b (h c) x y -> b h c (x y)", h=self.heads)
        v = rearrange(v, "b (h c) x y -> b h c (x y)", h=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        output = attention @ v
        output = rearrange(output, "b h c (x y) -> b (h c) x y", x=x.shape[-2], y=x.shape[-1])
        return self.project(output)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, expansion: float = 2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = MDTA(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = FeedForward(channels, expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class PGCA(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.q = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        self.kv = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
        )
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        q = self.q(guidance)
        k, v = self.kv(x).chunk(2, dim=1)
        q = rearrange(q, "b (h c) x y -> b h c (x y)", h=self.heads)
        k = rearrange(k, "b (h c) x y -> b h c (x y)", h=self.heads)
        v = rearrange(v, "b (h c) x y -> b h c (x y)", h=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        output = attention @ v
        output = rearrange(output, "b h c (x y) -> b (h c) x y", x=x.shape[-2], y=x.shape[-1])
        return self.project(output)


class EnhancedTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, expansion: float = 2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.guidance_norm = LayerNorm2d(channels)
        self.attention = PGCA(channels, heads)
        self.mdta_norm = LayerNorm2d(channels)
        self.mdta = MDTA(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = FeedForward(channels, expansion)

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        cross = self.attention(self.norm1(x), self.guidance_norm(guidance))
        x = guidance + self.mdta(self.mdta_norm(cross))
        return x + self.ffn(self.norm2(x))


class EnhancedStage(nn.Module):
    def __init__(self, channels: int, heads: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(EnhancedTransformerBlock(channels, heads) for _ in range(depth))

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, guidance)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, 3, padding=1), nn.PixelUnshuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class CompactFeatureExtractor(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.body(x).flatten(1)), dim=-1)


class PromptGuidance(nn.Module):
    def __init__(self, channels: int, semantic_dim: int = 768, degradation_dim: int = 128):
        super().__init__()
        self.affine = nn.Sequential(
            nn.Linear(semantic_dim + degradation_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, semantic: torch.Tensor, degradation: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(torch.cat((semantic, degradation), dim=-1)).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class PerceiveIR(nn.Module):
    def __init__(
        self,
        dim: int = 48,
        blocks: Sequence[int] = (4, 6, 6, 8),
        heads: Sequence[int] = (1, 2, 4, 8),
        refinement_blocks: int = 4,
        semantic_dim: int = 768,
    ):
        super().__init__()
        if len(blocks) != 4 or len(heads) != 4:
            raise ValueError("blocks and heads must each contain four values")
        channels = (dim, dim * 2, dim * 4, dim * 8)
        self.embed = nn.Conv2d(3, channels[0], 3, padding=1)
        self.encoder1 = nn.Sequential(*(TransformerBlock(channels[0], heads[0]) for _ in range(blocks[0])))
        self.down1 = Downsample(channels[0])
        self.encoder2 = nn.Sequential(*(TransformerBlock(channels[1], heads[1]) for _ in range(blocks[1])))
        self.down2 = Downsample(channels[1])
        self.encoder3 = nn.Sequential(*(TransformerBlock(channels[2], heads[2]) for _ in range(blocks[2])))
        self.down3 = Downsample(channels[2])

        self.cfe = CompactFeatureExtractor(128)
        self.pgm = nn.ModuleList(PromptGuidance(c, semantic_dim, 128) for c in channels)
        self.latent = EnhancedStage(channels[3], heads[3], blocks[3])

        self.up3 = Upsample(channels[3])
        self.reduce3 = nn.Conv2d(channels[2] * 2, channels[2], 1)
        self.decoder3 = EnhancedStage(channels[2], heads[2], blocks[2])
        self.up2 = Upsample(channels[2])
        self.reduce2 = nn.Conv2d(channels[1] * 2, channels[1], 1)
        self.decoder2 = EnhancedStage(channels[1], heads[1], blocks[1])
        self.up1 = Upsample(channels[1])
        self.reduce1 = nn.Conv2d(channels[0] * 2, channels[0], 1)
        self.decoder1 = EnhancedStage(channels[0], heads[0], blocks[0])
        self.refinement = nn.Sequential(*(TransformerBlock(channels[0], heads[0]) for _ in range(refinement_blocks)))
        self.output = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(
        self,
        image: torch.Tensor,
        semantics: Sequence[torch.Tensor],
        return_positive: bool = False,
        positive_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if len(semantics) != 4:
            raise ValueError("four DINO-v2 semantic levels are required")
        degradation = self.cfe(image)
        positive = self.cfe(positive_image if positive_image is not None else torch.flip(image, dims=(-1,))) if return_positive else None

        enc1 = self.encoder1(self.embed(image))
        enc2 = self.encoder2(self.down1(enc1))
        enc3 = self.encoder3(self.down2(enc2))
        latent_in = self.down3(enc3)
        latent = self.latent(latent_in, self.pgm[3](latent_in, semantics[3], degradation))

        dec3 = self.reduce3(torch.cat((self.up3(latent), enc3), dim=1))
        dec3 = self.decoder3(dec3, self.pgm[2](dec3, semantics[2], degradation))
        dec2 = self.reduce2(torch.cat((self.up2(dec3), enc2), dim=1))
        dec2 = self.decoder2(dec2, self.pgm[1](dec2, semantics[1], degradation))
        dec1 = self.reduce1(torch.cat((self.up1(dec2), enc1), dim=1))
        dec1 = self.decoder1(dec1, self.pgm[0](dec1, semantics[0], degradation))
        restored = self.output(self.refinement(dec1)) + image
        return restored, degradation, positive


class DinoSemanticEncoder(nn.Module):
    def __init__(self, model_path: str | Path):
        super().__init__()
        from transformers import Dinov2Model

        self.encoder = Dinov2Model.from_pretrained(str(model_path), local_files_only=True)
        self.encoder.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        resized = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
        normalized = (resized - self.mean) / self.std
        output = self.encoder(pixel_values=normalized, output_hidden_states=True)
        return tuple(output.hidden_states[index][:, 0] for index in (1, 4, 8, 12))
