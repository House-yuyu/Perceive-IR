import random
import json

import numpy as np
import torch
from PIL import Image
from torch import nn

from perceiveIR.data import PaperFiveDataset, Sample
from perceiveIR.losses import DifficultyAdaptivePerceptualLoss, degradation_contrastive_loss
from perceiveIR.proxy import FSNetProxy, MambaIRProxy
from perceiveIR.special_proxy import render_tiled


def test_denoise_positive_is_from_the_same_noisy_image(monkeypatch):
    dataset = PaperFiveDataset.__new__(PaperFiveDataset)
    dataset.patch_size = 16
    positions = iter((0, 0, 8, 8))
    monkeypatch.setattr(random, "randint", lambda _low, _high: next(positions))
    monkeypatch.setattr(random, "random", lambda: 1.0)
    torch.manual_seed(13)

    low, high, positive = dataset._denoise_crops_from_one_image(
        Image.new("RGB", (32, 32), (128, 128, 128)), sigma=25.0,
    )

    assert low.shape == high.shape == positive.shape == (3, 16, 16)
    torch.testing.assert_close(low[:, 8:, 8:], positive[:, :8, :8], rtol=0, atol=0)
    assert not torch.equal(low, positive)


class _IdentityFeatures(nn.Module):
    def forward(self, image):
        return [image] * 4


def test_dpl_sums_independent_proxy_negatives():
    loss = DifficultyAdaptivePerceptualLoss.__new__(DifficultyAdaptivePerceptualLoss)
    nn.Module.__init__(loss)
    loss.vgg = _IdentityFeatures()
    loss.layer_weights = (1 / 12, 1 / 6, 1 / 3, 1.0)
    restored = torch.full((1, 3, 4, 4), 0.5, requires_grad=True)
    gt = torch.full_like(restored, 0.9)
    degraded = torch.zeros_like(restored)
    first = torch.full_like(restored, 0.2)
    second = torch.full_like(restored, 0.7)

    value = loss(restored, gt, degraded, (first, second), (torch.tensor([1.25]), torch.tensor([0.75])))
    expected = sum(loss.layer_weights) * 0.4 / (2 * 0.5 + 1.25 * 0.3 + 0.75 * 0.2)
    torch.testing.assert_close(value, torch.tensor(expected), rtol=1e-6, atol=1e-6)
    value.backward()
    assert restored.grad is not None and torch.isfinite(restored.grad).all()


def test_contrastive_reports_when_a_batch_has_no_cross_task_negative():
    features = torch.eye(2, requires_grad=True)
    value, fraction = degradation_contrastive_loss(
        features, features, torch.zeros(2, dtype=torch.long), return_valid_fraction=True,
    )
    assert fraction == 0.0
    assert value.item() == 0.0


def test_dpl_ignores_proxy_outside_its_valid_task():
    loss = DifficultyAdaptivePerceptualLoss.__new__(DifficultyAdaptivePerceptualLoss)
    nn.Module.__init__(loss)
    loss.vgg = _IdentityFeatures()
    loss.layer_weights = (1 / 12, 1 / 6, 1 / 3, 1.0)
    restored = torch.full((2, 3, 4, 4), 0.5, requires_grad=True)
    gt = torch.full_like(restored, 0.9)
    degraded = torch.zeros_like(restored)
    first = torch.full_like(restored, 0.2)
    irrelevant = torch.full_like(restored, 100.0)
    masked = loss(restored, gt, degraded, (first, irrelevant),
                  (torch.ones(2), torch.ones(2)),
                  (torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)))
    unmasked = loss(restored, gt, degraded, (first,))
    torch.testing.assert_close(masked, unmasked)


class _ToyFSNet(nn.Module):
    def forward(self, image):
        return [image, image, image + 0.1]


class _ToyDenoiser(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, image):
        return image + self.value


def test_specialized_proxy_task_and_sigma_routing():
    image = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([0, 0, 1, 2])
    sigmas = torch.tensor([15, 50, 0, 0])
    fsnet = FSNetProxy.__new__(FSNetProxy)
    fsnet.model = _ToyFSNet()
    haze, haze_mask = fsnet.render(image, labels, torch.bfloat16)
    torch.testing.assert_close(haze_mask, torch.tensor([False, False, True, False]))
    assert haze[2].mean().item() > 0
    assert haze[[0, 1, 3]].sum().item() == 0
    mamba = MambaIRProxy.__new__(MambaIRProxy)
    mamba.models = {15: _ToyDenoiser(0.1), 25: _ToyDenoiser(0.2), 50: _ToyDenoiser(0.3)}
    noise, noise_mask = mamba.render(image, labels, sigmas, torch.bfloat16)
    torch.testing.assert_close(noise_mask, torch.tensor([True, True, False, False]))
    torch.testing.assert_close(noise[:, 0, 0, 0], torch.tensor([0.1, 0.3, 0.0, 0.0]))


def test_task_specific_precomputed_proxy_is_aligned_and_masked(tmp_path, monkeypatch):
    pattern = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) % 180
    low = np.stack((pattern, pattern + 10, pattern + 20), axis=-1).astype(np.uint8)
    boosted = (low.astype(np.uint16) + 10).astype(np.uint8)
    low_path = tmp_path / "input.png"
    high_path = tmp_path / "target.png"
    extra_path = tmp_path / "special.png"
    Image.fromarray(low).save(low_path)
    Image.fromarray(low).save(high_path)
    Image.fromarray(boosted).save(extra_path)
    samples = [Sample(low_path, high_path, task) for task in ("deblur", "lowlight", "dehaze")]
    monkeypatch.setattr("perceiveIR.data.build_paper_samples", lambda *args, **kwargs: samples)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"task": task, "input": str(low_path),
                                            "output": str(extra_path)}) + "\n"
                                for task in ("deblur", "lowlight")))
    dataset = PaperFiveDataset(tmp_path, patch_size=16, special_proxy_manifest=manifest)
    for index in (0, 1):
        item = dataset[index]
        assert item["special_proxy_valid"].item()
        torch.testing.assert_close(item["special_proxy"] - item["lq"],
                                   torch.full_like(item["lq"], 10 / 255), atol=1e-7, rtol=0)
    other = dataset[2]
    assert not other["special_proxy_valid"].item()
    torch.testing.assert_close(other["special_proxy"], other["lq"])


def test_tiled_proxy_render_reconstructs_identity():
    image = torch.rand(1, 3, 23, 29)
    rendered = render_tiled(nn.Identity(), image, tile=16, overlap=4)
    torch.testing.assert_close(rendered, image, atol=2e-7, rtol=0)
