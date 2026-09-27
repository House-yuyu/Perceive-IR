from __future__ import annotations

import random
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TASKS = ("denoise", "dehaze", "derain", "deblur", "lowlight")
TASK_TO_ID = {name: index for index, name in enumerate(TASKS)}
PAPER_REPEAT = {
    "denoise": 3,
    "dehaze": 1,
    "derain": 120,
    "deblur": 5,
    "lowlight": 200,
}
NOISE_SIGMAS = (15, 25, 50)


@dataclass(frozen=True)
class Sample:
    lq_path: Path | None
    gt_path: Path
    task: str
    noise_sigma: int | None = None


def list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def denoise_target_root(train_root: Path) -> Path:
    """Prefer the canonical spelling while accepting existing dataset layouts."""
    canonical = train_root / "Denoise" / "gt"
    return canonical if canonical.is_dir() else train_root / "Denosie" / "gt"


def index_by_stem(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in list_images(root):
        if path.stem in result:
            raise RuntimeError(f"duplicate image stem {path.stem!r} under {root}")
        result[path.stem] = path
    return result


def _paired_samples(input_root: Path, gt_root: Path, task: str) -> list[Sample]:
    targets = index_by_stem(gt_root)
    result: list[Sample] = []
    for lq_path in list_images(input_root):
        if task == "dehaze":
            gt_stem = lq_path.stem.split("_", 1)[0]
        elif task == "derain":
            gt_stem = "norain-" + lq_path.stem.removeprefix("rain-")
        else:
            gt_stem = lq_path.stem
        gt_path = targets.get(gt_stem)
        if gt_path is not None:
            result.append(Sample(lq_path, gt_path, task))
    if not result:
        raise RuntimeError(f"no {task} pairs found under {input_root}")
    return result


def build_paper_samples(
    data_root: str | Path,
    tasks: Iterable[str] = TASKS,
    task_resampling: dict[str, int] | None = None,
) -> list[Sample]:
    root = Path(data_root)
    if (root / "AiO").is_dir():
        root = root / "AiO"
    train_root = root / "train"
    task_set = tuple(tasks)
    unknown = sorted(set(task_set) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    repeats = dict(PAPER_REPEAT)
    if task_resampling is not None:
        repeats.update({task: int(ratio) for task, ratio in task_resampling.items()})
    if any(repeats[task] < 1 for task in task_set):
        raise ValueError("task resampling ratios must be positive")
    if "denoise" in task_set and repeats["denoise"] % len(NOISE_SIGMAS):
        raise ValueError("denoise resampling ratio must be divisible by three noise levels")

    samples: list[Sample] = []
    for task in task_set:
        if task == "denoise":
            targets = list_images(denoise_target_root(train_root))
            if not targets:
                raise RuntimeError("no denoising targets found")
            # The paper uses three noise levels and a denoising resampling ratio of 3.
            denoise_samples = [
                Sample(None, target, task, sigma)
                for target in targets
                for sigma in NOISE_SIGMAS
            ]
            samples.extend(denoise_samples * (repeats["denoise"] // len(NOISE_SIGMAS)))
            continue

        locations = {
            "dehaze": ("Dehaze", "input", "gt"),
            "derain": ("Derain", "input", "gt"),
            "deblur": ("Deblur", "input", "gt"),
            "lowlight": ("Enhance", "input", "gt"),
        }
        folder, lq_name, gt_name = locations[task]
        base = train_root / folder
        paired = _paired_samples(base / lq_name, base / gt_name, task)
        samples.extend(paired * repeats[task])
    return samples


class PaperFiveDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        patch_size: int = 128,
        seed: int = 0,
        tasks: Iterable[str] = TASKS,
        task_resampling: dict[str, int] | None = None,
        proxy_manifests: Iterable[str | Path] = (),
        special_proxy_manifest: str | Path | None = None,
        dynamic_denoise_proxy: bool = False,
        shared_noise_positive: bool = False,
    ):
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.dynamic_denoise_proxy = bool(dynamic_denoise_proxy)
        self.shared_noise_positive = bool(shared_noise_positive)
        if self.patch_size < 8 or self.patch_size % 8:
            raise ValueError("patch_size must be a positive multiple of 8")
        tasks = tuple(tasks)
        self.samples = build_paper_samples(data_root, tasks=tasks, task_resampling=task_resampling)
        self.proxy_lookup: dict[tuple[str, str], Path] = {}
        self.proxy_folds: dict[tuple[str, str], int] = {}
        for manifest in proxy_manifests:
            with open(manifest, "r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record["task"] not in tasks:
                        continue
                    source = record["high"] if record["task"] == "denoise" else record["low"]
                    key = (record["task"], source)
                    self.proxy_lookup[key] = Path(record["medium"])
                    self.proxy_folds[key] = int(record["fold"])
        if self.proxy_lookup:
            missing = [sample for sample in self.samples if self._proxy_key(sample) not in self.proxy_lookup]
            if missing:
                raise RuntimeError(f"missing proxy restorations for {len(missing)} training samples")
        self.special_proxy_lookup: dict[tuple[str, str], Path] = {}
        if special_proxy_manifest is not None:
            with open(special_proxy_manifest, "r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record["task"] not in ("deblur", "lowlight"):
                        raise ValueError("special proxy manifest may only contain deblur/lowlight")
                    key = (record["task"], record["input"])
                    if key in self.special_proxy_lookup:
                        raise RuntimeError(f"duplicate special proxy: {key}")
                    path = Path(record["output"])
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    self.special_proxy_lookup[key] = path
            missing = {self._proxy_key(sample) for sample in self.samples
                       if sample.task in ("deblur", "lowlight")
                       and self._proxy_key(sample) not in self.special_proxy_lookup}
            if missing:
                raise RuntimeError(f"missing task-specific proxy images for {len(missing)} source pairs")

    @staticmethod
    def _proxy_key(sample: Sample) -> tuple[str, str]:
        return sample.task, str(sample.gt_path if sample.lq_path is None else sample.lq_path)

    @property
    def task_counts(self) -> Counter:
        return Counter(sample.task for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        return pil_to_tensor(image).float().div_(255.0)

    def _paired_crop(self, lq: Image.Image, gt: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patch_size
        width, height = gt.size
        if lq.size != gt.size:
            raise RuntimeError(f"LQ/GT size mismatch: {lq.size} versus {gt.size}")
        if width < patch or height < patch:
            scale = patch / min(width, height)
            new_size = (max(patch, round(width * scale)), max(patch, round(height * scale)))
            lq = lq.resize(new_size, Image.Resampling.BICUBIC)
            gt = gt.resize(new_size, Image.Resampling.BICUBIC)
            width, height = new_size
        left = random.randint(0, width - patch)
        top = random.randint(0, height - patch)
        box = (left, top, left + patch, top + patch)
        lq = lq.crop(box)
        gt = gt.crop(box)
        if random.random() < 0.5:
            lq = lq.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            lq = lq.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return self._to_tensor(lq), self._to_tensor(gt)

    def _denoise_crops_from_one_image(
        self, image: Image.Image, sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Crop the CFE pair from one realization of a degraded image."""
        patch = self.patch_size
        width, height = image.size
        if width < patch or height < patch:
            scale = patch / min(width, height)
            image = image.resize(
                (max(patch, round(width * scale)), max(patch, round(height * scale))),
                Image.Resampling.BICUBIC,
            )
            width, height = image.size
        clean = self._to_tensor(image)
        noisy = (clean * 255.0 + torch.randn_like(clean) * sigma).clamp(0, 255) / 255.0

        def crop_pair() -> tuple[torch.Tensor, torch.Tensor]:
            left = random.randint(0, width - patch)
            top = random.randint(0, height - patch)
            low = noisy[:, top:top + patch, left:left + patch]
            high = clean[:, top:top + patch, left:left + patch]
            if random.random() < 0.5:
                low, high = low.flip(-1), high.flip(-1)
            if random.random() < 0.5:
                low, high = low.flip(-2), high.flip(-2)
            return low.contiguous(), high.contiguous()

        low, high = crop_pair()
        positive_low, _ = crop_pair()
        return low, high, positive_low

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        gt_image = Image.open(sample.gt_path).convert("RGB")
        proxy_key = self._proxy_key(sample)
        proxy_path = self.proxy_lookup.get(proxy_key)
        dynamic_proxy = self.dynamic_denoise_proxy and sample.task == "denoise" and proxy_path is not None
        proxy_image = Image.open(proxy_path).convert("RGB") if proxy_path is not None and not dynamic_proxy else None
        if sample.lq_path is None:
            crop_state = random.getstate()
            sigma = float(sample.noise_sigma)
            if self.shared_noise_positive:
                lq, gt, positive_lq = self._denoise_crops_from_one_image(gt_image, sigma)
                if proxy_image is not None:
                    after_crops_state = random.getstate()
                    random.setstate(crop_state)
                    proxy, _ = self._paired_crop(proxy_image, gt_image)
                    random.setstate(after_crops_state)
            else:
                gt, clean = self._paired_crop(gt_image, gt_image.copy())
                if proxy_image is not None:
                    random.setstate(crop_state)
                    proxy, _ = self._paired_crop(proxy_image, gt_image)
                _, positive_clean = self._paired_crop(gt_image, gt_image.copy())
                lq = (clean.mul(255.0).add(torch.randn_like(clean), alpha=sigma)).clamp_(0, 255).div_(255.0)
                positive_lq = (
                    positive_clean.mul(255.0).add(torch.randn_like(positive_clean), alpha=sigma)
                ).clamp_(0, 255).div_(255.0)
        else:
            lq_image = Image.open(sample.lq_path).convert("RGB")
            crop_state = random.getstate()
            lq, gt = self._paired_crop(lq_image, gt_image)
            if proxy_image is not None:
                random.setstate(crop_state)
                proxy, _ = self._paired_crop(proxy_image, gt_image)
            positive_lq, _ = self._paired_crop(lq_image, gt_image)
        result = {
            "lq": lq,
            "positive_lq": positive_lq,
            "gt": gt,
            "task": torch.tensor(TASK_TO_ID[sample.task], dtype=torch.long),
            "task_name": sample.task,
            "noise_sigma": torch.tensor(sample.noise_sigma or 0, dtype=torch.long),
        }
        if proxy_path is not None:
            result["proxy"] = lq.clone() if dynamic_proxy else proxy
            result["proxy_fold"] = torch.tensor(self.proxy_folds[proxy_key], dtype=torch.long)
        if self.special_proxy_lookup:
            special_path = self.special_proxy_lookup.get(proxy_key)
            if special_path is None:
                result["special_proxy"] = lq.clone()
                result["special_proxy_valid"] = torch.tensor(False)
            else:
                after_crops_state = random.getstate()
                random.setstate(crop_state)
                with Image.open(special_path) as opened:
                    special_image = opened.convert("RGB")
                    special_proxy, _ = self._paired_crop(special_image, gt_image)
                random.setstate(after_crops_state)
                result["special_proxy"] = special_proxy
                result["special_proxy_valid"] = torch.tensor(True)
        return result
