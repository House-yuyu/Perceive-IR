from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import pil_to_tensor

from .data import NOISE_SIGMAS, TASKS, Sample, _paired_samples, denoise_target_root, list_images


def build_source_samples(data_root: str | Path) -> list[Sample]:
    root = Path(data_root)
    if (root / "AiO").is_dir():
        root = root / "AiO"
    train_root = root / "train"
    denoise_targets = list_images(denoise_target_root(train_root))
    samples = [
        Sample(None, target, "denoise", NOISE_SIGMAS[index % len(NOISE_SIGMAS)])
        for index, target in enumerate(denoise_targets)
    ]
    locations = {
        "dehaze": ("Dehaze", "input", "gt"),
        "derain": ("Derain", "input", "gt"),
        "deblur": ("Deblur", "input", "gt"),
        "lowlight": ("Enhance", "input", "gt"),
    }
    for task, (folder, lq_name, gt_name) in locations.items():
        base = train_root / folder
        samples.extend(_paired_samples(base / lq_name, base / gt_name, task))
    return samples


def sample_fold(sample: Sample, folds: int = 2) -> int:
    identity = f"{sample.task}|{sample.lq_path or sample.gt_path}".encode("utf-8")
    return int.from_bytes(hashlib.sha1(identity).digest()[:8], "big") % folds


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor(image).float().div_(255.0)


def paired_random_crop(lq: Image.Image, gt: Image.Image, patch: int) -> tuple[torch.Tensor, torch.Tensor]:
    if lq.size != gt.size:
        raise RuntimeError(f"LQ/GT size mismatch: {lq.size} versus {gt.size}")
    width, height = lq.size
    if min(width, height) < patch:
        scale = patch / min(width, height)
        size = (max(patch, round(width * scale)), max(patch, round(height * scale)))
        lq = lq.resize(size, Image.Resampling.BICUBIC)
        gt = gt.resize(size, Image.Resampling.BICUBIC)
        width, height = size
    left = random.randint(0, width - patch)
    top = random.randint(0, height - patch)
    box = (left, top, left + patch, top + patch)
    lq, gt = lq.crop(box), gt.crop(box)
    if random.random() < 0.5:
        lq, gt = lq.transpose(Image.Transpose.FLIP_LEFT_RIGHT), gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        lq, gt = lq.transpose(Image.Transpose.FLIP_TOP_BOTTOM), gt.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return _to_tensor(lq), _to_tensor(gt)


def load_source_pair(sample: Sample, deterministic_noise_seed: int | None = None) -> tuple[Image.Image, Image.Image]:
    gt = Image.open(sample.gt_path).convert("RGB")
    if sample.lq_path is not None:
        return Image.open(sample.lq_path).convert("RGB"), gt
    clean = _to_tensor(gt)
    if deterministic_noise_seed is None:
        noise = torch.randn_like(clean)
    else:
        generator = torch.Generator().manual_seed(deterministic_noise_seed)
        noise = torch.randn(clean.shape, generator=generator)
    lq = (clean * 255.0 + noise * float(sample.noise_sigma)).clamp(0, 255).byte()
    array = lq.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB"), gt


class MediumTrainingDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        heldout_fold: int,
        patch_size: int = 128,
        task_resampling: dict[str, int] | None = None,
    ):
        if heldout_fold not in (0, 1):
            raise ValueError("heldout_fold must be 0 or 1")
        self.patch_size = int(patch_size)
        self.raw_samples = [
            sample for sample in build_source_samples(data_root) if sample_fold(sample) != heldout_fold
        ]
        ratios = {task: 1 for task in TASKS}
        if task_resampling is not None:
            unknown = set(task_resampling) - set(TASKS)
            if unknown:
                raise ValueError(f"unknown resampling tasks: {sorted(unknown)}")
            ratios.update({task: int(value) for task, value in task_resampling.items()})
        if any(value < 1 for value in ratios.values()):
            raise ValueError("all task resampling ratios must be positive integers")
        self.task_resampling = ratios
        self.raw_task_counts = Counter(sample.task for sample in self.raw_samples)
        self.samples = [
            sample
            for sample in self.raw_samples
            for _ in range(self.task_resampling[sample.task])
        ]
        self.effective_task_counts = Counter(sample.task for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        lq, gt = load_source_pair(sample)
        lq_tensor, gt_tensor = paired_random_crop(lq, gt, self.patch_size)
        return {"lq": lq_tensor, "gt": gt_tensor, "task": sample.task}


class HeldoutSourceDataset(Dataset):
    def __init__(self, data_root: str | Path, heldout_fold: int):
        self.samples = [sample for sample in build_source_samples(data_root) if sample_fold(sample) == heldout_fold]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[int, Sample]:
        return index, self.samples[index]


class PromptTripletDataset(Dataset):
    def __init__(self, manifests: list[str | Path], crop_size: int = 224):
        self.crop_size = int(crop_size)
        self.records: list[dict[str, str]] = []
        for manifest in manifests:
            with open(manifest, "r", encoding="utf-8") as handle:
                self.records.extend(json.loads(line) for line in handle if line.strip())
        if not self.records:
            raise RuntimeError("prompt triplet manifests are empty")

    def __len__(self) -> int:
        return len(self.records)

    def _synchronized_crop(self, images: list[Image.Image]) -> list[torch.Tensor]:
        width = min(image.width for image in images)
        height = min(image.height for image in images)
        images = [image.crop((0, 0, width, height)) for image in images]
        crop = self.crop_size
        if min(width, height) < crop:
            scale = crop / min(width, height)
            size = (max(crop, round(width * scale)), max(crop, round(height * scale)))
            images = [image.resize(size, Image.Resampling.BICUBIC) for image in images]
            width, height = size
        left = random.randint(0, width - crop)
        top = random.randint(0, height - crop)
        box = (left, top, left + crop, top + crop)
        images = [image.crop(box) for image in images]
        if random.random() < 0.5:
            images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]
        return [_to_tensor(image) for image in images]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        images = [Image.open(record[key]).convert("RGB") for key in ("low", "medium", "high")]
        low, medium, high = self._synchronized_crop(images)
        return {"low": low, "medium": medium, "high": high}

class TaskBalancedDistributedSampler(Sampler[int]):
    """Sample nearly equal task counts within each global batch, with replacement."""

    def __init__(
        self,
        dataset: PromptTripletDataset,
        batch_size_per_gpu: int,
        world_size: int,
        rank: int,
        seed: int,
    ) -> None:
        self.batch_size_per_gpu = int(batch_size_per_gpu)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size_per_gpu < 1 or self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed sampler batch or rank")
        self.tasks = tuple(TASKS)
        self.indices_by_task = {
            task: torch.tensor(
                [index for index, record in enumerate(dataset.records) if record["task"] == task],
                dtype=torch.long,
            )
            for task in self.tasks
        }
        if any(len(indices) == 0 for indices in self.indices_by_task.values()):
            raise ValueError("task-balanced sampler requires every task in the manifests")
        self.global_batch = self.batch_size_per_gpu * self.world_size
        self.num_global_batches = max(1, (len(dataset) + self.global_batch - 1) // self.global_batch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_global_batches * self.batch_size_per_gpu

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        task_count = len(self.tasks)
        base, extra = divmod(self.global_batch, task_count)
        for batch_number in range(self.num_global_batches):
            selected = []
            for task_number, task in enumerate(self.tasks):
                quota = base + int((task_number - batch_number * extra) % task_count < extra)
                candidates = self.indices_by_task[task]
                offsets = torch.randint(len(candidates), (quota,), generator=generator)
                selected.extend(candidates[offsets].tolist())
            if len(selected) != self.global_batch:
                raise RuntimeError("task-balanced batch size mismatch")
            order = torch.randperm(self.global_batch, generator=generator).tolist()
            global_indices = [selected[index] for index in order]
            yield from global_indices[self.rank :: self.world_size]
