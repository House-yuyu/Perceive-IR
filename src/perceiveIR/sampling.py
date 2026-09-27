from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class GlobalBatchDistributedSampler(Sampler[int]):
    """Partition each shuffled global batch according to unequal rank batch sizes."""

    def __init__(self, dataset_size: int, batch_sizes: Sequence[int], rank: int, seed: int):
        if dataset_size < 1 or not batch_sizes or any(size < 1 for size in batch_sizes):
            raise ValueError("dataset_size and every rank batch size must be positive")
        if not 0 <= rank < len(batch_sizes):
            raise ValueError("invalid rank")
        self.dataset_size = dataset_size
        self.batch_sizes = tuple(batch_sizes)
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.start_step = 0
        self.steps_per_epoch = math.ceil(dataset_size / sum(batch_sizes))

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        if not 0 <= start_step <= self.steps_per_epoch:
            raise ValueError("start_step is outside this epoch")
        self.epoch = epoch
        self.start_step = start_step

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shuffled = torch.randperm(self.dataset_size, generator=generator).tolist()
        total = self.steps_per_epoch * sum(self.batch_sizes)
        padding = total - self.dataset_size
        shuffled.extend((shuffled * math.ceil(padding / self.dataset_size))[:padding])
        offset = sum(self.batch_sizes[:self.rank])
        width = self.batch_sizes[self.rank]
        global_width = sum(self.batch_sizes)
        for step in range(self.start_step, self.steps_per_epoch):
            begin = step * global_width + offset
            yield from shuffled[begin:begin + width]

    def __len__(self) -> int:
        return (self.steps_per_epoch - self.start_step) * self.batch_sizes[self.rank]
