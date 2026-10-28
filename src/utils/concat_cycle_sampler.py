import math

import torch
from torch.utils.data import ConcatDataset, Sampler


class ConcatDatasetCycleSampler(Sampler[int]):
    """
    Samples configurable subsets from a ConcatDataset.
    IMPORTANT: epoch must be advanced manually in pytorch (Pytorch Lightning does this automatically)

    Args:
        dataset: The ConcatDataset to sample from.
        samples_per_epoch: Optional mapping from dataset indices to the number
            of samples per epoch. Unspecified datasets use their full size.
        generator: Optional PyTorch generator for reproducible random sampling.
        pad_last: If True, pad the final incomplete chunk to the configured size
            by repeating samples from the beginning of the current permutation.
            If False, return only the remaining samples.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        samples_per_epoch: dict[int, int] | None = None,
        generator: torch.Generator | None = None,
        pad_last: bool = False,
    ):
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("dataset must be a ConcatDataset")

        samples_per_epoch = samples_per_epoch
        if samples_per_epoch is None:
            samples_per_epoch = {}

        if any(i >= len(dataset.datasets) for i in samples_per_epoch.keys()):
            raise ValueError(f"Cannot set samples_per_epoch for invalid index, make sure you only specify indices < {len(dataset.datasets)} (number of datasets in ConcatDataset)")

        self.dataset = dataset
        self.generator = generator
        self.pad_last = pad_last
        self.epoch = 0

        self.sizes = [len(ds) for ds in dataset.datasets]

        # Unspecified datasets retain their full size.
        self.chunk_sizes = []
        for i, size in enumerate(self.sizes):
            chunk_size = samples_per_epoch.get(i, size)

            if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
                raise TypeError(f"Samples per epoch for dataset {i} must be an integer")
            if chunk_size < 0:
                raise ValueError("Samples cannot be negative")
            if size > 0 and chunk_size == 0:
                raise ValueError(f"Non-empty dataset {i} must have a positive number of samples per epoch")

            self.chunk_sizes.append(min(chunk_size, size))

        self.offsets = [0]
        for size in self.sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)

        self.epochs_per_cycle = [
            math.ceil(size / chunk_size) if chunk_size else 1
            for size, chunk_size in zip(self.sizes, self.chunk_sizes)
        ]

        # [(cycle, permutation), ...]
        self._cycle_permutations = [(None,None)] * len(self.sizes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = 0

        for size, chunk_size, cycle_length in zip(
            self.sizes, self.chunk_sizes, self.epochs_per_cycle
        ):
            # should not happen, checked in __init__
            if chunk_size == 0:
                continue

            section = self.epoch % cycle_length
            start = section * chunk_size
            
            if self.pad_last:
                total += chunk_size
            else:
                total += min(chunk_size, size - start)

        return total

    def __iter__(self):
        indices = []

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # this function is called at the beginning of every epoch in normal training
        # select which indices will be present in this epoch
        for i, (size, chunk_size, offset, cycle_length) in enumerate(
            zip(
                self.sizes,
                self.chunk_sizes,
                self.offsets,
                self.epochs_per_cycle,
            )
        ):
            if chunk_size == 0:
                continue

            cycle, section = divmod(self.epoch, cycle_length)

            perm_cycle, permutation = self._cycle_permutations[i]
            
            if perm_cycle is None or perm_cycle != cycle or permutation is None:
                permutation = torch.randperm(size, generator=generator)
                self._cycle_permutations[i] = (cycle, permutation)

            start = section * chunk_size
            stop = min(start + chunk_size, size)
            local_indices = permutation[start:stop].tolist()

            # padding items are taken from the beginning of the permutation
            # (model has not seen them the longest)
            if self.pad_last and len(local_indices) < chunk_size:
                missing = chunk_size - len(local_indices)
                local_indices.extend(
                    permutation[:missing].tolist()
                )

            # convert local indices to global ConcatDataset indices
            indices.extend(offset + idx for idx in local_indices)

        # shuffle the samples from different datasets
        if len(indices) > 1:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]

        return iter(indices)