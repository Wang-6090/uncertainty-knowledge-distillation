from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms

from .utils import load_json, save_json, split_list


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_outlier_dataset(
    dataset_name: str,
    root: str,
    image_size: int,
    download: bool = False,
) -> Dataset:
    """Build an unlabeled auxiliary dataset for Outlier Exposure.

    Labels are intentionally discarded by the training pipeline. CIFAR-10 is
    provided as a lightweight example; ImageFolder supports a user-supplied
    collection of images from a disjoint source such as Tiny-ImageNet.
    """
    name = dataset_name.lower()
    transform = build_transforms(image_size, train=True)
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=True, transform=transform, download=download)
    if name == "imagefolder":
        return datasets.ImageFolder(root=root, transform=transform)
    raise ValueError(f"Unsupported outlier dataset: {dataset_name}")


def make_class_split(all_classes: Sequence, num_known: int, seed: int, split_path: str | None = None):
    all_classes = list(all_classes)
    if split_path and Path(split_path).exists():
        data = load_json(split_path)
        known = data.get("known_classes")
        novel = data.get("novel_classes")
        validate_class_split(all_classes, known, novel, num_known=num_known)
        return known, novel
    known, novel = split_list(all_classes, num_known, seed)
    validate_class_split(all_classes, known, novel, num_known=num_known)
    if split_path:
        save_json(split_path, {"known_classes": known, "novel_classes": novel})
    return known, novel


def validate_class_split(
    all_classes: Sequence,
    known_classes: Sequence | None,
    novel_classes: Sequence | None,
    num_known: int | None = None,
) -> None:
    """Validate an open-set class protocol before building any datasets.

    A split file is part of the experiment definition. Failing early here is
    preferable to silently training with duplicated, missing, or mismatched
    classes and then comparing invalid results.
    """
    if known_classes is None or novel_classes is None:
        raise ValueError("Split file must contain known_classes and novel_classes")
    all_set = set(all_classes)
    known = list(known_classes)
    novel = list(novel_classes)
    if len(set(known)) != len(known):
        raise ValueError("known_classes contains duplicates")
    if len(set(novel)) != len(novel):
        raise ValueError("novel_classes contains duplicates")
    if set(known) & set(novel):
        raise ValueError("known_classes and novel_classes must be disjoint")
    if set(known) | set(novel) != all_set:
        missing = sorted(all_set - (set(known) | set(novel)), key=str)
        extra = sorted((set(known) | set(novel)) - all_set, key=str)
        raise ValueError(f"Class split must cover all classes; missing={missing}, extra={extra}")
    if num_known is not None and len(known) != int(num_known):
        raise ValueError(
            f"Split has {len(known)} known classes, but num_known={int(num_known)}"
        )


class OpenSetCIFAR100(Dataset):
    def __init__(
        self,
        root: str,
        known_classes: Sequence[int] | None,
        train: bool,
        transform=None,
        download: bool = False,
        include_unknown: bool = True,
    ) -> None:
        self.base = datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
        self.include_unknown = include_unknown
        self.known_classes = list(known_classes) if known_classes is not None else list(range(100))
        self.known_to_idx = {int(cls): i for i, cls in enumerate(self.known_classes)}
        self.allowed_indices = []
        for i, target in enumerate(self.base.targets):
            target = int(target)
            if include_unknown or target in self.known_to_idx:
                self.allowed_indices.append(i)

    def __len__(self) -> int:
        return len(self.allowed_indices)

    def __getitem__(self, index: int):
        real_index = self.allowed_indices[index]
        img, raw_label = self.base[real_index]
        raw_label = int(raw_label)
        mapped_label = self.known_to_idx.get(raw_label, -1)
        is_known = 1 if mapped_label >= 0 else 0
        return img, mapped_label, raw_label, is_known, real_index


class OpenSetImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        known_class_names: Sequence[str] | None,
        transform=None,
        include_unknown: bool = True,
    ) -> None:
        self.base = datasets.ImageFolder(root=root, transform=transform)
        self.include_unknown = include_unknown
        if known_class_names is None:
            known_class_names = self.base.classes
        self.known_class_names = list(known_class_names)
        self.known_to_idx = {name: i for i, name in enumerate(self.known_class_names)}
        self.allowed_indices = []
        for i, (_, raw_label) in enumerate(self.base.samples):
            class_name = self.base.classes[int(raw_label)]
            if include_unknown or class_name in self.known_to_idx:
                self.allowed_indices.append(i)

    def __len__(self) -> int:
        return len(self.allowed_indices)

    def __getitem__(self, index: int):
        real_index = self.allowed_indices[index]
        img, raw_label = self.base[real_index]
        raw_label = int(raw_label)
        class_name = self.base.classes[raw_label]
        mapped_label = self.known_to_idx.get(class_name, -1)
        is_known = 1 if mapped_label >= 0 else 0
        return img, mapped_label, raw_label, is_known, real_index


class OpenSetFakeData(Dataset):
    def __init__(
        self,
        size: int,
        known_classes: Sequence[int] | None,
        image_size: int,
        transform=None,
        include_unknown: bool = True,
        num_classes: int = 100,
        random_offset: int = 0,
    ) -> None:
        self.base = datasets.FakeData(
            size=size,
            image_size=(3, image_size, image_size),
            num_classes=num_classes,
            transform=transform,
            random_offset=random_offset,
        )
        self.include_unknown = include_unknown
        self.known_classes = list(known_classes) if known_classes is not None else list(range(num_classes))
        self.known_to_idx = {int(cls): i for i, cls in enumerate(self.known_classes)}
        self.allowed_indices = []
        for i in range(size):
            _, raw_label = self.base[i]
            raw_label = int(raw_label)
            if include_unknown or raw_label in self.known_to_idx:
                self.allowed_indices.append(i)

    def __len__(self) -> int:
        return len(self.allowed_indices)

    def __getitem__(self, index: int):
        real_index = self.allowed_indices[index]
        img, raw_label = self.base[real_index]
        raw_label = int(raw_label)
        mapped_label = self.known_to_idx.get(raw_label, -1)
        is_known = 1 if mapped_label >= 0 else 0
        return img, mapped_label, raw_label, is_known, real_index


def _known_labels(dataset: Dataset) -> list[int]:
    """Return mapped known labels without relying on image transforms."""
    if isinstance(dataset, Subset):
        base_labels = _known_labels(dataset.dataset)
        return [base_labels[int(index)] for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        labels: list[int] = []
        for child in dataset.datasets:
            labels.extend(_known_labels(child))
        return labels
    if hasattr(dataset, "allowed_indices") and hasattr(dataset, "base"):
        allowed_indices = list(dataset.allowed_indices)
        if hasattr(dataset.base, "targets"):
            return [
                int(dataset.known_to_idx[int(dataset.base.targets[index])])
                for index in allowed_indices
            ]
        if hasattr(dataset.base, "samples"):
            return [
                int(dataset.known_to_idx[dataset.base.classes[int(dataset.base.samples[index][1])]])
                for index in allowed_indices
            ]
    # FakeData and custom datasets have no stable raw-label table. Their
    # labels are still cheap to read and are independent of image transforms.
    return [int(dataset[index][1]) for index in range(len(dataset))]


def known_proportion(dataset: Dataset) -> float | None:
    """Return the known-sample fraction of an open-set dataset when available.

    This is dataset-protocol metadata, not per-example supervision.  It lets
    PU experiments use the actual composition after sampling/limiting instead
    of silently assuming that the requested mixed-pool ratio survived those
    operations unchanged.
    """
    if dataset is None or len(dataset) == 0:
        return None
    if isinstance(dataset, Subset):
        flags = known_proportion(dataset.dataset)
        if flags is None:
            return None
        # A subset can change the composition, so inspect its selected rows.
        base_flags = _known_flags(dataset.dataset)
        return float(sum(base_flags[int(index)] for index in dataset.indices)) / len(dataset)
    if isinstance(dataset, ConcatDataset):
        child_sizes = [len(child) for child in dataset.datasets]
        child_props = [known_proportion(child) for child in dataset.datasets]
        if any(prop is None for prop in child_props):
            return None
        known_count = sum(size * float(prop) for size, prop in zip(child_sizes, child_props))
        return float(known_count) / len(dataset)
    flags = _known_flags(dataset)
    return float(sum(flags)) / len(flags) if flags else None


def _known_flags(dataset: Dataset) -> list[bool]:
    """Read known/unknown protocol flags without applying image transforms."""
    if isinstance(dataset, Subset):
        base_flags = _known_flags(dataset.dataset)
        return [base_flags[int(index)] for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        flags: list[bool] = []
        for child in dataset.datasets:
            flags.extend(_known_flags(child))
        return flags
    if hasattr(dataset, "allowed_indices") and hasattr(dataset, "base"):
        allowed_indices = list(dataset.allowed_indices)
        known_to_idx = getattr(dataset, "known_to_idx", {})
        if hasattr(dataset.base, "targets"):
            return [int(dataset.base.targets[index]) in known_to_idx for index in allowed_indices]
        if hasattr(dataset.base, "samples"):
            return [
                dataset.base.classes[int(dataset.base.samples[index][1])] in known_to_idx
                for index in allowed_indices
            ]
    return [bool(dataset[index][3]) for index in range(len(dataset))]


def split_known_dataset(dataset: Dataset, val_ratio: float, seed: int, mode: str = "random"):
    labels = _known_labels(dataset) if mode == "stratified" else None
    train_indices, val_indices = split_known_indices(len(dataset), val_ratio, seed, labels=labels)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def split_validation_for_calibration(
    dataset: Dataset,
    calibration_ratio: float,
    seed: int,
    mode: str = "stratified",
    overlap_control: bool = False,
):
    """Split known validation data for model selection and threshold calibration.

    With ``overlap_control=True``, calibration has the same size as the
    reserved holdout would have, but is sampled from the selection subset.
    This is intended for controlled experiments on calibration/selection
    overlap, not for the default evaluation protocol.
    """
    ratio = float(calibration_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("calibration_ratio must be in [0, 1)")
    if ratio == 0.0:
        if overlap_control:
            raise ValueError("overlap_control requires a non-zero calibration_ratio")
        return dataset, None
    if mode not in {"random", "stratified"}:
        raise ValueError("calibration split mode must be 'random' or 'stratified'")
    labels = _known_labels(dataset) if mode == "stratified" else None
    selection_indices, calibration_indices = split_known_indices(
        len(dataset), ratio, seed, labels=labels
    )
    if overlap_control:
        if len(calibration_indices) > len(selection_indices):
            raise ValueError(
                "overlap_control requires calibration_ratio <= 0.5 so the "
                "selection subset can supply an equal-size calibration subset"
            )
        generator = torch.Generator().manual_seed(seed + 1)
        order = torch.randperm(len(selection_indices), generator=generator).tolist()
        calibration_indices = [
            selection_indices[index] for index in order[: len(calibration_indices)]
        ]
    return Subset(dataset, selection_indices), Subset(dataset, calibration_indices)


def split_known_for_discovery(
    dataset: Dataset,
    pool_ratio: float,
    seed: int,
    mode: str = "random",
):
    """Reserve known samples for an unlabeled mixed pool without sample overlap.

    The returned pair is ``(supervised_train, unlabeled_known_pool)``.  Both
    subsets are drawn only from the known training partition, so validation
    and test examples remain untouched.
    """
    ratio = float(pool_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("mixed_known_pool_ratio must be in [0, 1)")
    if ratio == 0.0:
        return dataset, None
    labels = _known_labels(dataset) if mode == "stratified" else None
    supervised_indices, pool_indices = split_known_indices(
        len(dataset), val_ratio=ratio, seed=seed, labels=labels
    )
    return Subset(dataset, supervised_indices), Subset(dataset, pool_indices)


def split_known_indices(
    n: int,
    val_ratio: float,
    seed: int,
    labels: Sequence[int] | None = None,
):
    if labels is not None:
        if len(labels) != n:
            raise ValueError("labels must have the same length as the dataset")
        groups: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(int(label), []).append(index)
        rng = torch.Generator().manual_seed(seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        for label in sorted(groups):
            group = torch.tensor(groups[label], dtype=torch.long)
            order = torch.randperm(len(group), generator=rng).tolist()
            shuffled = group[order].tolist()
            if len(shuffled) <= 1:
                val_size = 0
            else:
                val_size = min(len(shuffled) - 1, max(1, int(len(shuffled) * val_ratio)))
            val_indices.extend(shuffled[:val_size])
            train_indices.extend(shuffled[val_size:])
        if not train_indices or not val_indices:
            raise ValueError("Stratified split needs at least two total samples")
        return train_indices, val_indices
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_size = max(1, int(n * val_ratio))
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    return train_indices, val_indices


def limit_dataset(
    dataset: Dataset,
    max_items: int | None,
    seed: int,
    labels: Sequence[int] | None = None,
):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    if labels is not None:
        if len(labels) != len(dataset):
            raise ValueError("labels must have the same length as the dataset")
        groups: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(int(label), []).append(index)
        generator = torch.Generator().manual_seed(seed)
        group_ids = sorted(groups)
        if max_items < len(group_ids):
            group_order = torch.randperm(len(group_ids), generator=generator).tolist()
            chosen_groups = [group_ids[index] for index in group_order[:max_items]]
            selected = []
            for label in chosen_groups:
                order = torch.randperm(len(groups[label]), generator=generator).tolist()
                selected.append(groups[label][order[0]])
            return Subset(dataset, selected)

        selected: list[int] = []
        remaining: list[int] = []
        for label in group_ids:
            order = torch.randperm(len(groups[label]), generator=generator).tolist()
            selected.append(groups[label][order[0]])
            remaining.extend(groups[label][index] for index in order[1:])
        if len(selected) < max_items:
            order = torch.randperm(len(remaining), generator=generator).tolist()
            selected.extend(remaining[index] for index in order[: max_items - len(selected)])
        return Subset(dataset, selected)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    return Subset(dataset, perm[:max_items])


def limit_known_dataset(dataset: Dataset, max_items: int | None, seed: int, mode: str):
    labels = _known_labels(dataset) if mode == "stratified" and max_items else None
    return limit_dataset(dataset, max_items, seed, labels=labels)


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    open_val: Dataset | None
    test: Dataset
    known_classes: Sequence
    novel_classes: Sequence
    discovery_pool: Dataset | None = None
    calibration: Dataset | None = None


class TwoViewDataset(Dataset):
    """Return two independently augmented views from one base sample."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        first = self.dataset[index]
        second = self.dataset[index]
        return first[0], second[0]


def unknown_subset(dataset: Dataset) -> Dataset:
    """Build an unlabeled view of the samples outside the known classes."""
    if hasattr(dataset, "allowed_indices") and hasattr(dataset, "base"):
        allowed_indices = list(dataset.allowed_indices)
        if hasattr(dataset.base, "targets"):
            raw_targets = dataset.base.targets
            indices = [
                local_index
                for local_index, real_index in enumerate(allowed_indices)
                if int(raw_targets[real_index]) not in dataset.known_to_idx
            ]
            return Subset(dataset, indices)
        if hasattr(dataset.base, "samples"):
            indices = []
            for local_index, real_index in enumerate(allowed_indices):
                raw_label = int(dataset.base.samples[real_index][1])
                class_name = dataset.base.classes[raw_label]
                if class_name not in dataset.known_to_idx:
                    indices.append(local_index)
            return Subset(dataset, indices)

    indices = []
    for index in range(len(dataset)):
        if int(dataset[index][3]) == 0:
            indices.append(index)
    return Subset(dataset, indices)


def split_dataset(dataset: Dataset, ratio: float, seed: int):
    if ratio <= 0.0 or len(dataset) <= 1:
        return None, dataset
    ratio = min(max(ratio, 0.0), 0.9)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    split = max(1, int(len(dataset) * ratio))
    first = Subset(dataset, perm[:split])
    second = Subset(dataset, perm[split:])
    return first, second


def build_open_validation_and_discovery(
    known_train: Dataset,
    known_val: Dataset,
    unknown_pool: Dataset,
    open_val_ratio: float,
    seed: int,
    discovery_pool_mode: str,
    unknown_open_val_pool: Dataset | None = None,
    known_discovery_pool: Dataset | None = None,
):
    """Reserve training-split samples for open validation, never from test.

    The known validation set is divided between threshold calibration and
    open-score selection. Novel training samples are divided between
    open-score selection and the unlabeled discovery pool.
    """
    if discovery_pool_mode not in {"unknown", "mixed"}:
        raise ValueError(f"Unsupported discovery pool mode: {discovery_pool_mode}")
    if open_val_ratio > 0.0:
        open_known, known_val = split_dataset(known_val, open_val_ratio, seed)
        open_unknown_train, unknown_pool = split_dataset(unknown_pool, open_val_ratio, seed + 1)
        if unknown_open_val_pool is not None:
            if len(unknown_open_val_pool) != len(unknown_pool) + len(open_unknown_train):
                raise ValueError("evaluation and training unknown pools must have matching samples")
            open_unknown, _ = split_dataset(unknown_open_val_pool, open_val_ratio, seed + 1)
        else:
            open_unknown = open_unknown_train
        open_val = ConcatDataset([open_known, open_unknown])
    else:
        open_val = None

    if discovery_pool_mode == "unknown":
        discovery_pool = unknown_pool
    else:
        if known_discovery_pool is None:
            raise ValueError(
                "mixed discovery requires a disjoint known_discovery_pool; "
                "do not reuse labeled known_train samples as unlabeled data"
            )
        discovery_pool = ConcatDataset([known_discovery_pool, unknown_pool])
    return known_val, open_val, discovery_pool


def build_data_bundle(
    dataset_name: str,
    root: str,
    num_known: int,
    seed: int,
    image_size: int,
    download: bool = False,
    split_path: str | None = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
    limit_test: int | None = None,
    limit_discovery: int | None = None,
    discovery_pool_mode: str = "unknown",
    open_val_ratio: float = 0.0,
    calibration_ratio: float = 0.0,
    calibration_overlap_control: bool = False,
    known_split_mode: str = "random",
    mixed_known_pool_ratio: float = 0.2,
) -> DataBundle:
    if discovery_pool_mode not in {"unknown", "mixed"}:
        raise ValueError(f"Unsupported discovery pool mode: {discovery_pool_mode}")
    if known_split_mode not in {"random", "stratified"}:
        raise ValueError(f"Unsupported known split mode: {known_split_mode}")
    if not 0.0 <= float(calibration_ratio) < 1.0:
        raise ValueError("calibration_ratio must be in [0, 1)")
    if discovery_pool_mode == "mixed" and not 0.0 < float(mixed_known_pool_ratio) < 1.0:
        raise ValueError(
            "mixed discovery requires 0 < mixed_known_pool_ratio < 1 so the "
            "known portion is disjoint from supervised training"
        )

    if dataset_name.lower() == "cifar100":
        known_classes, novel_classes = make_class_split(list(range(100)), num_known, seed, split_path)
        train_tf = build_transforms(image_size, train=True)
        test_tf = build_transforms(image_size, train=False)
        train_full = OpenSetCIFAR100(root, known_classes, train=True, transform=train_tf, download=download, include_unknown=False)
        val_full = OpenSetCIFAR100(root, known_classes, train=True, transform=test_tf, download=download, include_unknown=False)
        pool_full = OpenSetCIFAR100(root, known_classes, train=True, transform=train_tf, download=download, include_unknown=True)
        pool_eval = copy.copy(pool_full)
        pool_eval.base = copy.copy(pool_full.base)
        pool_eval.base.transform = test_tf
        test_open = OpenSetCIFAR100(root, known_classes, train=False, transform=test_tf, download=download, include_unknown=True)
        labels = _known_labels(train_full) if known_split_mode == "stratified" else None
        train_indices, val_indices = split_known_indices(
            len(train_full), val_ratio=0.1, seed=seed, labels=labels
        )
        train_set = Subset(train_full, train_indices)
        val_set = Subset(val_full, val_indices)
        known_discovery_pool = None
        if discovery_pool_mode == "mixed":
            train_set, known_discovery_pool = split_known_for_discovery(
                train_set,
                mixed_known_pool_ratio,
                seed + 2,
                known_split_mode,
            )
        unknown_pool = unknown_subset(pool_full)
        unknown_open_val_pool = unknown_subset(pool_eval)
        val_set, open_val, discovery_pool = build_open_validation_and_discovery(
            train_set,
            val_set,
            unknown_pool,
            open_val_ratio,
            seed,
            discovery_pool_mode,
            unknown_open_val_pool=unknown_open_val_pool,
            known_discovery_pool=known_discovery_pool,
        )
        train_set = limit_known_dataset(train_set, limit_train, seed, known_split_mode)
        val_set = limit_known_dataset(val_set, limit_val, seed, known_split_mode)
        val_set, calibration_set = split_validation_for_calibration(
            val_set,
            calibration_ratio,
            seed + 17,
            mode="stratified",
            overlap_control=calibration_overlap_control,
        )
        open_val = limit_dataset(open_val, limit_test, seed) if open_val is not None else None
        test_open = limit_dataset(test_open, limit_test, seed)
        return DataBundle(
            train=train_set,
            val=val_set,
            calibration=calibration_set,
            open_val=open_val,
            test=test_open,
            known_classes=known_classes,
            novel_classes=novel_classes,
            discovery_pool=limit_dataset(discovery_pool, limit_discovery, seed),
        )

    if dataset_name.lower() == "imagefolder":
        root_path = Path(root)
        has_train_dir = (root_path / "train").is_dir()
        has_test_dir = (root_path / "test").is_dir()
        if not has_train_dir or not has_test_dir:
            raise ValueError(
                "ImageFolder evaluation requires separate root/train and root/test directories "
                "to prevent train/test sample overlap."
            )
        train_root = root_path / "train"
        test_root = root_path / "test"
        base = datasets.ImageFolder(root=str(train_root))
        imagefolder_split = None
        if split_path:
            split_file = Path(split_path)
            imagefolder_split = str(split_file.with_name(split_file.stem + "_imagefolder.json"))
        known_classes, novel_classes = make_class_split(base.classes, num_known, seed, imagefolder_split)
        train_tf = build_transforms(image_size, train=True)
        test_tf = build_transforms(image_size, train=False)
        train_full = OpenSetImageFolder(str(train_root), known_classes, transform=train_tf, include_unknown=False)
        val_full = OpenSetImageFolder(str(train_root), known_classes, transform=test_tf, include_unknown=False)
        pool_full = OpenSetImageFolder(str(train_root), known_classes, transform=train_tf, include_unknown=True)
        pool_eval = copy.copy(pool_full)
        pool_eval.base = copy.copy(pool_full.base)
        pool_eval.base.transform = test_tf
        test_open = OpenSetImageFolder(str(test_root), known_classes, transform=test_tf, include_unknown=True)
        labels = _known_labels(train_full) if known_split_mode == "stratified" else None
        train_indices, val_indices = split_known_indices(
            len(train_full), val_ratio=0.1, seed=seed, labels=labels
        )
        train_set = Subset(train_full, train_indices)
        val_set = Subset(val_full, val_indices)
        known_discovery_pool = None
        if discovery_pool_mode == "mixed":
            train_set, known_discovery_pool = split_known_for_discovery(
                train_set,
                mixed_known_pool_ratio,
                seed + 2,
                known_split_mode,
            )
        unknown_pool = unknown_subset(pool_full)
        unknown_open_val_pool = unknown_subset(pool_eval)
        val_set, open_val, discovery_pool = build_open_validation_and_discovery(
            train_set,
            val_set,
            unknown_pool,
            open_val_ratio,
            seed,
            discovery_pool_mode,
            unknown_open_val_pool=unknown_open_val_pool,
            known_discovery_pool=known_discovery_pool,
        )
        train_set = limit_known_dataset(train_set, limit_train, seed, known_split_mode)
        val_set = limit_known_dataset(val_set, limit_val, seed, known_split_mode)
        val_set, calibration_set = split_validation_for_calibration(
            val_set,
            calibration_ratio,
            seed + 17,
            mode="stratified",
            overlap_control=calibration_overlap_control,
        )
        open_val = limit_dataset(open_val, limit_test, seed) if open_val is not None else None
        test_open = limit_dataset(test_open, limit_test, seed)
        return DataBundle(
            train=train_set,
            val=val_set,
            calibration=calibration_set,
            open_val=open_val,
            test=test_open,
            known_classes=known_classes,
            novel_classes=novel_classes,
            discovery_pool=limit_dataset(discovery_pool, limit_discovery, seed),
        )

    if dataset_name.lower() in {"toy", "fake"}:
        known_classes, novel_classes = make_class_split(list(range(100)), num_known, seed, split_path)
        train_tf = build_transforms(image_size, train=True)
        test_tf = build_transforms(image_size, train=False)
        train_full = OpenSetFakeData(1000, known_classes, image_size, transform=train_tf, include_unknown=False, random_offset=0)
        val_full = OpenSetFakeData(200, known_classes, image_size, transform=test_tf, include_unknown=False, random_offset=10000)
        pool_full = OpenSetFakeData(1000, known_classes, image_size, transform=train_tf, include_unknown=True, random_offset=30000)
        pool_eval = copy.copy(pool_full)
        pool_eval.base = copy.copy(pool_full.base)
        pool_eval.base.transform = test_tf
        test_open = OpenSetFakeData(1000, known_classes, image_size, transform=test_tf, include_unknown=True, random_offset=20000)
        train_set, _ = split_known_dataset(
            train_full, val_ratio=0.1, seed=seed, mode=known_split_mode
        )
        known_discovery_pool = None
        if discovery_pool_mode == "mixed":
            train_set, known_discovery_pool = split_known_for_discovery(
                train_set,
                mixed_known_pool_ratio,
                seed + 2,
                known_split_mode,
            )
        val_set = limit_dataset(val_full, limit_val, seed)
        unknown_pool = unknown_subset(pool_full)
        unknown_open_val_pool = unknown_subset(pool_eval)
        val_set, open_val, discovery_pool = build_open_validation_and_discovery(
            train_set,
            val_set,
            unknown_pool,
            open_val_ratio,
            seed,
            discovery_pool_mode,
            unknown_open_val_pool=unknown_open_val_pool,
            known_discovery_pool=known_discovery_pool,
        )
        train_set = limit_known_dataset(train_set, limit_train, seed, known_split_mode)
        val_set = limit_known_dataset(val_set, limit_val, seed, known_split_mode)
        val_set, calibration_set = split_validation_for_calibration(
            val_set,
            calibration_ratio,
            seed + 17,
            mode="stratified",
            overlap_control=calibration_overlap_control,
        )
        open_val = limit_dataset(open_val, limit_test, seed) if open_val is not None else None
        test_open = limit_dataset(test_open, limit_test, seed)
        return DataBundle(
            train=train_set,
            val=val_set,
            calibration=calibration_set,
            open_val=open_val,
            test=test_open,
            known_classes=known_classes,
            novel_classes=novel_classes,
            discovery_pool=limit_dataset(discovery_pool, limit_discovery, seed),
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")
