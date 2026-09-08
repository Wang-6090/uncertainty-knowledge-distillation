from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset
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


def make_class_split(all_classes: Sequence, num_known: int, seed: int, split_path: str | None = None):
    if split_path and Path(split_path).exists():
        data = load_json(split_path)
        return data["known_classes"], data["novel_classes"]
    known, novel = split_list(all_classes, num_known, seed)
    if split_path:
        save_json(split_path, {"known_classes": known, "novel_classes": novel})
    return known, novel


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


def split_known_dataset(dataset: Dataset, val_ratio: float, seed: int):
    train_indices, val_indices = split_known_indices(len(dataset), val_ratio, seed)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def split_known_indices(n: int, val_ratio: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_size = max(1, int(n * val_ratio))
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    return train_indices, val_indices


def limit_dataset(dataset: Dataset, max_items: int | None, seed: int):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    return Subset(dataset, perm[:max_items])


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    open_val: Dataset | None
    test: Dataset
    known_classes: Sequence
    novel_classes: Sequence


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
    open_val_ratio: float = 0.0,
) -> DataBundle:
    if dataset_name.lower() == "cifar100":
        known_classes, novel_classes = make_class_split(list(range(100)), num_known, seed, split_path)
        train_tf = build_transforms(image_size, train=True)
        test_tf = build_transforms(image_size, train=False)
        train_full = OpenSetCIFAR100(root, known_classes, train=True, transform=train_tf, download=download, include_unknown=False)
        val_full = OpenSetCIFAR100(root, known_classes, train=True, transform=test_tf, download=download, include_unknown=False)
        test_open = OpenSetCIFAR100(root, known_classes, train=False, transform=test_tf, download=download, include_unknown=True)
        train_indices, val_indices = split_known_indices(len(train_full), val_ratio=0.1, seed=seed)
        train_set = Subset(train_full, train_indices)
        val_set = Subset(val_full, val_indices)
        train_set = limit_dataset(train_set, limit_train, seed)
        val_set = limit_dataset(val_set, limit_val, seed)
        open_val, test_open = split_dataset(test_open, open_val_ratio, seed)
        open_val = limit_dataset(open_val, limit_test, seed) if open_val is not None else None
        test_open = limit_dataset(test_open, limit_test, seed)
        return DataBundle(
            train=train_set,
            val=val_set,
            open_val=open_val,
            test=test_open,
            known_classes=known_classes,
            novel_classes=novel_classes,
        )

    if dataset_name.lower() == "imagefolder":
        base = datasets.ImageFolder(root=root)
        known_classes, novel_classes = make_class_split(base.classes, num_known, seed, split_path)
        train_tf = build_transforms(image_size, train=True)
        test_tf = build_transforms(image_size, train=False)
        train_full = OpenSetImageFolder(root, known_classes, transform=train_tf, include_unknown=False)
        val_full = OpenSetImageFolder(root, known_classes, transform=test_tf, include_unknown=False)
        test_open = OpenSetImageFolder(root, known_classes, transform=test_tf, include_unknown=True)
        train_indices, val_indices = split_known_indices(len(train_full), val_ratio=0.1, seed=seed)
        train_set = Subset(train_full, train_indices)
        val_set = Subset(val_full, val_indices)
        train_set = limit_dataset(train_set, limit_train, seed)
        val_set = limit_dataset(val_set, limit_val, seed)
        open_val, test_open = split_dataset(test_open, open_val_ratio, seed)
        open_val = limit_dataset(open_val, limit_test, seed) if open_val is not None else None
        test_open = limit_dataset(test_open, limit_test, seed)
        return DataBundle(
            train=train_set,
            val=val_set,
            open_val=open_val,
            test=test_open,
            known_classes=known_classes,
            novel_classes=novel_classes,
        )

    if dataset_name.lower() in {"toy", "fake"}:
        known_classes, novel_classes = make_class_split(list(range(100)), num_known, seed, split_path)
        train_tf = build_transforms(image_size, train=True)
        test_tf = build_transforms(image_size, train=False)
        train_full = OpenSetFakeData(1000, known_classes, image_size, transform=train_tf, include_unknown=False, random_offset=0)
        val_full = OpenSetFakeData(200, known_classes, image_size, transform=test_tf, include_unknown=False, random_offset=10000)
        test_open = OpenSetFakeData(1000, known_classes, image_size, transform=test_tf, include_unknown=True, random_offset=20000)
        train_set, _ = split_known_dataset(train_full, val_ratio=0.1, seed=seed)
        train_set = limit_dataset(train_set, limit_train, seed)
        val_set = limit_dataset(val_full, limit_val, seed)
        open_val, test_open = split_dataset(test_open, open_val_ratio, seed)
        open_val = limit_dataset(open_val, limit_test, seed) if open_val is not None else None
        test_open = limit_dataset(test_open, limit_test, seed)
        return DataBundle(
            train=train_set,
            val=val_set,
            open_val=open_val,
            test=test_open,
            known_classes=known_classes,
            novel_classes=novel_classes,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")
