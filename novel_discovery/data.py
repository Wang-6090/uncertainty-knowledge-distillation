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


def build_view_transforms(image_size: int) -> transforms.Compose:
    """Independent augmentations for unlabeled discovery-pool views."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _ensure_pil(img):
    """Accept PIL images or tensors so toy/FakeData pools reuse the same pipeline."""
    if torch.is_tensor(img):
        img = img.detach().cpu()
        if img.ndim == 3 and img.shape[0] in {1, 3}:
            if img.dtype != torch.uint8:
                img = img.float()
                img = img - img.min()
                denom = img.max().clamp_min(1e-6)
                img = (img / denom * 255.0).clamp(0, 255).to(torch.uint8)
            return transforms.functional.to_pil_image(img)
    return img


class TransformDataset(Dataset):
    """Apply a transform to the image of an already-mapped open-set sample."""

    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, mapped_label, raw_label, is_known, real_index = self.dataset[index]
        if self.transform is not None:
            img = self.transform(_ensure_pil(img))
        return img, mapped_label, raw_label, is_known, index


class TwoViewDataset(Dataset):
    """Return two independently augmented views of the same unlabeled image."""

    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, mapped_label, raw_label, is_known, real_index = self.dataset[index]
        img = _ensure_pil(img)
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, mapped_label, raw_label, is_known, index


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
        include_known: bool = True,
    ) -> None:
        self.base = datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
        self.include_unknown = include_unknown
        self.include_known = include_known
        self.known_classes = list(known_classes) if known_classes is not None else list(range(100))
        self.known_to_idx = {int(cls): i for i, cls in enumerate(self.known_classes)}
        self.allowed_indices = []
        for i, target in enumerate(self.base.targets):
            target = int(target)
            is_known = target in self.known_to_idx
            if (is_known and include_known) or ((not is_known) and include_unknown):
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
        include_known: bool = True,
    ) -> None:
        self.base = datasets.ImageFolder(root=root, transform=transform)
        self.include_unknown = include_unknown
        self.include_known = include_known
        if known_class_names is None:
            known_class_names = self.base.classes
        self.known_class_names = list(known_class_names)
        self.known_to_idx = {name: i for i, name in enumerate(self.known_class_names)}
        self.allowed_indices = []
        for i, (_, raw_label) in enumerate(self.base.samples):
            class_name = self.base.classes[int(raw_label)]
            is_known = class_name in self.known_to_idx
            if (is_known and include_known) or ((not is_known) and include_unknown):
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
        include_known: bool = True,
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
        self.include_known = include_known
        self.known_classes = list(known_classes) if known_classes is not None else list(range(num_classes))
        self.known_to_idx = {int(cls): i for i, cls in enumerate(self.known_classes)}
        self.allowed_indices = []
        for i in range(size):
            _, raw_label = self.base[i]
            raw_label = int(raw_label)
            is_known = raw_label in self.known_to_idx
            if (is_known and include_known) or ((not is_known) and include_unknown):
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
    discovery_pool: Dataset | None
    discovery_pool_eval: Dataset | None
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


def _wrap_discovery_pool(dataset: Dataset | None, image_size: int) -> tuple[Dataset | None, Dataset | None]:
    if dataset is None:
        return None, None
    eval_set = TransformDataset(dataset, build_transforms(image_size, train=False))
    train_set = TwoViewDataset(dataset, build_view_transforms(image_size))
    return train_set, eval_set


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
    include_discovery_pool: bool = False,
    discovery_include_known: bool = True,
    limit_discovery: int | None = None,
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
        discovery_raw = None
        if include_discovery_pool:
            discovery_raw = OpenSetCIFAR100(
                root,
                known_classes,
                train=True,
                transform=None,
                download=download,
                include_unknown=True,
                include_known=discovery_include_known,
            )
            discovery_raw = limit_dataset(discovery_raw, limit_discovery, seed)
        discovery_pool, discovery_pool_eval = _wrap_discovery_pool(discovery_raw, image_size)
        return DataBundle(
            train=train_set,
            val=val_set,
            open_val=open_val,
            test=test_open,
            discovery_pool=discovery_pool,
            discovery_pool_eval=discovery_pool_eval,
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
        discovery_raw = None
        if include_discovery_pool:
            discovery_raw = OpenSetImageFolder(
                root,
                known_classes,
                transform=None,
                include_unknown=True,
                include_known=discovery_include_known,
            )
            discovery_raw = limit_dataset(discovery_raw, limit_discovery, seed)
        discovery_pool, discovery_pool_eval = _wrap_discovery_pool(discovery_raw, image_size)
        return DataBundle(
            train=train_set,
            val=val_set,
            open_val=open_val,
            test=test_open,
            discovery_pool=discovery_pool,
            discovery_pool_eval=discovery_pool_eval,
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
        discovery_raw = None
        if include_discovery_pool:
            discovery_raw = OpenSetFakeData(
                800,
                known_classes,
                image_size,
                transform=None,
                include_unknown=True,
                include_known=discovery_include_known,
                random_offset=30000,
            )
            discovery_raw = limit_dataset(discovery_raw, limit_discovery, seed)
        discovery_pool, discovery_pool_eval = _wrap_discovery_pool(discovery_raw, image_size)
        return DataBundle(
            train=train_set,
            val=val_set,
            open_val=open_val,
            test=test_open,
            discovery_pool=discovery_pool,
            discovery_pool_eval=discovery_pool_eval,
            known_classes=known_classes,
            novel_classes=novel_classes,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")
