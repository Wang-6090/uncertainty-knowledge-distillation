"""Audit frozen classification-representation stability under CIFAR augmentation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, build_transforms  # noqa: E402
from novel_discovery.models import build_model  # noqa: E402
from novel_discovery.pipeline import build_loader  # noqa: E402
from scripts.audit_teacher_uncertainty_alignment import read_json  # noqa: E402


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = read_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Unsupported checkpoint layout: {checkpoint_path}")
    model = build_model(
        num_classes=len(checkpoint["known_classes"]),
        backbone=config["backbone"],
        proj_dim=int(config.get("proj_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        pretrained=False,
        cifar_stem=bool(config.get("cifar_stem", False)),
        rejection_feature_dim=int(config.get("rejection_feature_dim", 0)),
        independent_rejection_encoder=bool(config.get("independent_rejection_encoder", False)),
        independent_rejection_encoder_init=config.get(
            "independent_rejection_encoder_init", "shared_backbone"
        ),
        independent_rejection_input_norm=config.get("independent_rejection_input_norm", "none"),
        structure_feature_dim=int(config.get("structure_feature_dim", 0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return config, checkpoint, model.to(device).eval()


def _clone_with_train_transform(dataset: Dataset, image_size: int) -> Dataset:
    if isinstance(dataset, Subset):
        cloned = copy.copy(dataset)
        cloned.dataset = _clone_with_train_transform(dataset.dataset, image_size)
        return cloned
    if hasattr(dataset, "base"):
        cloned = copy.copy(dataset)
        cloned.base = copy.copy(dataset.base)
        cloned.base.transform = build_transforms(image_size, train=True, profile="cifar")
        return cloned
    raise TypeError(f"unsupported dataset wrapper: {type(dataset)!r}")


class TwoAugmentedViews(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        first = self.dataset[index]
        second = self.dataset[index]
        if first[3] != 1 or second[3] != 1 or first[1] != second[1] or first[4] != second[4]:
            raise ValueError("paired augmented samples must share known label and source index")
        return first[0], second[0], first[1], first[4]


@torch.inference_mode()
def extract_two_views(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in ("features_a", "features_b", "logits_a", "logits_b", "labels", "indices")
    }
    for image_a, image_b, labels, indices in loader:
        output_a = model(image_a.to(device, non_blocking=True))
        output_b = model(image_b.to(device, non_blocking=True))
        for key, value in {
            "features_a": output_a["features"],
            "features_b": output_b["features"],
            "logits_a": output_a["logits"],
            "logits_b": output_b["logits"],
            "labels": labels,
            "indices": indices,
        }.items():
            chunks[key].append(value.detach().float().cpu().numpy())
    if not chunks["labels"]:
        raise ValueError("cannot audit an empty validation dataset")
    return {key: np.concatenate(values) for key, values in chunks.items()}


def _js_divergence(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    a = torch.from_numpy(logits_a).softmax(dim=-1).clamp_min(1e-12)
    b = torch.from_numpy(logits_b).softmax(dim=-1).clamp_min(1e-12)
    middle = (a + b) * 0.5
    return (
        0.5 * (a * (a.log() - middle.log())).sum(dim=-1)
        + 0.5 * (b * (b.log() - middle.log())).sum(dim=-1)
    ).numpy()


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def summarize_views(values: dict[str, np.ndarray]) -> dict[str, Any]:
    fa, fb = values["features_a"], values["features_b"]
    fa = fa / np.clip(np.linalg.norm(fa, axis=1, keepdims=True), 1e-12, None)
    fb = fb / np.clip(np.linalg.norm(fb, axis=1, keepdims=True), 1e-12, None)
    logits_a, logits_b = values["logits_a"], values["logits_b"]
    labels = values["labels"].astype(np.int64)
    pred_a, pred_b = logits_a.argmax(axis=1), logits_b.argmax(axis=1)
    correct_a, correct_b = pred_a == labels, pred_b == labels
    both_correct = correct_a & correct_b
    either_wrong = ~(correct_a & correct_b)
    feature_cosine = np.sum(fa * fb, axis=1)
    js = _js_divergence(logits_a, logits_b)
    groups = {
        "all": np.ones(labels.shape, dtype=bool),
        "both_views_correct": both_correct,
        "at_least_one_view_wrong": either_wrong,
    }
    return {
        "samples": int(len(labels)),
        "view_a_accuracy": float(correct_a.mean()),
        "view_b_accuracy": float(correct_b.mean()),
        "prediction_flip_rate": float(np.mean(pred_a != pred_b)),
        "groups": {
            name: {
                "count": int(mask.sum()),
                "feature_cosine_mean": _mean(feature_cosine[mask]),
                "feature_cosine_q05": float(np.quantile(feature_cosine[mask], 0.05))
                if mask.any()
                else None,
                "logit_js_mean": _mean(js[mask]),
            }
            for name, mask in groups.items()
        },
    }


def audit(args) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    arms = {}
    reference_config = None
    reference_classes = None
    for arm_name, config_arg, checkpoint_arg in args.arm:
        config, checkpoint, model = load_model(
            Path(config_arg).resolve(), Path(checkpoint_arg).resolve(), device
        )
        if int(config["seed"]) != args.seed:
            raise ValueError(f"{arm_name} config seed does not match requested seed")
        if reference_config is None:
            reference_config, reference_classes = config, checkpoint["known_classes"]
        elif (
            Path(config["split_path"]).resolve() != Path(reference_config["split_path"]).resolve()
            or checkpoint["known_classes"] != reference_classes
        ):
            raise ValueError("all compared arms must share the same split and known-class order")
        split_path = Path(config["split_path"]).resolve()
        bundle = build_data_bundle(
            "cifar100", config["data_root"], int(config["num_known"]), args.seed,
            int(config["image_size"]), split_path=split_path,
            discovery_pool_mode="unknown", transform_profile="cifar",
        )
        augmented = _clone_with_train_transform(bundle.val, int(config["image_size"]))
        paired_loader = DataLoader(
            TwoAugmentedViews(augmented), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )
        torch.manual_seed(args.seed + args.augmentation_seed_offset)
        np.random.seed(args.seed + args.augmentation_seed_offset)
        result = extract_two_views(model, paired_loader, device)
        arms[arm_name] = summarize_views(result)
        arms[arm_name]["source_indices_sha256"] = __import__("hashlib").sha256(
            result["indices"].astype(np.int64).tobytes()
        ).hexdigest()
    return {
        "purpose": "frozen known-validation classification view-stability diagnostic",
        "protocol": {
            "seed": args.seed,
            "split_path": str(Path(reference_config["split_path"]).resolve()),
            "transform": "two independently sampled existing CIFAR train augmentations",
            "labels_used_for": "known-validation post-hoc correctness stratification only",
            "unknown_or_test_labels_loaded": False,
            "used_for_training_or_calibration": False,
        },
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--arm", nargs=3, action="append", metavar=("NAME", "CONFIG", "CHECKPOINT"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augmentation-seed-offset", type=int, default=90210)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    names = [row[0] for row in args.arm]
    if len(names) != len(set(names)):
        raise ValueError("arm names must be unique")
    report = audit(args)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
