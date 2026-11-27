"""Read-only known-validation audit of teacher/CE true-class rank complementarity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import (  # noqa: E402
    OpenSetCIFAR100,
    build_transforms,
    split_known_indices,
)
from scripts.audit_teacher_uncertainty_alignment import load_teacher  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def true_class_ranks(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.size:
        raise ValueError("logits and labels must have aligned [N,C] and [N] shapes")
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels must be non-empty and within the logit class range")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    true_logits = logits[np.arange(labels.size), labels]
    ranks = 1 + np.sum(logits > true_logits[:, None], axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    true_probability = probabilities[np.arange(labels.size), labels]
    return ranks.astype(np.int64), true_probability.astype(np.float64)


def summarize_ranks(ranks: np.ndarray, true_probability: np.ndarray) -> dict[str, Any]:
    ranks = np.asarray(ranks, dtype=np.int64)
    true_probability = np.asarray(true_probability, dtype=np.float64)
    if ranks.ndim != 1 or true_probability.shape != ranks.shape or ranks.size == 0:
        raise ValueError("rank and probability arrays must be non-empty aligned vectors")
    return {
        "count": int(ranks.size),
        "mean_true_class_rank": float(ranks.mean()),
        "median_true_class_rank": float(np.median(ranks)),
        "mean_reciprocal_rank": float((1.0 / ranks).mean()),
        "top1_rate": float((ranks <= 1).mean()),
        "top5_rate": float((ranks <= 5).mean()),
        "top10_rate": float((ranks <= 10).mean()),
        "mean_true_class_probability": float(true_probability.mean()),
    }


@torch.inference_mode()
def extract(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for images, mapped_labels, _raw_labels, is_known, sample_indices in loader:
        if not bool(is_known.all()):
            raise ValueError("rank complementarity audit received a non-known row")
        output = model(images.to(device, non_blocking=True))
        logits.append(output["logits"].float().cpu().numpy())
        labels.append(mapped_labels.cpu().numpy())
        indices.append(sample_indices.cpu().numpy())
    if not logits:
        raise ValueError("known validation loader is empty")
    return {
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels),
        "indices": np.concatenate(indices),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    split_path = resolve_path(args.split_path)
    teacher_config_path = resolve_path(args.teacher_config)
    teacher_checkpoint_path = resolve_path(args.teacher_checkpoint)
    ce_config_path = resolve_path(args.ce_config)
    ce_checkpoint_path = resolve_path(args.ce_checkpoint)
    for path in (split_path, teacher_config_path, teacher_checkpoint_path, ce_config_path, ce_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    split = read_json(split_path)
    known_classes = list(split["known_classes"])
    novel_classes = list(split["novel_classes"])
    if (
        len(known_classes) != 60
        or len(set(known_classes)) != 60
        or len(novel_classes) != 40
        or set(known_classes) & set(novel_classes)
    ):
        raise ValueError("expected a disjoint CIFAR-100 60/40 split")
    if split.get("protocol") != "semantic_hard" or int(split.get("metadata", {}).get("seed", -1)) != args.seed:
        raise ValueError("split metadata does not match the requested semantic-hard seed")

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    teacher_config, teacher_payload, teacher = load_teacher(
        teacher_config_path, teacher_checkpoint_path, device
    )
    ce_config, ce_payload, ce_student = load_teacher(ce_config_path, ce_checkpoint_path, device)
    for name, config in (("teacher", teacher_config), ("ce_only", ce_config)):
        if int(config.get("seed", -1)) != args.seed or resolve_path(config["split_path"]) != split_path:
            raise ValueError(f"{name} config does not match the requested seed/split")
        if config.get("dataset") != "cifar100" or config.get("transform_profile") != "cifar":
            raise ValueError(f"{name} must use CIFAR-100 and the CIFAR transform profile")
        if int(config.get("num_known", -1)) != 60:
            raise ValueError(f"{name} must use 60 known classes")
    if float(ce_config.get("alpha_kd", -1)) != 0.0:
        raise ValueError("student checkpoint must be the CE-only control")
    for name, payload in (("teacher", teacher_payload), ("ce_only", ce_payload)):
        if payload.get("known_classes") != known_classes or payload.get("novel_classes") != novel_classes:
            raise ValueError(f"{name} checkpoint class lists differ from the registered split")

    source_root = Path(teacher_config["data_root"])
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    dataset = OpenSetCIFAR100(
        str(source_root), known_classes, train=True,
        transform=build_transforms(int(teacher_config["image_size"]), False, "cifar"),
        download=False, include_unknown=False,
    )
    _, validation_indices = split_known_indices(len(dataset), 0.1, args.seed)
    if len(validation_indices) != 3000:
        raise ValueError(f"expected 3000 known-validation rows, got {len(validation_indices)}")
    loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    teacher_values = extract(teacher, loader, device)
    ce_values = extract(ce_student, loader, device)
    if not np.array_equal(teacher_values["indices"], ce_values["indices"]):
        raise ValueError("teacher and CE validation sample order differs")
    if not np.array_equal(teacher_values["labels"], ce_values["labels"]):
        raise ValueError("teacher and CE validation labels differ")
    labels = teacher_values["labels"]
    teacher_ranks, teacher_probability = true_class_ranks(teacher_values["logits"], labels)
    ce_ranks, ce_probability = true_class_ranks(ce_values["logits"], labels)
    ce_wrong = ce_values["logits"].argmax(axis=1) != labels
    groups = {}
    for name, mask in (("all", np.ones(len(labels), dtype=bool)), ("ce_error", ce_wrong), ("ce_correct", ~ce_wrong)):
        groups[name] = {
            "teacher": summarize_ranks(teacher_ranks[mask], teacher_probability[mask]),
            "ce_only": summarize_ranks(ce_ranks[mask], ce_probability[mask]),
        }
    rank_delta = ce_ranks[ce_wrong] - teacher_ranks[ce_wrong]
    groups["ce_error"]["paired_teacher_vs_ce_rank"] = {
        "teacher_better_rate": float((rank_delta > 0).mean()),
        "equal_rate": float((rank_delta == 0).mean()),
        "teacher_worse_rate": float((rank_delta < 0).mean()),
        "mean_rank_advantage_ce_minus_teacher": float(rank_delta.mean()),
    }
    return {
        "purpose": "frozen known-validation teacher/CE true-class rank complementarity",
        "protocol": {
            "seed": args.seed,
            "split_path": str(split_path),
            "known_validation_samples": len(labels),
            "test_split_instantiated": False,
            "unknown_or_test_examples_iterated": False,
            "known_labels_used_for": "post-hoc CE-error grouping and true-class rank metrics only",
            "used_for_training_calibration_checkpoint_or_method_selection": False,
        },
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--ce-config", required=True)
    parser.add_argument("--ce-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = resolve_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = audit(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["groups"], indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
