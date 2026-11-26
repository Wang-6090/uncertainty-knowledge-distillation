"""Audit how frozen KD students change known-label classification behavior.

This is a read-only Team A diagnostic. It evaluates CE-only, standard-KD, and
uncertainty-KD checkpoints on the same known validation samples, then reports
student NLL, correctness, and margins relative to CE-only. Groups are formed
from the detached teacher's correctness and uncertainty, and from known class
labels. Unknown/test labels are never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, build_transforms  # noqa: E402
from novel_discovery.pipeline import build_loader  # noqa: E402
from scripts.audit_teacher_uncertainty_alignment import (  # noqa: E402
    load_teacher,
    replace_with_eval_transform,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_protocol(
    seed: int,
    split_path: Path,
    teacher_config: dict[str, Any],
    student_configs: dict[str, dict[str, Any]],
) -> None:
    """Reject mismatched data/model protocols before loading any validation data."""
    if int(teacher_config["seed"]) != seed:
        raise ValueError("teacher seed does not match requested seed")
    if resolved_path(teacher_config["split_path"]) != split_path:
        raise ValueError("teacher split does not match requested split")
    if teacher_config.get("transform_profile") != "cifar":
        raise ValueError("teacher must use the CIFAR transform profile")
    fields = (
        "data_root",
        "num_known",
        "image_size",
        "transform_profile",
        "known_split_mode",
        "open_val_ratio",
    )
    for arm, config in student_configs.items():
        if int(config["seed"]) != seed:
            raise ValueError(f"{arm} seed does not match requested seed")
        if resolved_path(config["split_path"]) != split_path:
            raise ValueError(f"{arm} split does not match requested split")
        if config.get("transform_profile") != "cifar":
            raise ValueError(f"{arm} must use the CIFAR transform profile")
        for field in fields:
            if config.get(field) != teacher_config.get(field):
                raise ValueError(f"{arm} differs from teacher on protocol field {field}")
    for field in fields:
        expected = student_configs["ce_only"].get(field)
        for arm, config in student_configs.items():
            if config.get(field) != expected:
                raise ValueError(f"{arm} differs from ce_only on protocol field {field}")


@torch.inference_mode()
def extract(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    logits: list[np.ndarray] = []
    uncertainty: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for images, mapped_labels, _raw_labels, is_known, sample_indices in loader:
        if not bool(is_known.all()):
            raise ValueError("classification audit requires known-only validation data")
        output = model(images.to(device, non_blocking=True))
        logits.append(output["logits"].float().cpu().numpy())
        uncertainty.append(output["uncertainty"].reshape(-1).float().cpu().numpy())
        labels.append(mapped_labels.cpu().numpy())
        indices.append(sample_indices.cpu().numpy())
    if not logits:
        raise ValueError("Cannot audit an empty validation set")
    return {
        "logits": np.concatenate(logits),
        "uncertainty": np.concatenate(uncertainty),
        "labels": np.concatenate(labels),
        "indices": np.concatenate(indices),
    }


def classification_values(logits: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels).long()
    nll = F.cross_entropy(logits_t, labels_t, reduction="none").numpy()
    predicted = logits_t.argmax(dim=1)
    correct = predicted.eq(labels_t).numpy()
    true_logits = logits_t.gather(1, labels_t[:, None]).squeeze(1)
    other_logits = logits_t.clone()
    other_logits.scatter_(1, labels_t[:, None], float("-inf"))
    true_margin = (true_logits - other_logits.max(dim=1).values).numpy()
    top2 = torch.topk(logits_t, k=2, dim=1).values
    top1_margin = (top2[:, 0] - top2[:, 1]).numpy()
    return {
        "nll": nll.astype(np.float64),
        "correct": correct.astype(bool),
        "true_margin": true_margin.astype(np.float64),
        "top1_margin": top1_margin.astype(np.float64),
    }


def mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def summarize(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "accuracy": None, "nll": None, "true_margin": None, "top1_margin": None}
    return {
        "count": count,
        "accuracy": mean(values["correct"][mask].astype(np.float64)),
        "nll": mean(values["nll"][mask]),
        "true_margin": mean(values["true_margin"][mask]),
        "top1_margin": mean(values["top1_margin"][mask]),
    }


def delta(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    return {
        "accuracy": None if summary["accuracy"] is None else summary["accuracy"] - baseline["accuracy"],
        "nll": None if summary["nll"] is None else summary["nll"] - baseline["nll"],
        "true_margin": None if summary["true_margin"] is None else summary["true_margin"] - baseline["true_margin"],
        "top1_margin": None if summary["top1_margin"] is None else summary["top1_margin"] - baseline["top1_margin"],
    }


def quintile_masks(values: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(values, kind="stable")
    return [np.isin(np.arange(len(values)), rows) for rows in np.array_split(order, 5)]


def grouped_report(
    arm_values: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    teacher_error: np.ndarray,
    teacher_uncertainty: np.ndarray,
) -> dict[str, Any]:
    masks: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(labels), dtype=bool)),
        ("teacher_correct", ~teacher_error),
        ("teacher_error", teacher_error),
    ]
    masks.extend(
        (f"uncertainty_q{index}", mask)
        for index, mask in enumerate(quintile_masks(teacher_uncertainty), 1)
    )
    masks.extend((f"class_{int(label)}", labels == label) for label in np.unique(labels))
    baseline = arm_values["ce_only"]
    report: dict[str, Any] = {}
    for name, mask in masks:
        ce_summary = summarize(baseline, mask)
        report[name] = {
            "teacher_uncertainty_mean": mean(teacher_uncertainty[mask]),
            "teacher_error_rate": mean(teacher_error[mask].astype(np.float64)),
            "arms": {
                arm: {
                    "absolute": summarize(values, mask),
                    "delta_vs_ce_only": delta(summarize(values, mask), ce_summary),
                }
                for arm, values in arm_values.items()
            },
        }
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
        if args.device == "auto"
        else args.device
    )
    split_path = resolved_path(args.split_path)
    teacher_config, _teacher_checkpoint, teacher = load_teacher(
        resolved_path(args.teacher_config), resolved_path(args.teacher_checkpoint), device
    )
    student_paths = {
        "ce_only": (args.ce_config, args.ce_checkpoint),
        "standard": (args.standard_config, args.standard_checkpoint),
        "uncertainty": (args.uncertainty_config, args.uncertainty_checkpoint),
    }
    student_configs = {arm: read_json(resolved_path(paths[0])) for arm, paths in student_paths.items()}
    validate_protocol(args.seed, split_path, teacher_config, student_configs)
    bundle = build_data_bundle(
        "cifar100",
        teacher_config["data_root"],
        int(teacher_config["num_known"]),
        int(args.seed),
        int(teacher_config["image_size"]),
        split_path=split_path,
        discovery_pool_mode="unknown",
        transform_profile="cifar",
    )
    replace_with_eval_transform(bundle.train, build_transforms(int(teacher_config["image_size"]), False, "cifar"))
    loader = build_loader(bundle.val, args.batch_size, False, 0)
    teacher_values = extract(teacher, loader, device)
    labels = teacher_values["labels"]
    teacher_logits = teacher_values["logits"]
    teacher_error = teacher_logits.argmax(axis=1) != labels
    teacher_uncertainty = teacher_values["uncertainty"].astype(np.float64)
    arm_values: dict[str, dict[str, np.ndarray]] = {}
    for arm, (config_path, checkpoint_path) in student_paths.items():
        _config, _checkpoint, model = load_teacher(
            resolved_path(config_path), resolved_path(checkpoint_path), device
        )
        values = extract(model, loader, device)
        if not np.array_equal(values["indices"], teacher_values["indices"]):
            raise ValueError(f"{arm} validation sample order differs from teacher")
        if not np.array_equal(values["labels"], labels):
            raise ValueError(f"{arm} validation labels differ from teacher")
        arm_values[arm] = classification_values(values["logits"], labels)
    return {
        "purpose": "frozen known-validation student classification impact diagnostic",
        "protocol": {
            "seed": int(args.seed),
            "split_path": str(split_path),
            "samples": int(len(labels)),
            "groups": ["teacher_correctness", "teacher_uncertainty_quintile", "known_class"],
            "known_labels_used_for": "post-hoc classification metrics and grouping",
            "unknown_test_labels_loaded": False,
            "used_for_training_calibration_or_checkpoint_selection": False,
        },
        "overall": grouped_report(arm_values, labels, teacher_error, teacher_uncertainty)["all"],
        "teacher": {
            "error_rate": float(teacher_error.mean()),
            "uncertainty_mean": float(teacher_uncertainty.mean()),
            "uncertainty_quintile_means": [
                mean(teacher_uncertainty[mask]) for mask in quintile_masks(teacher_uncertainty)
            ],
        },
        "groups": grouped_report(arm_values, labels, teacher_error, teacher_uncertainty),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    for arm in ("ce", "standard", "uncertainty"):
        parser.add_argument(f"--{arm}-config", required=True)
        parser.add_argument(f"--{arm}-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args)
    output = resolved_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
