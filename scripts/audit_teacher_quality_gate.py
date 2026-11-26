"""Audit whether formal teachers provide useful known-class KD information.

This is a read-only Team A diagnostic. It evaluates the frozen teacher and
the CE-only, standard-KD, and uncertainty-KD students on the same known train
and validation samples. Unknown/test samples are never loaded, and no result
is used for training, calibration, checkpoint selection, or tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, build_transforms  # noqa: E402
from novel_discovery.pipeline import build_loader  # noqa: E402
from scripts.audit_frozen_student_classification_impact import (  # noqa: E402
    classification_values,
    extract,
    load_teacher,
    read_json,
    resolved_path,
    validate_protocol,
)
from scripts.audit_teacher_uncertainty_alignment import replace_with_eval_transform  # noqa: E402


def summarize(values: dict[str, np.ndarray], sample_count: int) -> dict[str, float]:
    return {
        "samples": int(sample_count),
        "accuracy": float(values["correct"].mean()),
        "nll": float(values["nll"].mean()),
        "true_margin": float(values["true_margin"].mean()),
        "top1_margin": float(values["top1_margin"].mean()),
    }


def error_overlap(
    teacher: dict[str, np.ndarray], student: dict[str, np.ndarray]
) -> dict[str, float]:
    teacher_error = ~teacher["correct"]
    student_error = ~student["correct"]
    return {
        "teacher_correct_student_correct": float((~teacher_error & ~student_error).mean()),
        "teacher_correct_student_error": float((~teacher_error & student_error).mean()),
        "teacher_error_student_correct": float((teacher_error & ~student_error).mean()),
        "teacher_error_student_error": float((teacher_error & student_error).mean()),
        "student_error_rate_on_teacher_correct": float(
            student_error[~teacher_error].mean()
        )
        if (~teacher_error).any()
        else 0.0,
        "student_error_rate_on_teacher_error": float(
            student_error[teacher_error].mean()
        )
        if teacher_error.any()
        else 0.0,
    }


def audit_split(
    models: dict[str, torch.nn.Module], loader, device: torch.device
) -> dict[str, Any]:
    outputs = {name: extract(model, loader, device) for name, model in models.items()}
    reference = outputs["teacher"]
    labels = reference["labels"]
    values = {
        name: classification_values(output["logits"], labels)
        for name, output in outputs.items()
    }
    for name, output in outputs.items():
        if not np.array_equal(output["indices"], reference["indices"]):
            raise ValueError(f"{name} sample order differs from teacher")
        if not np.array_equal(output["labels"], labels):
            raise ValueError(f"{name} labels differ from teacher")
    return {
        "summary": {
            name: summarize(item, len(labels)) for name, item in values.items()
        },
        "error_overlap_vs_teacher": {
            name: error_overlap(values["teacher"], item)
            for name, item in values.items()
            if name != "teacher"
        },
        "teacher_error_rate": float((~values["teacher"]["correct"]).mean()),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
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
    student_configs = {
        arm: read_json(resolved_path(paths[0])) for arm, paths in student_paths.items()
    }
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
    eval_transform = build_transforms(
        int(teacher_config["image_size"]), False, "cifar"
    )
    replace_with_eval_transform(bundle.train, eval_transform)
    replace_with_eval_transform(bundle.val, eval_transform)
    loaders = {
        "known_train": build_loader(bundle.train, args.batch_size, False, 0),
        "known_validation": build_loader(bundle.val, args.batch_size, False, 0),
    }
    models: dict[str, torch.nn.Module] = {"teacher": teacher}
    for arm, (config_path, checkpoint_path) in student_paths.items():
        _config, _checkpoint, model = load_teacher(
            resolved_path(config_path), resolved_path(checkpoint_path), device
        )
        models[arm] = model
    results = {name: audit_split(models, loader, device) for name, loader in loaders.items()}
    for model in models.values():
        del model
    return {
        "purpose": "frozen known-only teacher quality gate; no training or model selection",
        "protocol": {
            "seed": int(args.seed),
            "split_path": str(split_path),
            "unknown_test_labels_loaded": False,
            "used_for_training_calibration_or_checkpoint_selection": False,
            "known_labels_used_for": "post-hoc classification metrics and error overlap",
        },
        "device": str(device),
        "splits": results,
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
    print(json.dumps({"output": str(output), "splits": report["splits"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
