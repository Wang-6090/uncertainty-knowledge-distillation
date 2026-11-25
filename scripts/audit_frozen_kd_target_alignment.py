"""Audit frozen student-teacher KD error by teacher uncertainty groups."""

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
    return json.loads(path.read_text(encoding="utf-8"))


@torch.inference_mode()
def extract(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    logits: list[np.ndarray] = []
    uncertainty: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for images, mapped_labels, _raw_labels, is_known, sample_indices in loader:
        if not bool(is_known.all()):
            raise ValueError("KD alignment audit requires known-only validation data")
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


def per_sample_kl(student_logits: np.ndarray, teacher_logits: np.ndarray, temperature: float) -> np.ndarray:
    student = torch.from_numpy(student_logits)
    teacher = torch.from_numpy(teacher_logits)
    teacher_prob = F.softmax(teacher / temperature, dim=-1)
    student_log_prob = F.log_softmax(student / temperature, dim=-1)
    return (
        F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1)
        * temperature**2
    ).numpy()


def mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def grouped(values: np.ndarray, error: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
    order = np.argsort(uncertainty, kind="stable")
    bins = []
    for index, rows in enumerate(np.array_split(order, 5)):
        bins.append(
            {
                "bin": int(index + 1),
                "count": int(len(rows)),
                "uncertainty_mean": mean_or_none(uncertainty[rows]),
                "teacher_error_rate": mean_or_none(error[rows].astype(np.float64)),
                "student_teacher_kl_mean": mean_or_none(values[rows]),
            }
        )
    return {
        "all": {
            "count": int(len(values)),
            "kl_mean": mean_or_none(values),
            "kl_median": float(np.median(values)),
        },
        "teacher_correct": {
            "count": int((~error).sum()),
            "kl_mean": mean_or_none(values[~error]),
        },
        "teacher_error": {
            "count": int(error.sum()),
            "kl_mean": mean_or_none(values[error]),
        },
        "uncertainty_quintiles": bins,
    }


def audit(args) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    teacher_config_path = Path(args.teacher_config).resolve()
    teacher_checkpoint_path = Path(args.teacher_checkpoint).resolve()
    teacher_config, teacher_checkpoint, teacher = load_teacher(
        teacher_config_path, teacher_checkpoint_path, device
    )
    split_path = Path(args.split_path).resolve()
    if Path(teacher_config["split_path"]).resolve() != split_path:
        raise ValueError("teacher config and requested split differ")
    if int(teacher_config["seed"]) != int(args.seed):
        raise ValueError("teacher config seed mismatch")
    if teacher_config.get("transform_profile") != "cifar":
        raise ValueError("this audit requires the CIFAR transform profile")

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
    replace_with_eval_transform(
        bundle.train,
        build_transforms(int(teacher_config["image_size"]), False, "cifar"),
    )
    loader = build_loader(bundle.val, args.batch_size, False, 0)
    teacher_values = extract(teacher, loader, device)

    student_values = {}
    for arm, config_path, checkpoint_path in (
        ("standard", Path(args.standard_config), Path(args.standard_checkpoint)),
        ("uncertainty", Path(args.uncertainty_config), Path(args.uncertainty_checkpoint)),
    ):
        config, checkpoint, model = load_teacher(
            config_path.resolve(), checkpoint_path.resolve(), device
        )
        if int(config["seed"]) != int(args.seed) or Path(config["split_path"]).resolve() != split_path:
            raise ValueError(f"{arm} student config does not match seed or split")
        values = extract(model, loader, device)
        if not np.array_equal(values["indices"], teacher_values["indices"]):
            raise ValueError(f"{arm} validation sample order differs from teacher")
        if not np.array_equal(values["labels"], teacher_values["labels"]):
            raise ValueError(f"{arm} validation labels differ from teacher")
        student_values[arm] = values

    teacher_logits = teacher_values["logits"]
    labels = teacher_values["labels"]
    uncertainty = teacher_values["uncertainty"].astype(np.float64)
    error = teacher_logits.argmax(axis=1) != labels
    reports = {}
    for arm, values in student_values.items():
        kl = per_sample_kl(values["logits"], teacher_logits, args.temperature)
        reports[arm] = grouped(kl, error, uncertainty)
        reports[arm]["teacher_uncertainty_kl_correlation"] = float(
            np.corrcoef(uncertainty, kl)[0, 1]
        )
    standard_bins = reports["standard"]["uncertainty_quintiles"]
    uncertainty_bins = reports["uncertainty"]["uncertainty_quintiles"]
    reports["uncertainty_minus_standard"] = {
        "all_kl_delta": reports["uncertainty"]["all"]["kl_mean"]
        - reports["standard"]["all"]["kl_mean"],
        "teacher_correct_kl_delta": reports["uncertainty"]["teacher_correct"]["kl_mean"]
        - reports["standard"]["teacher_correct"]["kl_mean"],
        "teacher_error_kl_delta": reports["uncertainty"]["teacher_error"]["kl_mean"]
        - reports["standard"]["teacher_error"]["kl_mean"],
        "quintile_kl_delta": [
            uncertainty_bins[i]["student_teacher_kl_mean"]
            - standard_bins[i]["student_teacher_kl_mean"]
            for i in range(5)
        ],
    }
    return {
        "purpose": "frozen known-validation KD target alignment diagnostic",
        "protocol": {
            "seed": int(args.seed),
            "split_path": str(split_path),
            "temperature": float(args.temperature),
            "validation_labels_used_for": "post-hoc teacher-error grouping only",
            "used_for_training_or_calibration": False,
        },
        "teacher_validation": {
            "samples": int(len(labels)),
            "error_rate": float(error.mean()),
            "uncertainty_mean": float(uncertainty.mean()),
        },
        "arms": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--standard-config", required=True)
    parser.add_argument("--standard-checkpoint", required=True)
    parser.add_argument("--uncertainty-config", required=True)
    parser.add_argument("--uncertainty-checkpoint", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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
