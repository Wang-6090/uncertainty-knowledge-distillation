"""Compare CE-only and standard-KD students using matched frozen teachers."""

from __future__ import annotations

import argparse
import hashlib
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

from novel_discovery.data import OpenSetCIFAR100, build_transforms, make_class_split, split_known_indices  # noqa: E402
from scripts.audit_srd_scale_geometry import summarize as summarize_geometry  # noqa: E402
from scripts.audit_teacher_uncertainty_alignment import load_teacher  # noqa: E402
from scripts.run_uncertainty_kd_core_pair import make_ce_only_student_config, read_json, write_json  # noqa: E402
from scripts.audit_frozen_student_classification_impact import classification_values, resolved_path  # noqa: E402
from train import fit_student, parse_args  # noqa: E402


IGNORED_TEACHER_CONFIG_FIELDS = {
    "best_epoch", "best_val_known_acc", "epochs", "split_path", "teacher_ckpt", "work_dir",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_teacher_pair(
    seed: int,
    split_path: Path,
    known_classes: list[int],
    novel_classes: list[int],
    teacher_config_paths: dict[str, Path],
    checkpoint_paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    configs = {name: read_json(path) for name, path in teacher_config_paths.items()}
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in checkpoint_paths.items()
    }
    reference = configs["epochs_5"]
    for name, config in configs.items():
        if int(config["seed"]) != seed:
            raise ValueError(f"{name} teacher seed mismatch")
        if resolved_path(config["split_path"]) != split_path:
            raise ValueError(f"{name} teacher split mismatch")
        if int(config["epochs"]) != (5 if name == "epochs_5" else 15):
            raise ValueError(f"unexpected {name} training budget")
        if config.get("transform_profile") != "cifar" or int(config["num_known"]) != 60:
            raise ValueError(f"{name} teacher does not use CIFAR semantic-hard 60/40")
        if not isinstance(payloads[name], dict) or not isinstance(payloads[name].get("model"), dict):
            raise ValueError(f"unsupported teacher checkpoint format: {checkpoint_paths[name]}")
        if payloads[name].get("known_classes") != known_classes:
            raise ValueError(f"{name} checkpoint known classes differ from the split")
        if payloads[name].get("novel_classes") != novel_classes:
            raise ValueError(f"{name} checkpoint novel classes differ from the split")

    normalized = {
        name: {key: value for key, value in config.items() if key not in IGNORED_TEACHER_CONFIG_FIELDS}
        for name, config in configs.items()
    }
    if normalized["epochs_5"] != normalized["epochs_15"]:
        differences = sorted(
            key for key in set(normalized["epochs_5"]) | set(normalized["epochs_15"])
            if normalized["epochs_5"].get(key) != normalized["epochs_15"].get(key)
        )
        raise ValueError(f"teacher recipes differ outside epoch/run metadata: {differences}")
    return configs


def build_student_config(
    source: dict[str, Any], seed: int, split_path: Path, alpha_kd: float,
) -> dict[str, Any]:
    config = make_ce_only_student_config(source)
    config.update({
        "seed": seed,
        "model_seed": None,
        "split_path": str(split_path),
        "alpha_kd": alpha_kd,
        "kd_mode": "standard",
        "uncertainty_weight_mode": "mean_normalized",
        "checkpoint_selection": "known_acc",
        "open_val_ratio": 0.0,
        "discovery_pool": False,
        "matched_discovery_pool": False,
        "matched_pool_size": 0,
        "audit_loss_gradients": True,
    })
    non_kd_auxiliary = {
        key: value for key, value in config.items()
        if key.startswith("alpha_") and key != "alpha_kd" and isinstance(value, (int, float))
        and not isinstance(value, bool) and float(value) != 0.0
    }
    if non_kd_auxiliary:
        raise ValueError(f"CE/KD transfer experiment has active auxiliary losses: {non_kd_auxiliary}")
    return config


def make_args(config: dict[str, Any], run_dir: Path, teacher_checkpoint: Path):
    args = parse_args(["train_student"])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.work_dir = str(run_dir)
    args.student_ckpt = str(run_dir / "student.pt")
    args.teacher_ckpt = str(teacher_checkpoint)
    args.kd_mode = "standard"
    args.uncertainty_weight_mode = "mean_normalized"
    return args


@torch.inference_mode()
def extract(model, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for images, mapped_labels, _raw_labels, is_known, sample_indices in loader:
        if not bool(is_known.all()):
            raise ValueError("student-transfer audit received a non-known validation row")
        output = model(images.to(device, non_blocking=True))
        features.append(output["features"].float().cpu())
        logits.append(output["logits"].float().cpu())
        labels.append(mapped_labels.cpu())
        indices.append(sample_indices.cpu())
    if not labels:
        raise ValueError("known validation loader is empty")
    return {
        "features": torch.cat(features),
        "logits": torch.cat(logits).numpy(),
        "labels": torch.cat(labels).numpy(),
        "indices": torch.cat(indices),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    split_path = resolved_path(args.split_path)
    student_source_path = resolved_path(args.student_source_config)
    teacher_config_paths = {
        "epochs_5": resolved_path(args.teacher_config_5),
        "epochs_15": resolved_path(args.teacher_config_15),
    }
    checkpoint_paths = {
        "epochs_5": resolved_path(args.teacher_checkpoint_5),
        "epochs_15": resolved_path(args.teacher_checkpoint_15),
    }
    for path in [split_path, student_source_path, *teacher_config_paths.values(), *checkpoint_paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    split = read_json(split_path)
    known_classes = list(split["known_classes"])
    novel_classes = list(split["novel_classes"])
    if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != args.seed:
        raise ValueError("split metadata does not match seed/protocol")
    if len(known_classes) != 60 or len(novel_classes) != 40 or set(known_classes) & set(novel_classes):
        raise ValueError("expected a disjoint semantic-hard CIFAR-100 60/40 split")
    teacher_configs = validate_teacher_pair(
        args.seed, split_path, known_classes, novel_classes, teacher_config_paths, checkpoint_paths
    )

    student_source = read_json(student_source_path)
    if student_source.get("dataset") != "cifar100" or int(student_source.get("num_known", -1)) != 60:
        raise ValueError("student source must be a CIFAR-100 60-known-class recipe")
    if student_source.get("backbone") != "resnet18" or int(student_source.get("epochs", -1)) != 4:
        raise ValueError("student source must match the fixed ResNet-18, four-epoch recipe")
    if student_source.get("transform_profile") != "cifar":
        raise ValueError("student source must use the CIFAR transform profile")
    if student_source.get("checkpoint_selection") != "known_acc":
        raise ValueError("student checkpoint selection must use known validation only")
    if float(student_source.get("calibration_ratio", 0.0)) != 0.0:
        raise ValueError("the student known-validation set must remain complete")
    for name, teacher_config in teacher_configs.items():
        if teacher_config.get("backbone") != student_source.get("teacher_backbone"):
            raise ValueError(f"{name} backbone does not match the student's teacher_backbone")
        for field in ("dataset", "data_root", "num_known", "image_size", "transform_profile", "batch_size"):
            if teacher_config.get(field) != student_source.get(field):
                raise ValueError(f"student and {name} differ on protocol field {field}")
    source_root = Path(student_source["data_root"])
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    known_train_dataset = OpenSetCIFAR100(
        str(source_root), known_classes, train=True,
        transform=build_transforms(int(student_source["image_size"]), False, "cifar"),
        download=False, include_unknown=False,
    )
    train_indices, _validation_indices = split_known_indices(
        len(known_train_dataset), 0.1, args.seed
    )
    known_train_samples = len(train_indices)

    student_configs = {
        "ce_only": build_student_config(student_source, args.seed, split_path, 0.0),
        "standard_kd_e5": build_student_config(student_source, args.seed, split_path, 1.0),
        "standard_kd_e15": build_student_config(student_source, args.seed, split_path, 1.0),
    }
    # Enforce a single common student recipe; only KD activation and its frozen teacher may differ.
    reference = {key: value for key, value in student_configs["ce_only"].items() if key != "alpha_kd"}
    for name, config in student_configs.items():
        candidate = {key: value for key, value in config.items() if key != "alpha_kd"}
        if candidate != reference:
            raise ValueError(f"student recipe mismatch in arm {name}")

    run_dirs = {
        arm: ROOT / "runs" / f"{args.tag}_s{args.seed}_{arm}"
        for arm in student_configs
    }
    output_path = resolved_path(args.output)
    protocol_path = ROOT / "analysis" / f"{args.tag}_s{args.seed}_protocol.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output_path}")
    if args.audit_only:
        if args.dry_run:
            raise ValueError("--audit-only and --dry-run cannot be combined")
        if not protocol_path.is_file():
            raise FileNotFoundError(f"missing training protocol record: {protocol_path}")
        for arm, run_dir in run_dirs.items():
            for artifact in ("student.pt", "train_history.json", "train_student_config.json"):
                if not (run_dir / artifact).is_file():
                    raise FileNotFoundError(f"audit-only missing {artifact} for {arm}")
            saved = read_json(run_dir / "train_student_config.json")
            expected_alpha = float(student_configs[arm]["alpha_kd"])
            if (
                int(saved.get("seed", -1)) != args.seed
                or resolved_path(saved.get("split_path", "")) != split_path
                or float(saved.get("alpha_kd", -1)) != expected_alpha
                or saved.get("kd_mode") != "standard"
                or saved.get("checkpoint_selection") != "known_acc"
                or bool(saved.get("matched_discovery_pool", True))
            ):
                raise ValueError(f"saved student config does not match audited arm {arm}")
        protocol = read_json(protocol_path)
        for name, path in checkpoint_paths.items():
            recorded = protocol.get("teacher_checkpoints", {}).get(name, {})
            if recorded.get("sha256") != digest(path):
                raise ValueError(f"frozen teacher checkpoint changed since training: {name}")
    else:
        protected = [*run_dirs.values(), protocol_path]
        existing = [path for path in protected if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite existing training outputs: " + ", ".join(map(str, existing)))
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite audit output: {output_path}")

    arms = {
        "ce_only": {"teacher": str(checkpoint_paths["epochs_5"]), "alpha_kd": 0.0},
        "standard_kd_e5": {"teacher": str(checkpoint_paths["epochs_5"]), "alpha_kd": student_configs["standard_kd_e5"]["alpha_kd"]},
        "standard_kd_e15": {"teacher": str(checkpoint_paths["epochs_15"]), "alpha_kd": student_configs["standard_kd_e15"]["alpha_kd"]},
    }
    if args.dry_run:
        return {
            "dry_run": True,
            "seed": args.seed,
            "split_path": str(split_path),
            "student_epochs": student_source.get("epochs"),
            "known_train_samples": known_train_samples,
            "training_batches_per_epoch": (known_train_samples + int(student_source["batch_size"]) - 1) // int(student_source["batch_size"]),
            "known_only_checkpoint_selection": "known_acc",
            "teacher_fit_calls": 0,
            "arms": arms,
            "outputs": {name: str(path) for name, path in run_dirs.items()},
        }

    protocol = {
        "seed": args.seed,
        "split_path": str(split_path),
        "student_source_config": str(student_source_path),
        "teacher_configs": {name: str(path) for name, path in teacher_config_paths.items()},
        "teacher_checkpoints": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in checkpoint_paths.items()
        },
        "teacher_fit_calls": 0,
        "unknown_or_test_training": False,
        "known_train_samples": known_train_samples,
        "training_batches_per_epoch": (known_train_samples + int(student_source["batch_size"]) - 1) // int(student_source["batch_size"]),
        "checkpoint_selection": "known_acc",
        "arms": arms,
    }
    if not args.audit_only:
        write_json(protocol_path, protocol)
        teacher_for_arm = {
            "ce_only": checkpoint_paths["epochs_5"],
            "standard_kd_e5": checkpoint_paths["epochs_5"],
            "standard_kd_e15": checkpoint_paths["epochs_15"],
        }
        for arm, config in student_configs.items():
            print(f"[seed={args.seed}] train {arm}; teacher training disabled", flush=True)
            fit_student(make_args(config, run_dirs[arm], teacher_for_arm[arm]))
            for artifact in ("student.pt", "train_history.json", "train_student_config.json"):
                if not (run_dirs[arm] / artifact).is_file():
                    raise FileNotFoundError(f"training did not produce {artifact} for {arm}")

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    known_dataset = OpenSetCIFAR100(
        str(source_root), known_classes, train=True,
        transform=build_transforms(int(student_source["image_size"]), False, "cifar"),
        download=False, include_unknown=False,
    )
    _, validation_indices = split_known_indices(len(known_dataset), 0.1, args.seed)
    loader = DataLoader(
        Subset(known_dataset, validation_indices), batch_size=args.eval_batch_size,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
    )

    models: dict[str, torch.nn.Module] = {}
    configs: dict[str, dict[str, Any]] = {}
    checkpoints = {
        "teacher_e5": (teacher_config_paths["epochs_5"], checkpoint_paths["epochs_5"]),
        "teacher_e15": (teacher_config_paths["epochs_15"], checkpoint_paths["epochs_15"]),
    }
    for arm in student_configs:
        checkpoints[arm] = (run_dirs[arm] / "train_student_config.json", run_dirs[arm] / "student.pt")
    for name, (config_path, checkpoint_path) in checkpoints.items():
        configs[name], _payload, models[name] = load_teacher(config_path, checkpoint_path, device)

    values = {name: extract(model, loader, device) for name, model in models.items()}
    reference_ids = values["teacher_e5"]["indices"]
    reference_labels = values["teacher_e5"]["labels"]
    for name, result in values.items():
        if not torch.equal(reference_ids, result["indices"]) or not np.array_equal(reference_labels, result["labels"]):
            raise ValueError(f"validation rows differ for {name}")

    metric_report: dict[str, Any] = {}
    for name, result in values.items():
        summary = classification_values(result["logits"], result["labels"])
        metric_report[name] = {
            "samples": int(len(result["labels"])),
            "accuracy": float(summary["correct"].mean()),
            "nll": float(summary["nll"].mean()),
            "representation_geometry": summarize_geometry(
                result["features"], torch.from_numpy(result["labels"]).long(), len(known_classes)
            ),
        }
    history_report = {}
    for arm in student_configs:
        history = read_json(run_dirs[arm] / "train_history.json")
        alpha_kd = float(read_json(run_dirs[arm] / "train_student_config.json")["alpha_kd"])
        raw_kd_losses = [entry.get("train", {}).get("kd") for entry in history]
        raw_kd_gradients = [
            entry.get("train", {}).get("gradient_audit_grad_norm_kd") for entry in history
        ]
        classifier_updates = [
            entry.get("train", {}).get("student_classifier_update_norm") for entry in history
        ]
        history_report[arm] = {
            "epochs_completed": len(history),
            "alpha_kd": alpha_kd,
            "raw_unweighted_kd_loss": raw_kd_losses,
            "effective_weighted_kd_loss": [alpha_kd * value for value in raw_kd_losses],
            "raw_unweighted_kd_gradient_norm": raw_kd_gradients,
            "effective_weighted_kd_gradient_norm": [alpha_kd * value for value in raw_kd_gradients],
            "total_classifier_update_norm": classifier_updates,
            "kd_mechanism_verified": bool(
                alpha_kd > 0.0
                and all(value is not None and value > 0.0 for value in raw_kd_losses)
                and all(value is not None and value > 0.0 for value in raw_kd_gradients)
                and all(value is not None and value > 0.0 for value in classifier_updates)
            ),
        }
    report = {
        "purpose": "known-only teacher-budget KD transfer gate",
        "seed": args.seed,
        "split_path": str(split_path),
        "test_split_instantiated": False,
        "unknown_or_test_examples_iterated": False,
        "known_validation_samples": int(len(reference_labels)),
        "checkpoint_selection": "known_acc",
        "arms": arms,
        "metrics": metric_report,
        "training_mechanism": history_report,
    }
    for name in ("standard_kd_e5", "standard_kd_e15"):
        report["metrics"][name]["delta_vs_ce_only"] = {
            "accuracy": metric_report[name]["accuracy"] - metric_report["ce_only"]["accuracy"],
            "nll": metric_report[name]["nll"] - metric_report["ce_only"]["nll"],
            "normalized_within_cosine_distance_mean": (
                metric_report[name]["representation_geometry"]["normalized_within_cosine_distance_mean"]
                - metric_report["ce_only"]["representation_geometry"]["normalized_within_cosine_distance_mean"]
            ),
            "normalized_between_total_scatter_ratio": (
                metric_report[name]["representation_geometry"]["normalized_between_total_scatter_ratio"]
                - metric_report["ce_only"]["representation_geometry"]["normalized_between_total_scatter_ratio"]
            ),
        }
    write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2182)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--student-source-config", required=True)
    parser.add_argument("--teacher-config-5", required=True)
    parser.add_argument("--teacher-checkpoint-5", required=True)
    parser.add_argument("--teacher-config-15", required=True)
    parser.add_argument("--teacher-checkpoint-15", required=True)
    parser.add_argument("--tag", default="team_a_teacher_budget_transfer_full_known_20261127")
    parser.add_argument("--output", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = str(ROOT / "analysis" / f"{args.tag}_s{args.seed}.json")
    result = run(args)
    print(json.dumps(result if args.dry_run else result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
