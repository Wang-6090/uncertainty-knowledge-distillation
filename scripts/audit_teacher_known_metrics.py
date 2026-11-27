"""Compare frozen teachers on identical deterministic known-only data."""

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

from novel_discovery.data import (  # noqa: E402
    OpenSetCIFAR100,
    build_transforms,
    make_class_split,
    split_known_indices,
)
from novel_discovery.pipeline import build_loader  # noqa: E402
from scripts.audit_frozen_student_classification_impact import (  # noqa: E402
    classification_values,
    extract,
    load_teacher,
    resolved_path,
)
from scripts.audit_teacher_uncertainty_alignment import replace_with_eval_transform  # noqa: E402


def audit(args: argparse.Namespace) -> dict[str, Any]:
    split_path = resolved_path(args.split_path)
    config_paths = {
        "epochs_5": resolved_path(args.config_5),
        "epochs_15": resolved_path(args.config_15),
    }
    checkpoint_paths = {
        "epochs_5": resolved_path(args.checkpoint_5),
        "epochs_15": resolved_path(args.checkpoint_15),
    }
    configs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in config_paths.items()
    }
    for name, config in configs.items():
        if int(config["seed"]) != args.seed:
            raise ValueError(f"{name} teacher seed mismatch")
        if resolved_path(config["split_path"]) != split_path:
            raise ValueError(f"{name} teacher split mismatch")
        if config.get("transform_profile") != "cifar":
            raise ValueError(f"{name} teacher transform mismatch")
        if int(config.get("limit_train", 0)) != 0:
            raise ValueError(f"{name} teacher must use full known train")
    invariant_fields = (
        "backbone", "pretrained", "image_size", "batch_size", "lr",
        "weight_decay", "alpha_unc", "freeze_bn_stats", "transform_profile",
    )
    for field in invariant_fields:
        if configs["epochs_5"].get(field) != configs["epochs_15"].get(field):
            raise ValueError(f"teacher arms differ unexpectedly on {field}")
    if int(configs["epochs_5"]["epochs"]) != 5 or int(configs["epochs_15"]["epochs"]) != 15:
        raise ValueError("expected the fixed 5 vs 15 epoch comparison")

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    known_classes, _novel_classes = make_class_split(
        list(range(100)), 60, args.seed, str(split_path)
    )
    train_transform = build_transforms(64, train=True, profile="cifar")
    eval_transform = build_transforms(64, train=False, profile="cifar")
    train_full = OpenSetCIFAR100(
        configs["epochs_5"]["data_root"], known_classes, train=True,
        transform=train_transform, download=False, include_unknown=False,
    )
    val_full = OpenSetCIFAR100(
        configs["epochs_5"]["data_root"], known_classes, train=True,
        transform=eval_transform, download=False, include_unknown=False,
    )
    train_indices, val_indices = split_known_indices(
        len(train_full), val_ratio=0.1, seed=args.seed
    )
    train_set = torch.utils.data.Subset(train_full, train_indices)
    val_set = torch.utils.data.Subset(val_full, val_indices)
    replace_with_eval_transform(train_set, eval_transform)
    loaders = {
        "known_train": build_loader(train_set, args.batch_size, False, 0),
        "known_validation": build_loader(val_set, args.batch_size, False, 0),
    }
    models = {
        name: load_teacher(config_paths[name], checkpoint_paths[name], device)[2]
        for name in configs
    }
    reports: dict[str, Any] = {}
    for split_name, loader in loaders.items():
        split_values = {name: extract(model, loader, device) for name, model in models.items()}
        reference = split_values["epochs_5"]
        for name, values in split_values.items():
            if not np.array_equal(reference["indices"], values["indices"]):
                raise ValueError(f"{split_name} sample order mismatch for {name}")
            if not np.array_equal(reference["labels"], values["labels"]):
                raise ValueError(f"{split_name} labels mismatch for {name}")
        metrics = {}
        for name, values in split_values.items():
            summary = classification_values(values["logits"], values["labels"])
            metrics[name] = {
                "samples": int(len(values["labels"])),
                "accuracy": float(summary["correct"].mean()),
                "nll": float(summary["nll"].mean()),
            }
        metrics["delta_15_minus_5"] = {
            key: metrics["epochs_15"][key] - metrics["epochs_5"][key]
            for key in ("accuracy", "nll")
        }
        reports[split_name] = metrics
    return {
        "purpose": "frozen known-only teacher training-budget audit",
        "seed": args.seed,
        "split_path": str(split_path),
        "test_split_instantiated": False,
        "unknown_or_test_examples_iterated": False,
        "novel_class_metadata_used_only_for_known_subset_exclusion": True,
        "unknown_metrics_used_for_selection": False,
        "used_for_training_selection_or_tuning": False,
        "device": str(device),
        "invariant_fields": list(invariant_fields),
        "results": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2182)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--config-5", required=True)
    parser.add_argument("--checkpoint-5", required=True)
    parser.add_argument("--config-15", required=True)
    parser.add_argument("--checkpoint-15", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args)
    output = resolved_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
