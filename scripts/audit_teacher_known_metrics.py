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

from novel_discovery.data import build_data_bundle, build_transforms  # noqa: E402
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
    bundle = build_data_bundle(
        "cifar100", configs["epochs_5"]["data_root"], 60, args.seed, 64,
        split_path=split_path, discovery_pool_mode="unknown", transform_profile="cifar",
    )
    eval_transform = build_transforms(64, train=False, profile="cifar")
    replace_with_eval_transform(bundle.train, eval_transform)
    replace_with_eval_transform(bundle.val, eval_transform)
    loaders = {
        "known_train": build_loader(bundle.train, args.batch_size, False, 0),
        "known_validation": build_loader(bundle.val, args.batch_size, False, 0),
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
        "unknown_test_samples_loaded": False,
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
