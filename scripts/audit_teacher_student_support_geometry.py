"""Compare frozen teacher/student known-support geometry without retraining.

Known training labels define empirical class centers. The fixed test set is
used only for retrospective geometry and AUROC diagnostics; no threshold,
checkpoint, or model is selected from these values.
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, build_transforms  # noqa: E402
from novel_discovery.models import build_model  # noqa: E402
from novel_discovery.pipeline import build_loader  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    config = read_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Unsupported checkpoint layout: {checkpoint_path}")
    model = build_model(
        num_classes=len(checkpoint["known_classes"]),
        backbone=config["backbone"],
        proj_dim=int(config["proj_dim"]),
        dropout=float(config["dropout"]),
        pretrained=False,
        cifar_stem=bool(config.get("cifar_stem", False)),
        rejection_feature_dim=int(config.get("rejection_feature_dim", 0)),
        independent_rejection_encoder=bool(
            config.get("independent_rejection_encoder", False)
        ),
        independent_rejection_encoder_init=config.get(
            "independent_rejection_encoder_init", "shared_backbone"
        ),
        independent_rejection_input_norm=config.get(
            "independent_rejection_input_norm", "none"
        ),
        structure_feature_dim=int(config.get("structure_feature_dim", 0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return config, checkpoint, model


@torch.inference_mode()
def extract(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    known: list[np.ndarray] = []
    for images, mapped_labels, _raw_labels, is_known, _indices in loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        features.append(output["features"].float().cpu().numpy())
        labels.append(mapped_labels.cpu().numpy())
        known.append(is_known.cpu().numpy().astype(bool))
    if not features:
        raise ValueError("Cannot diagnose an empty data split")
    return {
        "features": np.concatenate(features),
        "labels": np.concatenate(labels),
        "known": np.concatenate(known),
    }


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def empirical_centers(features: np.ndarray, labels: np.ndarray, known: np.ndarray, count: int):
    x = normalized(features[known])
    y = np.asarray(labels[known], dtype=np.int64)
    centers = np.zeros((count, x.shape[1]), dtype=np.float64)
    present = np.zeros(count, dtype=bool)
    for class_id in range(count):
        rows = x[y == class_id]
        if len(rows):
            centers[class_id] = rows.mean(axis=0)
            present[class_id] = True
    if not present.all():
        raise ValueError(f"Known training support is missing classes: {np.flatnonzero(~present).tolist()}")
    return normalized(centers)


def distance_summary(scores: np.ndarray, known: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    known = np.asarray(known, dtype=bool)
    if len(np.unique(known)) != 2:
        raise ValueError("Evaluation must contain both known and unknown samples")
    return {
        "unknown_auroc": float(roc_auc_score((~known).astype(np.int64), values)),
        "known_mean": float(values[known].mean()),
        "unknown_mean": float(values[~known].mean()),
        "known_q90": float(np.quantile(values[known], 0.90)),
        "unknown_q10": float(np.quantile(values[~known], 0.10)),
    }


def model_metrics(model, train_data: dict[str, np.ndarray], test_data: dict[str, np.ndarray]):
    known_classes = model.classifier.weight.shape[0]
    train_centers = empirical_centers(
        train_data["features"], train_data["labels"], train_data["known"], known_classes
    )
    test_features = normalized(test_data["features"])
    classifier_prototypes = normalized(model.classifier.weight.detach().float().cpu().numpy())
    centroid_distance = 1.0 - (test_features @ train_centers.T).max(axis=1)
    classifier_distance = 1.0 - (test_features @ classifier_prototypes.T).max(axis=1)
    known = test_data["known"]
    return {
        "unknownness_scores": {
            "classifier_prototype_distance": distance_summary(classifier_distance, known),
            "empirical_class_centroid_distance": distance_summary(centroid_distance, known),
        },
    }


def audit_seed(seed: int, arms: list[str], tag: str, device: torch.device, batch_size: int):
    split_path = ROOT / f"splits/semantic_hard/seed{seed}/cifar100_60_40_semantic_hard.json"
    split = read_json(split_path)
    if split.get("metadata", {}).get("seed") != seed or len(split.get("known_classes", [])) != 60:
        raise ValueError(f"Unexpected semantic-hard split for seed {seed}")

    teacher_dir = ROOT / "runs" / f"{tag}_s{seed}_teacher"
    teacher_config_path = teacher_dir / "train_teacher_config.json"
    teacher_checkpoint_path = teacher_dir / "teacher.pt"
    reference_config = read_json(teacher_config_path)
    if reference_config.get("split_path") != str(split_path):
        raise ValueError(f"Teacher config points to a different split for seed {seed}")
    if int(reference_config.get("limit_train", 0)) != 0 or int(reference_config.get("limit_test", 0)) != 0:
        raise ValueError("This audit expects full known-train and test splits")
    if reference_config.get("transform_profile") != "cifar":
        raise ValueError("This CIFAR-100 audit requires the saved CIFAR transform profile")

    bundle = build_data_bundle(
        "cifar100",
        reference_config["data_root"],
        60,
        seed,
        int(reference_config["image_size"]),
        split_path=split_path,
        discovery_pool_mode="unknown",
        transform_profile=reference_config["transform_profile"],
    )
    train_subset = bundle.train
    while hasattr(train_subset, "dataset"):
        train_subset = train_subset.dataset
    if not hasattr(train_subset, "base"):
        raise TypeError("Expected a CIFAR-style known training dataset")
    train_subset.base.transform = build_transforms(
        int(reference_config["image_size"]),
        train=False,
        profile=reference_config["transform_profile"],
    )
    train_loader = build_loader(bundle.train, batch_size, False, 0)
    test_loader = build_loader(bundle.test, batch_size, False, 0)

    teacher_config, teacher_checkpoint, teacher_model = load_model(
        teacher_config_path, teacher_checkpoint_path, device
    )
    expected_classes = split["known_classes"]
    if teacher_checkpoint.get("known_classes") != expected_classes:
        raise ValueError(f"Teacher checkpoint classes mismatch for seed {seed}")
    reference_train = extract(teacher_model, train_loader, device)
    reference_test = extract(teacher_model, test_loader, device)
    if not reference_train["known"].all() or reference_test["known"].all():
        raise ValueError(f"Unexpected known/unknown membership for seed {seed}")
    if len(reference_train["known"]) != 27000 or int(reference_test["known"].sum()) != 6000 or int((~reference_test["known"]).sum()) != 4000:
        raise ValueError(f"Unexpected full CIFAR-100 60/40 sample counts for seed {seed}")
    results = {"teacher": model_metrics(teacher_model, reference_train, reference_test)}
    results["teacher"]["feature_dimension"] = int(teacher_model.classifier.weight.shape[1])
    del teacher_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for arm in arms:
        student_dir = ROOT / "runs" / f"{tag}_s{seed}_{arm}"
        student_config, checkpoint, model = load_model(
            student_dir / "train_student_config.json",
            student_dir / "student.pt",
            device,
        )
        if checkpoint.get("known_classes") != expected_classes:
            raise ValueError(f"{arm} checkpoint classes mismatch for seed {seed}")
        if checkpoint.get("novel_classes") != teacher_checkpoint.get("novel_classes"):
            raise ValueError(f"{arm} checkpoint novel classes mismatch for seed {seed}")
        for field in ("dataset", "image_size", "transform_profile", "seed", "num_known"):
            if student_config.get(field) != reference_config.get(field):
                raise ValueError(f"Student {field} mismatch for {arm}, seed {seed}")
        if Path(str(student_config.get("split_path"))).as_posix().casefold() != Path(
            str(split_path)
        ).as_posix().casefold():
            raise ValueError(f"Student split mismatch for {arm}, seed {seed}")
        discover_dir = student_dir.with_name(f"{student_dir.name}_discover")
        discover_config = read_json(discover_dir / "discover_config.json")
        discovery_report = read_json(discover_dir / "discovery_report.json")
        for field in ("dataset", "image_size", "transform_profile", "seed"):
            if discover_config.get(field) != reference_config.get(field):
                raise ValueError(f"Discovery {field} mismatch for {arm}, seed {seed}")
        if Path(str(discover_config.get("split_path"))).as_posix().casefold() != Path(
            str(split_path)
        ).as_posix().casefold():
            raise ValueError(f"Discovery split mismatch for {arm}, seed {seed}")
        train_data = extract(model, train_loader, device)
        test_data = extract(model, test_loader, device)
        if not np.array_equal(train_data["known"], reference_train["known"]):
            raise RuntimeError(f"Known training sample order differs for {arm}, seed {seed}")
        if not np.array_equal(train_data["labels"], reference_train["labels"]):
            raise RuntimeError(f"Known training labels/order differ for {arm}, seed {seed}")
        if not np.array_equal(test_data["known"], reference_test["known"]):
            raise RuntimeError(f"Test membership/order differs for {arm}, seed {seed}")
        if not np.array_equal(test_data["labels"], reference_test["labels"]):
            raise RuntimeError(f"Test labels/order differs for {arm}, seed {seed}")
        result = model_metrics(model, train_data, test_data)
        result["feature_dimension"] = int(train_data["features"].shape[1])
        result["official_discovery_metrics"] = {
            "max_softmax_auroc": float(discovery_report["auroc"]),
            "known_test_accuracy": float(discovery_report["known_class_accuracy_all_known"]),
            "score_mode": discover_config.get("score_mode"),
            "mc_samples": int(discover_config.get("mc_samples", 1)),
            "transform_profile": discover_config["transform_profile"],
        }
        results[arm] = result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "seed": seed,
        "protocol": "semantic-hard CIFAR-100 60/40",
        "known_training_samples": int(train_data["known"].sum()),
        "test_known_samples": int(test_data["known"].sum()),
        "test_unknown_samples": int((~test_data["known"]).sum()),
        "unknown_labels_used_only_for_posthoc_metrics": True,
        "teacher_and_student_share_feature_dimension": len(
            {entry["feature_dimension"] for entry in results.values()}
        ) == 1,
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2163, 2164])
    parser.add_argument("--arms", nargs="+", choices=("standard", "selective", "uncertainty"),
                        default=["standard", "selective", "uncertainty"])
    parser.add_argument("--tag", default="team_a_selective_kd_confirm_20261122")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis/team_a_teacher_student_support_geometry_20261123_cifar_transform.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    seeds = [audit_seed(seed, args.arms, args.tag, device, args.batch_size) for seed in args.seeds]
    payload = {
        "purpose": "frozen teacher/student known-support geometry audit; no training or model selection",
        "device": str(device),
        "geometry_scores": [
            "classifier prototype cosine distance",
            "empirical known-class centroid cosine distance",
        ],
        "official_detection_metrics": "read from completed discovery_report.json; not recomputed",
        "seeds": seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": seeds}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
