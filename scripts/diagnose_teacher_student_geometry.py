"""Compare teacher/student pairwise feature geometry on known validation only.

This is a read-only representation diagnostic. It uses only the common known
validation subset available to both checkpoints: the teacher's model-selection
portion after its configured calibration holdout. It never iterates over the
unknown test split, open-validation set, or discovery pool, and it does not
train models or choose a checkpoint/threshold.
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
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import OpenSetCIFAR100, build_transforms, split_known_indices  # noqa: E402
from novel_discovery.models import build_model  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"Unsupported checkpoint layout: {path}")
    return payload


def make_model(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device):
    model = build_model(
        num_classes=len(checkpoint["known_classes"]),
        backbone=config["backbone"],
        proj_dim=int(config["proj_dim"]),
        dropout=float(config["dropout"]),
        pretrained=False,
        cifar_stem=bool(config["cifar_stem"]),
        rejection_feature_dim=int(config.get("rejection_feature_dim", 0)),
        independent_rejection_encoder=bool(config.get("independent_rejection_encoder", False)),
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
    return model


@torch.inference_mode()
def extract_features(model, loader: DataLoader, device: torch.device):
    features: list[torch.Tensor] = []
    sample_ids: list[torch.Tensor] = []
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        # The returned class labels are intentionally ignored. Only sample IDs
        # are retained to verify that both networks saw the same known images.
        sample_ids.append(batch[4].cpu())
        features.append(model(images)["features"].float().cpu())
    return torch.cat(features), torch.cat(sample_ids)


def geometry_metrics(teacher: torch.Tensor, student: torch.Tensor) -> dict[str, Any]:
    teacher = F.normalize(teacher.double(), dim=1)
    student = F.normalize(student.double(), dim=1)
    n = int(teacher.shape[0])

    # Exact upper-triangle pairwise cosine-distance comparison.
    teacher_distance = 1.0 - teacher @ teacher.T
    student_distance = 1.0 - student @ student.T
    upper = torch.triu_indices(n, n, offset=1)
    td = teacher_distance[upper[0], upper[1]].numpy()
    sd = student_distance[upper[0], upper[1]].numpy()
    rho = float(spearmanr(td, sd).statistic)

    # Linear CKA compares centered sample Gram matrices and is invariant to
    # orthogonal changes of feature coordinates.
    tx = teacher - teacher.mean(dim=0, keepdim=True)
    sx = student - student.mean(dim=0, keepdim=True)
    tg = tx @ tx.T
    sg = sx @ sx.T
    cka = float((tg * sg).sum() / (tg.square().sum().sqrt() * sg.square().sum().sqrt()))

    # Neighborhood preservation is reported at several local scales.
    teacher_similarity = teacher @ teacher.T
    student_similarity = student @ student.T
    teacher_similarity.fill_diagonal_(-torch.inf)
    student_similarity.fill_diagonal_(-torch.inf)
    neighborhoods: dict[str, float] = {}
    for k in (5, 10, 20):
        tn = teacher_similarity.topk(k, dim=1).indices
        sn = student_similarity.topk(k, dim=1).indices
        overlap = (tn.unsqueeze(2) == sn.unsqueeze(1)).any(dim=2).float().sum(dim=1) / k
        neighborhoods[str(k)] = float(overlap.mean())

    return {
        "samples": n,
        "unordered_pairs": int(n * (n - 1) // 2),
        "cosine_distance_spearman": rho,
        "linear_cka": cka,
        "teacher_student_knn_overlap": neighborhoods,
        "teacher_cosine_distance_mean": float(td.mean()),
        "student_cosine_distance_mean": float(sd.mean()),
        "teacher_cosine_distance_std": float(td.std()),
        "student_cosine_distance_std": float(sd.std()),
    }


def run_seed(seed: int, device: torch.device, batch_size: int) -> dict[str, Any]:
    split_path = ROOT / f"splits/semantic_hard/seed{seed}/cifar100_60_40_semantic_hard.json"
    teacher_dir = ROOT / "runs" / f"structure_representation_pilot_s{seed}_teacher"
    student_dir = ROOT / "runs" / f"structure_representation_pilot_s{seed}_student"
    split = read_json(split_path)
    teacher_config = read_json(teacher_dir / "train_teacher_config.json")
    student_config = read_json(student_dir / "train_student_config.json")
    teacher_ckpt = checkpoint_payload(teacher_dir / "teacher.pt")
    student_ckpt = checkpoint_payload(student_dir / "student.pt")

    if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != seed:
        raise ValueError(f"Split metadata mismatch for seed {seed}")
    known = list(split["known_classes"])
    novel = list(split["novel_classes"])
    if len(known) != 60 or len(novel) != 40 or set(known) & set(novel):
        raise ValueError(f"Invalid 60/40 split for seed {seed}")
    for role, config, checkpoint in (
        ("teacher", teacher_config, teacher_ckpt),
        ("student", student_config, student_ckpt),
    ):
        if config.get("seed") != seed:
            raise ValueError(f"{role} config seed mismatch for seed {seed}")
        if config.get("split_path") != str(split_path):
            raise ValueError(f"{role} config points at a different split for seed {seed}")
        if checkpoint.get("known_classes") != known or checkpoint.get("novel_classes") != novel:
            raise ValueError(f"{role} checkpoint class metadata mismatch for seed {seed}")
        if config.get("limit_val", 0) not in (0, None):
            raise ValueError(f"Unexpected validation subsampling in {role} config")

    teacher_calibration_ratio = float(teacher_config.get("calibration_ratio", 0.0))
    student_calibration_ratio = float(student_config.get("calibration_ratio", 0.0))
    if not 0.0 <= teacher_calibration_ratio < 1.0:
        raise ValueError(f"Invalid teacher calibration ratio for seed {seed}")
    if student_calibration_ratio != 0.0:
        raise ValueError(
            "Expected the student checkpoint to use the complete known validation "
            f"set (calibration_ratio=0), got {student_calibration_ratio} for seed {seed}"
        )

    shared_fields = (
        "dataset", "data_root", "seed", "image_size", "transform_profile",
        "limit_val", "known_split_mode",
    )
    mismatched = {
        key: (teacher_config.get(key), student_config.get(key))
        for key in shared_fields
        if teacher_config.get(key) != student_config.get(key)
    }
    if mismatched:
        raise ValueError(f"Teacher/student validation protocol mismatch: {mismatched}")
    if teacher_config.get("dataset") != "cifar100":
        raise ValueError("This diagnostic is restricted to CIFAR-100")

    data_root = Path(teacher_config["data_root"])
    if not data_root.is_absolute():
        data_root = (ROOT / data_root).resolve()
    val_dataset = OpenSetCIFAR100(
        str(data_root),
        known,
        train=True,
        transform=build_transforms(
            int(teacher_config["image_size"]),
            train=False,
            profile=teacher_config["transform_profile"],
        ),
        download=False,
        include_unknown=False,
    )
    _, val_indices = split_known_indices(len(val_dataset), val_ratio=0.1, seed=seed)
    if teacher_calibration_ratio > 0.0:
        known_to_idx = {int(label): index for index, label in enumerate(known)}
        val_labels = [
            known_to_idx[int(val_dataset.base.targets[val_dataset.allowed_indices[index]])]
            for index in val_indices
        ]
        selection_indices, _ = split_known_indices(
            len(val_indices),
            val_ratio=teacher_calibration_ratio,
            seed=seed + 17,
            labels=val_labels,
        )
        val_indices = [val_indices[index] for index in selection_indices]
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    teacher_model = make_model(teacher_config, teacher_ckpt, device)
    student_model = make_model(student_config, student_ckpt, device)
    teacher_features, teacher_ids = extract_features(teacher_model, val_loader, device)
    student_features, student_ids = extract_features(student_model, val_loader, device)
    if not torch.equal(teacher_ids, student_ids):
        raise RuntimeError(f"Teacher/student validation sample order differs for seed {seed}")

    return {
        "seed": seed,
        "protocol": split["protocol"],
        "known_class_count": len(known),
        "unknown_class_labels_used": False,
        "data_accessed": "known validation subset of CIFAR-100 train split only",
        "known_validation_role": "teacher model-selection subset; teacher calibration holdout excluded",
        "teacher_calibration_ratio": teacher_calibration_ratio,
        "student_calibration_ratio": student_calibration_ratio,
        "teacher_backbone": teacher_config["backbone"],
        "student_backbone": student_config["backbone"],
        "teacher_checkpoint_epoch": teacher_config.get("best_epoch"),
        "student_checkpoint_epoch": student_config.get("best_epoch"),
        "known_validation_geometry": geometry_metrics(teacher_features, student_features),
        "classification_feature_dim": {
            "teacher": int(teacher_features.shape[1]),
            "student": int(student_features.shape[1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2128, 2129])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis/teacher_student_known_val_geometry_20261114.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    results = [run_seed(seed, device, args.batch_size) for seed in args.seeds]
    payload = {
        "purpose": "descriptive teacher/student feature-geometry diagnostic; no training",
        "device": str(device),
        "labels_used": "known-validation membership only; no unknown labels or test samples",
        "metrics": {
            "cosine_distance_spearman": "Spearman correlation between teacher and student pairwise cosine distances",
            "linear_cka": "linear centered-kernel alignment between feature matrices",
            "teacher_student_knn_overlap": "mean fraction of each model's k nearest known-validation neighbors shared",
        },
        "seeds": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
