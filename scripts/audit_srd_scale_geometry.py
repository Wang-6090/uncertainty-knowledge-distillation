"""Read-only known-validation geometry audit for standard KD and SRD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import OpenSetCIFAR100, build_transforms, split_known_indices  # noqa: E402
from scripts.diagnose_teacher_student_geometry import checkpoint_payload, make_model  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@torch.inference_mode()
def extract(model, loader, device):
    feature_parts = []
    label_parts = []
    id_parts = []
    for batch in loader:
        image = batch[0].to(device, non_blocking=True)
        output = model(image)
        feature_parts.append(output["features"].float().cpu())
        label_parts.append(batch[1].cpu())
        id_parts.append(batch[4].cpu())
    return torch.cat(feature_parts), torch.cat(label_parts), torch.cat(id_parts)


def summarize(features: torch.Tensor, labels: torch.Tensor, classes: int) -> dict[str, Any]:
    raw_norms = features.norm(dim=1).double().numpy()
    unit = F.normalize(features.double(), dim=1)
    centers = []
    within = []
    class_means = []
    for class_id in range(classes):
        selected = unit[labels == class_id]
        if not len(selected):
            continue
        center = F.normalize(selected.mean(dim=0, keepdim=True), dim=1).squeeze(0)
        centers.append(center)
        within.extend((1.0 - selected @ center).tolist())
        class_means.append(selected.mean(dim=0))
    center_matrix = torch.stack(centers)
    center_similarity = center_matrix @ center_matrix.T
    center_similarity.fill_diagonal_(-torch.inf)
    nearest_center_distance = 1.0 - center_similarity.max(dim=1).values

    # Decompose normalized-feature scatter using the known validation labels.
    class_means_tensor = torch.stack(class_means)
    overall_mean = unit.mean(dim=0, keepdim=True)
    between_scatter = float(
        ((class_means_tensor - overall_mean).square().sum(dim=1)).mean().item()
    )
    total_scatter = float(((unit - overall_mean).square().sum(dim=1)).mean().item())
    return {
        "samples": int(len(features)),
        "raw_feature_norm_mean": float(raw_norms.mean()),
        "raw_feature_norm_std": float(raw_norms.std()),
        "raw_feature_norm_q05_q50_q95": [
            float(value) for value in np.quantile(raw_norms, [0.05, 0.5, 0.95])
        ],
        "normalized_within_cosine_distance_mean": float(np.mean(within)),
        "normalized_within_cosine_distance_q95": float(np.quantile(within, 0.95)),
        "normalized_nearest_center_cosine_distance_mean": float(
            nearest_center_distance.mean().item()
        ),
        "normalized_center_margin": float(nearest_center_distance.mean().item() - np.mean(within)),
        "normalized_between_total_scatter_ratio": between_scatter / max(total_scatter, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2170)
    parser.add_argument("--tag", default="team_a_srd_20261123")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "analysis/team_a_srd_geometry_20261123_s2170.json",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output}")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    split_path = ROOT / f"splits/semantic_hard/seed{args.seed}/cifar100_60_40_semantic_hard.json"
    split = read_json(split_path)
    if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != args.seed:
        raise ValueError("split metadata does not match requested seed/protocol")
    known_classes = list(split["known_classes"])
    if len(known_classes) != 60 or len(split["novel_classes"]) != 40:
        raise ValueError("expected a disjoint CIFAR-100 60/40 split")

    run_dirs = {
        "standard": ROOT / "runs" / f"{args.tag}_s{args.seed}_standard",
        "srd": ROOT / "runs" / f"{args.tag}_s{args.seed}_srd",
    }
    configs = {arm: read_json(path / "train_student_config.json") for arm, path in run_dirs.items()}
    checkpoints = {arm: checkpoint_payload(path / "student.pt") for arm, path in run_dirs.items()}
    reference = configs["standard"]
    for arm in ("standard", "srd"):
        cfg, ckpt = configs[arm], checkpoints[arm]
        for field in ("seed", "split_path", "dataset", "data_root", "image_size", "transform_profile", "known_split_mode", "limit_val"):
            if cfg.get(field) != reference.get(field):
                raise ValueError(f"student validation protocol mismatch for {arm}: {field}")
        if cfg.get("seed") != args.seed or cfg.get("split_path") != str(split_path):
            raise ValueError(f"student config points to a different split: {arm}")
        if float(cfg.get("calibration_ratio", 0.0)) != 0.0:
            raise ValueError(f"student validation subset is not complete: {arm}")
        if ckpt.get("known_classes") != known_classes or ckpt.get("novel_classes") != split["novel_classes"]:
            raise ValueError(f"checkpoint class metadata mismatch: {arm}")

    data_root = Path(reference["data_root"])
    if not data_root.is_absolute():
        data_root = (ROOT / data_root).resolve()
    dataset = OpenSetCIFAR100(
        str(data_root),
        known_classes,
        train=True,
        transform=build_transforms(
            int(reference["image_size"]),
            train=False,
            profile=reference["transform_profile"],
        ),
        download=False,
        include_unknown=False,
    )
    _, validation_indices = split_known_indices(len(dataset), val_ratio=0.1, seed=args.seed)
    loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    output = {}
    shared_ids = None
    for arm in ("standard", "srd"):
        model = make_model(configs[arm], checkpoints[arm], device)
        features, labels, ids = extract(model, loader, device)
        if shared_ids is not None and not torch.equal(shared_ids, ids):
            raise RuntimeError("standard and SRD validation sample order differs")
        shared_ids = ids
        output[arm] = summarize(features, labels, len(known_classes))

    fields = output["standard"].keys()
    output["srd_minus_standard"] = {
        field: (
            [srd_value - standard_value for srd_value, standard_value in zip(output["srd"][field], output["standard"][field])]
            if isinstance(output["srd"][field], list)
            else output["srd"][field] - output["standard"][field]
        )
        for field in fields
        if field != "samples"
    }
    payload = {
        "purpose": "read-only known-validation raw norm and normalized geometry comparison",
        "seed": args.seed,
        "split": str(split_path),
        "data_accessed": "known validation portion of CIFAR-100 training data only",
        "unknown_labels_used": False,
        "transform_profile": reference["transform_profile"],
        "sample_count": int(len(shared_ids)),
        "device": str(device),
        "arms": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
