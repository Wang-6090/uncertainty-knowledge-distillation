"""Audit two-view stability of frozen rejection and classification representations.

This script is diagnostic only.  The frozen student is evaluated on two
independent train-style augmentations of each test image.  The rejector is
refit from the saved discovery protocol using only the known training support
and the mixed unlabeled discovery pool; the score threshold is calibrated on
the known validation/calibration split.  Test known/unknown labels are used
only to stratify the final report and are never used for fitting or threshold
selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import build_data_bundle, build_transforms  # noqa: E402
from novel_discovery.models import build_model  # noqa: E402
from novel_discovery.pipeline import (  # noqa: E402
    attach_feature_rejector_score,
    attach_known_support_score,
    build_loader,
    build_rejector_features,
    calibrate_coverage_threshold,
    extract_outputs,
    fit_known_support_rejector,
    fit_nnpu_feature_rejector,
    select_discovery_candidates,
)


def _clone_with_transform(dataset: Dataset, image_size: int, profile: str, train: bool):
    if isinstance(dataset, Subset):
        cloned = copy.copy(dataset)
        cloned.dataset = _clone_with_transform(dataset.dataset, image_size, profile, train)
        return cloned
    if hasattr(dataset, "base"):
        cloned = copy.copy(dataset)
        cloned.base = copy.copy(dataset.base)
        cloned.base.transform = build_transforms(image_size, train=train, profile=profile)
        return cloned
    raise TypeError(f"unsupported dataset wrapper: {type(dataset)!r}")


class TwoAugmentedViews(Dataset):
    """Return two independently sampled augmentations and stable metadata."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        first = self.dataset[index]
        second = self.dataset[index]
        return first[0], second[0], first[1], first[2], first[3], first[4]


def _two_view_loader(dataset: Dataset, batch_size: int, num_workers: int):
    return DataLoader(
        TwoAugmentedViews(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _extract_two_views(model, loader, device):
    chunks = {key: [] for key in (
        "features_a", "features_b", "rejection_a", "rejection_b",
        "logits_a", "logits_b", "uncertainty_a", "uncertainty_b",
        "labels", "raw_labels", "is_known",
    )}
    model.eval()
    for images_a, images_b, labels, raw_labels, is_known, _indices in loader:
        out_a = model(images_a.to(device), stochastic=False)
        out_b = model(images_b.to(device), stochastic=False)
        for key, value in {
            "features_a": out_a["features"],
            "features_b": out_b["features"],
            "rejection_a": out_a["rejection_features"],
            "rejection_b": out_b["rejection_features"],
            "logits_a": out_a["logits"],
            "logits_b": out_b["logits"],
            "uncertainty_a": out_a["uncertainty"],
            "uncertainty_b": out_b["uncertainty"],
            "labels": labels,
            "raw_labels": raw_labels,
            "is_known": is_known,
        }.items():
            chunks[key].append(value.detach().cpu())
    return {key: torch.cat(values).numpy() for key, values in chunks.items()}


def _outputs_from_view(payload, suffix):
    logits = payload[f"logits_{suffix}"].astype(np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-8, None)), axis=1)
    return {
        "features": payload[f"features_{suffix}"].astype(np.float32),
        "rejection_features": payload[f"rejection_{suffix}"].astype(np.float32),
        "logits": logits,
        "probs": probs.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "head_uncertainty": payload[f"uncertainty_{suffix}"].astype(np.float32),
        "labels": payload["labels"].astype(np.int64),
        "raw_labels": payload["raw_labels"].astype(np.int64),
        "is_known": payload["is_known"].astype(bool),
    }


def _cosine(a, b):
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return np.sum(a * b, axis=1)


def _summarize_group(mask, payload, out_a, out_b, score_a, score_b, threshold):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"count": 0}
    pred_a = out_a["logits"].argmax(axis=1)
    pred_b = out_b["logits"].argmax(axis=1)
    score_delta = np.abs(score_a - score_b)
    return {
        "count": int(mask.sum()),
        "classification_cosine_mean": float(_cosine(out_a["features"][mask], out_b["features"][mask]).mean()),
        "rejection_cosine_mean": float(_cosine(out_a["rejection_features"][mask], out_b["rejection_features"][mask]).mean()),
        "classification_prediction_flip_rate": float(np.mean(pred_a[mask] != pred_b[mask])),
        "head_uncertainty_abs_delta_mean": float(np.abs(out_a["head_uncertainty"][mask] - out_b["head_uncertainty"][mask]).mean()),
        "entropy_abs_delta_mean": float(np.abs(out_a["entropy"][mask] - out_b["entropy"][mask]).mean()),
        "rejector_score_abs_delta_mean": float(score_delta[mask].mean()),
        "rejector_score_abs_delta_q95": float(np.quantile(score_delta[mask], 0.95)),
        "candidate_flip_rate": float(
            np.mean((score_a[mask] > threshold) != (score_b[mask] > threshold))
        ),
    }


def _saved_formal_threshold(discover_dir: Path) -> float:
    """Read the threshold saved by the formal discovery run.

    The formal pipeline writes the scalar known-coverage threshold to
    ``calibration_report.json``.  Keeping this reader separate makes it
    explicit when a diagnostic reuses the formal operating point instead of
    silently recalibrating it.
    """
    report_path = discover_dir / "calibration_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"formal calibration report not found: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    value = report.get("calibrated_threshold")
    if value is None:
        raise KeyError(
            f"{report_path} does not contain calibrated_threshold; "
            "refusing to substitute a newly calibrated threshold"
        )
    return float(value)


def _fit_rejector(args, cfg, bundle, model, device, *, threshold_source="refit", discover_dir=None):
    known_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
    pool_loader = build_loader(bundle.discovery_pool, args.batch_size, False, args.num_workers)
    calibration_dataset = bundle.calibration if bundle.calibration is not None else bundle.val
    calibration_loader = build_loader(calibration_dataset, args.batch_size, False, args.num_workers)
    known = extract_outputs(model, known_loader, device, mc_samples=1)
    pool = extract_outputs(model, pool_loader, device, mc_samples=1)
    calibration = extract_outputs(model, calibration_loader, device, mc_samples=1)
    support = fit_known_support_rejector(known, quantile=float(cfg.get("support_quantile", 0.95)))
    for outputs in (known, pool, calibration):
        attach_known_support_score(outputs, support)
    candidate_mask = select_discovery_candidates(
        torch.from_numpy(pool["logits"]),
        torch.from_numpy(pool["head_uncertainty"]),
        ratio=float(cfg.get("rejector_candidate_ratio", 0.25)),
        mode=str(cfg.get("rejector_candidate_mode", "entropy_uncertainty")),
    ).numpy()
    selected_pool = {
        key: value[candidate_mask]
        if isinstance(value, np.ndarray) and value.shape[:1] == candidate_mask.shape
        else value
        for key, value in pool.items()
    }
    rejector = fit_nnpu_feature_rejector(
        known,
        selected_pool,
        known_prior=float(cfg.get("rejector_known_prior", 0.2)),
        max_samples=(None if int(cfg.get("rejector_max_samples", 0)) <= 0 else int(cfg["rejector_max_samples"])),
        iterations=max(50, int(cfg.get("rejector_pu_iterations", 4)) * 75),
        feature_mode=str(cfg.get("rejector_feature_mode", "support_augmented")),
        stratified_known=bool(cfg.get("rejector_stratified_known", False)),
        risk_mode=str(cfg.get("rejector_nnpu_risk", "nu_corrected")),
        model_type=str(cfg.get("rejector_nnpu_model", "linear")),
        seed=int(cfg["seed"]),
        score_margin_weight=float(cfg.get("rejector_score_margin_weight", 0.0)),
        score_margin=float(cfg.get("rejector_score_margin", 0.5)),
    )
    attach_feature_rejector_score(calibration, rejector, feature_mode=str(cfg.get("rejector_feature_mode", "support_augmented")))
    refit_threshold, threshold_diag = calibrate_coverage_threshold(
        calibration["feature_rejector_score"],
        target_known_coverage=float(cfg.get("target_known_coverage", 0.95)),
    )
    formal_threshold = None
    if threshold_source == "formal":
        if discover_dir is None:
            raise ValueError("discover_dir is required when threshold_source='formal'")
        formal_threshold = _saved_formal_threshold(Path(discover_dir))
        threshold = formal_threshold
    elif threshold_source == "refit":
        threshold = float(refit_threshold)
    else:
        raise ValueError(f"unsupported threshold_source: {threshold_source}")
    return support, rejector, float(threshold), {
        "calibration_known_accept_rate": float(threshold_diag.get("known_accept_rate", np.nan)),
        "pool_size": int(len(pool["features"])),
        "pool_candidate_count": int(candidate_mask.sum()),
        "threshold_source": threshold_source,
        "refit_threshold": float(refit_threshold),
        "formal_threshold": formal_threshold,
        "rejector_weights_recovered": False,
        "rejector_refit": True,
    }


def audit_case(args, case, *, threshold_source="refit"):
    cfg = json.loads((Path(case["discover_dir"]) / "config.json").read_text(encoding="utf-8"))
    split_path = Path(cfg["split_path"])
    if not split_path.exists():
        split_path = ROOT / "splits" / "semantic_hard" / f"seed{cfg['seed']}" / "cifar100_60_40_semantic_hard.json"
    data_root = Path(cfg.get("data_root", "data"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    bundle = build_data_bundle(
        "cifar100", str(data_root), int(cfg["num_known"]), int(cfg["seed"]),
        int(cfg["image_size"]), split_path=str(split_path),
        discovery_pool_mode=str(cfg.get("discovery_pool_mode", "mixed")),
        mixed_known_pool_ratio=float(cfg.get("mixed_known_pool_ratio", 0.2)),
        matched_discovery_pool=bool(cfg.get("matched_discovery_pool", True)),
        matched_pool_size=int(cfg.get("matched_pool_size", 5400)),
        matched_known_prior=float(cfg.get("matched_known_prior", 0.2)),
        calibration_ratio=float(cfg.get("calibration_ratio", 0.25)),
        transform_profile=str(cfg.get("transform_profile", "cifar")),
    )
    device = torch.device(args.device)
    model = build_model(
        int(cfg["num_known"]), backbone=str(cfg.get("student_backbone", cfg.get("backbone", "resnet18"))),
        proj_dim=int(cfg.get("proj_dim", 128)), dropout=float(cfg.get("dropout", 0.2)),
        pretrained=False, cifar_stem=bool(cfg.get("cifar_stem", False)),
        rejection_feature_dim=int(cfg.get("rejection_feature_dim", 128)),
        independent_rejection_encoder=bool(cfg.get("independent_rejection_encoder", True)),
        independent_rejection_encoder_init=str(cfg.get("independent_rejection_encoder_init", "shared_backbone")),
        independent_rejection_input_norm=str(cfg.get("independent_rejection_input_norm", "none")),
    ).to(device)
    checkpoint = torch.load(case["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    support, rejector, threshold, fit_diag = _fit_rejector(
        args,
        cfg,
        bundle,
        model,
        device,
        threshold_source=threshold_source,
        discover_dir=case["discover_dir"],
    )
    test_aug = _clone_with_transform(bundle.test, int(cfg["image_size"]), str(cfg.get("transform_profile", "cifar")), train=True)
    views = _extract_two_views(model, _two_view_loader(test_aug, args.batch_size, args.num_workers), device)
    out_a = _outputs_from_view(views, "a")
    out_b = _outputs_from_view(views, "b")
    feature_mode = str(cfg.get("rejector_feature_mode", "support_augmented"))
    for outputs in (out_a, out_b):
        attach_known_support_score(outputs, support)
        attach_feature_rejector_score(outputs, rejector, feature_mode=feature_mode)
    score_a = out_a["feature_rejector_score"]
    score_b = out_b["feature_rejector_score"]
    candidate_a = score_a > threshold
    candidate_b = score_b > threshold
    known = views["is_known"].astype(bool)
    unknown = ~known
    candidate_ref = (candidate_a | candidate_b)
    summary = {
        "protocol": {
            "seed": int(cfg["seed"]),
            "arm": case["arm"],
            "checkpoint": str(case["checkpoint"]),
            "split_path": str(split_path),
            "two_independent_train_style_views": True,
            "rejector_fit_uses_test_labels": False,
            "test_labels_used_only_for_cohort_statistics": True,
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "formal_threshold_reused": threshold_source == "formal",
            "rejector_weights_recovered": False,
            "rejector_refit": True,
            "feature_mode": feature_mode,
        },
        "fit": fit_diag,
        "cohort_sizes": {
            "all": int(len(known)), "known": int(known.sum()), "unknown": int(unknown.sum()),
            "unknown_candidate_union": int((unknown & candidate_ref).sum()),
        },
        "all": _summarize_group(np.ones(len(known), dtype=bool), views, out_a, out_b, score_a, score_b, threshold),
        "known": _summarize_group(known, views, out_a, out_b, score_a, score_b, threshold),
        "unknown": _summarize_group(unknown, views, out_a, out_b, score_a, score_b, threshold),
        "unknown_candidate_union": _summarize_group(unknown & candidate_ref, views, out_a, out_b, score_a, score_b, threshold),
        "unknown_non_candidate_union": _summarize_group(unknown & ~candidate_ref, views, out_a, out_b, score_a, score_b, threshold),
        "candidate_counts": {
            "view_a": int((unknown & candidate_a).sum()),
            "view_b": int((unknown & candidate_b).sum()),
            "union": int((unknown & candidate_ref).sum()),
            "flip": int((unknown & (candidate_a != candidate_b)).sum()),
        },
    }
    return summary


def _case_from_dirs(run_root: Path):
    cases = []
    for seed in (2121, 2124, 2125, 2127):
        for arm in ("static_control", "late_decay"):
            run = run_root / f"rejection_ce_decay_full_s{seed}_{arm}"
            cases.append({
                "name": f"s{seed}_{arm}", "seed": seed, "arm": arm,
                "checkpoint": str(run / "student.pt"),
                "discover_dir": str(run_root / f"rejection_ce_decay_full_s{seed}_{arm}_discover"),
            })
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root", default=str(ROOT / "runs"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--threshold-source",
        choices=("formal", "refit"),
        default="refit",
        help="use the saved formal threshold or recalibrate on the diagnostic refit",
    )
    args = parser.parse_args()
    reports = {}
    for case in _case_from_dirs(Path(args.run_root)):
        print(f"[audit] {case['name']}", flush=True)
        reports[case["name"]] = audit_case(
            args, case, threshold_source=args.threshold_source
        )
        print(json.dumps({
            "unknown_candidate_flip_rate": reports[case["name"]]["unknown"]["candidate_flip_rate"],
            "unknown_rejection_cosine": reports[case["name"]]["unknown"]["rejection_cosine_mean"],
        }), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
