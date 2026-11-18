"""Diagnose whether two-view rejection flips are concentrated at the threshold.

This is a frozen-model analysis.  It reuses the existing rejection-view audit
protocol and does not train, calibrate with test labels, or change any model
checkpoint.  Test known/unknown labels are used only for the final cohort
 summaries and margin strata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_rejection_view_stability import (  # noqa: E402
    _clone_with_transform,
    _extract_two_views,
    _fit_rejector,
    _outputs_from_view,
    _two_view_loader,
)
from novel_discovery.data import build_data_bundle  # noqa: E402
from novel_discovery.models import build_model  # noqa: E402
from novel_discovery.pipeline import (  # noqa: E402
    attach_feature_rejector_score,
    attach_known_support_score,
)


def _quantile_bins(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return fixed quartile strata, retaining duplicate-boundary samples."""
    q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    return [
        ("q0_q25", values <= q25),
        ("q25_q50", (values > q25) & (values <= q50)),
        ("q50_q75", (values > q50) & (values <= q75)),
        ("q75_q100", values > q75),
    ]


def _summarize(mask, score_a, score_b, threshold, margin):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"count": 0}
    delta = np.abs(score_a - score_b)
    flip = (score_a > threshold) != (score_b > threshold)
    return {
        "count": int(mask.sum()),
        "threshold_margin_median": float(np.median(margin[mask])),
        "threshold_margin_q25": float(np.quantile(margin[mask], 0.25)),
        "threshold_margin_q75": float(np.quantile(margin[mask], 0.75)),
        "score_abs_delta_mean": float(delta[mask].mean()),
        "score_abs_delta_q95": float(np.quantile(delta[mask], 0.95)),
        "candidate_flip_rate": float(flip[mask].mean()),
        "score_view_pearson": float(np.corrcoef(score_a[mask], score_b[mask])[0, 1])
        if int(mask.sum()) > 1
        else None,
    }


def audit_case(case, device, batch_size, num_workers, *, threshold_source="refit"):
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
    model = build_model(
        int(cfg["num_known"]),
        backbone=str(cfg.get("student_backbone", cfg.get("backbone", "resnet18"))),
        proj_dim=int(cfg.get("proj_dim", 128)),
        dropout=float(cfg.get("dropout", 0.2)),
        pretrained=False,
        cifar_stem=bool(cfg.get("cifar_stem", False)),
        rejection_feature_dim=int(cfg.get("rejection_feature_dim", 128)),
        independent_rejection_encoder=bool(cfg.get("independent_rejection_encoder", True)),
        independent_rejection_encoder_init=str(cfg.get("independent_rejection_encoder_init", "shared_backbone")),
        independent_rejection_input_norm=str(cfg.get("independent_rejection_input_norm", "none")),
    ).to(device)
    checkpoint = torch.load(case["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    support, rejector, threshold, fit_diag = _fit_rejector(
        argparse.Namespace(batch_size=batch_size, num_workers=num_workers),
        cfg,
        bundle,
        model,
        device,
        threshold_source=threshold_source,
        discover_dir=case["discover_dir"],
    )
    test_aug = _clone_with_transform(
        bundle.test,
        int(cfg["image_size"]),
        str(cfg.get("transform_profile", "cifar")),
        train=True,
    )
    views = _extract_two_views(
        model,
        _two_view_loader(test_aug, batch_size, num_workers),
        device,
    )
    out_a = _outputs_from_view(views, "a")
    out_b = _outputs_from_view(views, "b")
    feature_mode = str(cfg.get("rejector_feature_mode", "support_augmented"))
    for outputs in (out_a, out_b):
        attach_known_support_score(outputs, support)
        attach_feature_rejector_score(outputs, rejector, feature_mode=feature_mode)
    score_a = np.asarray(out_a["feature_rejector_score"], dtype=np.float64)
    score_b = np.asarray(out_b["feature_rejector_score"], dtype=np.float64)
    is_known = views["is_known"].astype(bool)
    unknown = ~is_known
    margin = np.minimum(np.abs(score_a - threshold), np.abs(score_b - threshold))
    unknown_margin = margin[unknown]
    result = {
        "protocol": {
            "seed": int(cfg["seed"]),
            "arm": case["arm"],
            "checkpoint": str(case["checkpoint"]),
            "split_path": str(split_path),
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "formal_threshold_reused": threshold_source == "formal",
            "rejector_weights_recovered": False,
            "rejector_refit": True,
            "feature_mode": feature_mode,
            "test_labels_used_only_for_final_cohort_strata": True,
        },
        "fit": fit_diag,
        "all": _summarize(np.ones(len(is_known), dtype=bool), score_a, score_b, threshold, margin),
        "known": _summarize(is_known, score_a, score_b, threshold, margin),
        "unknown": _summarize(unknown, score_a, score_b, threshold, margin),
        "unknown_margin_quantiles": {
            "q25": float(np.quantile(unknown_margin, 0.25)),
            "q50": float(np.quantile(unknown_margin, 0.50)),
            "q75": float(np.quantile(unknown_margin, 0.75)),
        },
        "unknown_margin_strata": [],
    }
    for name, local_mask in _quantile_bins(unknown_margin):
        global_mask = np.zeros(len(is_known), dtype=bool)
        global_mask[np.flatnonzero(unknown)[local_mask]] = True
        row = {"stratum": name}
        row.update(_summarize(global_mask, score_a, score_b, threshold, margin))
        result["unknown_margin_strata"].append(row)
    return result


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
    parser.add_argument("--seeds", nargs="+", type=int, default=[2121, 2124, 2125, 2127])
    parser.add_argument("--arms", nargs="+", default=["static_control", "late_decay"])
    args = parser.parse_args()
    run_root = Path(args.run_root)
    reports = {}
    for seed in args.seeds:
        for arm in args.arms:
            run = run_root / f"rejection_ce_decay_full_s{seed}_{arm}"
            case = {
                "arm": arm,
                "checkpoint": str(run / "student.pt"),
                "discover_dir": str(run_root / f"rejection_ce_decay_full_s{seed}_{arm}_discover"),
            }
            name = f"s{seed}_{arm}"
            print(f"[audit] {name}", flush=True)
            reports[name] = audit_case(
                case,
                torch.device(args.device),
                args.batch_size,
                args.num_workers,
                threshold_source=args.threshold_source,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
