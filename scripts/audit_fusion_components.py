"""Audit component geometry of a frozen validation-only fusion score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="calibration_scores.npz with fusion components")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.scores, allow_pickle=False) as payload:
        required = {
            "score",
            "feature_rejector_component_z",
            "uncertainty_component_z",
            "source_index",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"missing fusion component arrays: {sorted(missing)}")
        score = np.asarray(payload["score"], dtype=np.float64)
        rejector = np.asarray(payload["feature_rejector_component_z"], dtype=np.float64)
        uncertainty = np.asarray(payload["uncertainty_component_z"], dtype=np.float64)
        source_index = np.asarray(payload["source_index"], dtype=np.int64)
    if not (score.shape == rejector.shape == uncertainty.shape == source_index.shape):
        raise ValueError("fusion component arrays are not aligned")
    if not np.isfinite(score).all() or not np.isfinite(rejector).all() or not np.isfinite(uncertainty).all():
        raise ValueError("fusion component arrays contain non-finite values")
    result = {
        "input": str(args.scores),
        "sample_count": int(len(score)),
        "source_index_unique": bool(len(np.unique(source_index)) == len(source_index)),
        "fusion_reconstruction_max_abs_error": float(
            np.max(np.abs(score - 0.5 * (rejector + uncertainty)))
        ),
        "component_pearson_correlation": float(np.corrcoef(rejector, uncertainty)[0, 1]),
        "component_rank_correlation": float(
            np.corrcoef(np.argsort(np.argsort(rejector)), np.argsort(np.argsort(uncertainty)))[0, 1]
        ),
        "fraction_rejector_positive": float(np.mean(rejector > 0.0)),
        "fraction_uncertainty_positive": float(np.mean(uncertainty > 0.0)),
        "fraction_both_positive": float(np.mean((rejector > 0.0) & (uncertainty > 0.0))),
        "unique_value_count": {
            "feature_rejector_component_z": int(len(np.unique(rejector))),
            "uncertainty_component_z": int(len(np.unique(uncertainty))),
        },
        "quantiles": {
            "feature_rejector_component_z": np.quantile(rejector, [0.5, 0.9, 0.95, 0.99]).tolist(),
            "uncertainty_component_z": np.quantile(uncertainty, [0.5, 0.9, 0.95, 0.99]).tolist(),
        },
        "labels_used": False,
        "interpretation_limit": (
            "Known-validation component geometry only; this does not estimate unknown rejection "
            "or justify changing the fixed fusion weight."
        ),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
