"""Audit alternative rejection score modes on frozen student checkpoints.

This runner changes only the discovery-time score mode. It never retrains a
teacher or student, refuses to overwrite output directories, keeps the
known-validation coverage calibration, and validates that mixed-pool hard
labels and test labels are not used before final evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_episode_pseudonovel_pair import (  # noqa: E402
    load_args,
    read_json,
    validate_report,
    write_json,
)
from train import discover  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-discover-config", type=Path, required=True)
    parser.add_argument("--control-ckpt", type=Path, required=True)
    parser.add_argument("--treatment-ckpt", type=Path, required=True)
    parser.add_argument(
        "--score-mode",
        choices=("feature_rejector_uncertainty_fusion", "feature_rejector_uncertainty_conjunction"),
        action="append",
        required=True,
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    options = parser.parse_args()

    source = options.source_discover_config.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_values = read_json(source)
    if source_values.get("threshold_policy") != "known_coverage":
        raise ValueError("source discovery config must use known_coverage")
    if float(source_values.get("target_known_coverage", 0.0)) != 0.95:
        raise ValueError("source discovery config must target 95% known coverage")
    if source_values.get("rejector_training") != "nnpu":
        raise ValueError("source discovery config must use NNPU rejector training")
    if not bool(source_values.get("matched_discovery_pool")):
        raise ValueError("source discovery config must use matched discovery pools")

    arms = {"control": options.control_ckpt.resolve(), "treatment": options.treatment_ckpt.resolve()}
    for checkpoint in arms.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    results = {}
    for arm, checkpoint in arms.items():
        for score_mode in options.score_mode:
            output = ROOT / "runs" / f"{options.tag}_s{options.seed}_{arm}_{score_mode}"
            report_path = output / "discovery_report.json"
            if output.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {output}")
            args = load_args(source, "discover", output, options.seed)
            args.student_ckpt = str(checkpoint)
            args.device = options.device
            args.score_mode = score_mode
            args.open_val_ratio = 0.0
            args.threshold_policy = "known_coverage"
            args.target_known_coverage = 0.95
            args.save_calibration_scores = True
            args.skip_clustering = False
            args.cluster_k = "auto"
            discover(args)
            report = validate_report(report_path)
            results[f"{arm}:{score_mode}"] = {
                key: report.get(key)
                for key in (
                    "auroc",
                    "aupr",
                    "fpr95",
                    "oscr",
                    "known_class_accuracy_all_known",
                    "known_accept_rate",
                    "unknown_reject_rate",
                    "cluster_candidate_count",
                    "cluster_candidate_purity",
                    "cluster_candidate_unknown_recall",
                    "cluster_k_abs_error",
                    "cluster_acc",
                    "cluster_nmi",
                    "cluster_ari",
                    "threshold_policy",
                    "calibration_known_accept_rate",
                    "rejector_protocol",
                    "effective_score_protocol",
                )
            }

    result_path = ROOT / "analysis" / f"{options.tag}_s{options.seed}.json"
    write_json(
        result_path,
        {
            "seed": options.seed,
            "source_discover_config": str(source),
            "student_checkpoints": {name: str(path) for name, path in arms.items()},
            "score_modes": options.score_mode,
            "test_labels_used_for_training_selection_or_calibration": False,
            "results": results,
        },
    )
    print(json.dumps(read_json(result_path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
