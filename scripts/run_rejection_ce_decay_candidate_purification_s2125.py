"""Evaluation-only candidate-pool purification pair for the CE-decay study.

The trained checkpoints are fixed.  Within each checkpoint, only the
clustering-only candidate purifier changes: no purification versus a
classwise head-uncertainty filter that keeps 80% of the candidates.  The
filter was chosen before reading the seed-2125 test metrics because it had
improved candidate purity and clustering on earlier fresh audits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 2125
SPLIT = ROOT / "splits/semantic_hard/seed2125/cifar100_60_40_semantic_hard.json"
KEEP_RATIO = 0.8
ARMS = {
    "static_control": ROOT / "runs/rejection_ce_decay_full_s2125_static_control",
    "late_decay": ROOT / "runs/rejection_ce_decay_full_s2125_late_decay",
}
MODES = ("none", "head_uncertainty_classwise")

from train import discover, parse_args  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(arm: str, mode: str) -> dict[str, Any]:
    source = ARMS[arm]
    source_discover = source.with_name(source.name + "_discover")
    source_config_path = source_discover / "discover_config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError(f"missing source discovery config: {source_config_path}")
    source_config = read_json(source_config_path)
    checkpoint = source / "student.pt"
    suffix = "none" if mode == "none" else "classwise_r08"
    # v2 keeps the earlier failed-attempt artifacts intact while ensuring all
    # four cells are regenerated from the same corrected discover recipe.
    output = ROOT / "runs" / f"rejection_ce_decay_candidate_purification_s{SEED}_v2_{arm}_{suffix}"
    report_path = output / "discovery_report.json"
    if not report_path.is_file():
        if output.exists():
            raise FileExistsError(f"refusing to reuse incomplete output: {output}")
        args = parse_args(["discover"])
        for key, value in source_config.items():
            # The source file records the original command.  Do not overwrite
            # parse_args(["discover"]) with "train_student", otherwise the
            # run is saved as train_student_config.json and cannot be audited
            # as a discovery run.
            if key == "command":
                continue
            if hasattr(args, key):
                setattr(args, key, value)
        args.seed = SEED
        args.model_seed = None
        args.work_dir = str(output)
        args.student_ckpt = str(checkpoint)
        args.split_path = str(SPLIT)
        args.score_mode = "feature_rejector"
        args.cluster_feature = "feature_pca"
        args.cluster_selection = "silhouette"
        args.cluster_k = "auto"
        args.skip_clustering = False
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.candidate_purify = mode
        args.candidate_keep_ratio = KEEP_RATIO if mode != "none" else 1.0
        print(f"[discover] arm={arm} candidate_purify={mode}", flush=True)
        discover(args)
    config_path = output / "discover_config.json"
    if not config_path.is_file():
        # The first failed attempt wrote the correct discovery outputs but
        # inherited the wrong command name.  Its legacy config.json is still
        # sufficient for the audit and avoids rerunning the completed arm.
        config_path = output / "config.json"
    config = read_json(config_path)
    report = read_json(report_path)
    calibration = read_json(output / "calibration_report.json")
    if config.get("candidate_purify") != mode:
        raise RuntimeError(f"candidate purifier did not materialize: {output}")
    if config.get("score_mode") != "feature_rejector":
        raise RuntimeError(f"score mode changed unexpectedly: {output}")
    if calibration.get("rejector_protocol", {}).get(
        "test_labels_used_for_training_or_calibration"
    ) is not False:
        raise RuntimeError(f"label-flow audit failed: {output}")
    return {"output": str(output), "config": config, "metrics": report, "calibration": calibration}


def main() -> None:
    results = {arm: {mode: run(arm, mode) for mode in MODES} for arm in ARMS}
    output = ROOT / "analysis" / "rejection_ce_decay_candidate_purification_s2125.json"
    payload = {
        "protocol": "seed-2125 evaluation-only candidate-pool purification pair",
        "seed": SEED,
        "training_performed": False,
        "score_mode": "feature_rejector",
        "cluster_feature": "feature_pca",
        "cluster_selection": "silhouette",
        "keep_ratio": KEEP_RATIO,
        "single_changed_factor": "candidate_purify: none vs head_uncertainty_classwise",
        "criterion_selected_using_test_or_mixed_labels": False,
        "arms": list(ARMS),
        "modes": list(MODES),
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
