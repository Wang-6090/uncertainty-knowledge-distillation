"""Re-evaluate the target-repair screening checkpoints with its corrected score protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, parse_args


TAG = "team_a_uncertainty_target_repair_20261125"
SEED = 2175
ARMS = ("standard", "target_repair", "uncertainty")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    result_path = ROOT / "analysis" / f"{TAG}_s{SEED}.json"
    original = read_json(result_path)
    corrected_arms = {}
    for arm in ARMS:
        source_dir = ROOT / "runs" / f"{TAG}_s{SEED}_{arm}"
        source_discover = source_dir.with_name(source_dir.name + "_discover")
        corrected_dir = source_dir.with_name(source_dir.name + "_corrected_discover")
        if corrected_dir.exists():
            raise FileExistsError(f"refusing to overwrite {corrected_dir}")
        source_config = read_json(source_discover / "discover_config.json")
        args = parse_args(["discover"])
        for key, value in source_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.seed = SEED
        args.work_dir = str(corrected_dir)
        args.student_ckpt = str(source_dir / "student.pt")
        args.score_mode = "max_softmax"
        args.rejector_feature_mode = "embedding"
        args.rejection_feature_dim = 0
        args.independent_rejection_encoder = False
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.cluster_k = "auto"
        args.skip_clustering = False
        print(f"[seed={SEED}] corrected max_softmax discovery arm={arm}", flush=True)
        discover(args)
        report = read_json(corrected_dir / "discovery_report.json")
        corrected_arms[arm] = {
            "train_dir": str(source_dir),
            "discover_dir": str(corrected_dir),
            "train_config": original["arms"][arm]["train_config"],
            "metrics": report,
        }

    output = ROOT / "analysis" / f"{TAG}_s{SEED}_corrected.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    corrected = {
        "purpose": "corrected frozen target-repair screening evaluation",
        "protocol": {
            "seed": SEED,
            "training_reused": True,
            "score_mode": "max_softmax",
            "threshold_policy": "known_coverage",
            "target_known_coverage": 0.95,
            "unknown_labels_used_for_training_or_calibration": False,
            "original_incorrect_discovery_result": str(result_path),
        },
        "objective_profile": original["objective_profile"],
        "mechanism_source": "training fields copied from the completed paired training result",
        "arms": corrected_arms,
    }
    write_json(output, corrected)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
