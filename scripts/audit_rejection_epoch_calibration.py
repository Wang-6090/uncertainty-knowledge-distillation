"""Known-validation-only OOB audit for a frozen rejection epoch-budget pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_validation_threshold_stability import audit  # noqa: E402
from train import discover, parse_args  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_scores(seed: int, arm: str, repeats: int, device: str) -> dict[str, Any]:
    split = ROOT / f"splits_cifar100_semantic_hard_seed{seed}" / "cifar100_60_40_semantic_hard.json"
    train_dir = ROOT / "runs" / f"rejection_epoch_budget_s{seed}_{arm}"
    source_config = ROOT / "runs" / f"rejection_epoch_budget_s{seed}_{arm}_discover" / "discover_config.json"
    out_dir = ROOT / "runs" / f"rejection_epoch_calibration_oob_s{seed}_{arm}"
    score_path = out_dir / "calibration_scores.npz"
    for required in (split, train_dir / "student.pt", source_config):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not score_path.is_file():
        if out_dir.exists():
            raise FileExistsError(f"refusing to reuse incomplete audit output: {out_dir}")
        args = parse_args(["discover"])
        for key, value in read_json(source_config).items():
            if hasattr(args, key):
                setattr(args, key, value)
        args.seed = int(seed)
        args.model_seed = None
        args.device = device
        args.work_dir = str(out_dir)
        args.split_path = str(split)
        args.student_ckpt = str(train_dir / "student.pt")
        args.score_mode = "feature_rejector"
        args.threshold_policy = "known_coverage"
        args.target_known_coverage = 0.95
        args.open_val_ratio = 0.0
        args.skip_clustering = True
        args.save_calibration_scores = True
        print(f"[export] seed={seed} arm={arm}", flush=True)
        discover(args)
    with np.load(score_path, allow_pickle=False) as payload:
        required = {"score", "predicted_class", "source_index", "calibrated_threshold"}
        if not required.issubset(payload.files):
            raise ValueError(f"calibration score export missing arrays: {required - set(payload.files)}")
        scores = np.asarray(payload["score"], dtype=np.float64)
        classes = np.asarray(payload["predicted_class"], dtype=np.int64)
        indices = np.asarray(payload["source_index"], dtype=np.int64)
        exported_threshold = float(np.asarray(payload["calibrated_threshold"]).reshape(-1)[0])
    if not (scores.ndim == classes.ndim == indices.ndim == 1 and scores.shape == classes.shape == indices.shape):
        raise ValueError("misaligned calibration export arrays")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("calibration source indices are not unique")
    metadata = read_json(out_dir / "calibration_report.json")
    export_meta = metadata.get("score_export", {})
    digest = hashlib.sha256(",".join(map(str, indices.tolist())).encode("utf-8")).hexdigest()
    if export_meta.get("contains_test_samples") is not False or export_meta.get("labels_exported") is not False:
        raise RuntimeError("calibration export metadata violates label isolation")
    if export_meta.get("source_index_sha256") != digest:
        raise RuntimeError("calibration source-index digest mismatch")
    return {
        "scores": scores,
        "classes": classes,
        "indices": indices,
        "exported_threshold": exported_threshold,
        "calibration_sample_count": len(scores),
        "audit": audit(scores, 0.95, repeats, seed + (0 if arm == "four_epoch" else 100), predicted_classes=classes),
        "export_metadata": export_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2108)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    options = parser.parse_args()
    control = export_scores(options.seed, "four_epoch", options.repeats, options.device)
    treatment = export_scores(options.seed, "eight_epoch", options.repeats, options.device)
    if not np.array_equal(control["indices"], treatment["indices"]):
        raise RuntimeError("paired calibration exports do not use identical validation samples")
    rng = np.random.default_rng(options.seed + 901)
    threshold_delta = np.empty(options.repeats, dtype=np.float64)
    for idx in range(options.repeats):
        sample = rng.integers(0, len(control["scores"]), size=len(control["scores"]))
        threshold_delta[idx] = (
            np.quantile(treatment["scores"][sample], 0.95)
            - np.quantile(control["scores"][sample], 0.95)
        )
    result = {
        "protocol": "frozen-checkpoint known-validation-only paired bootstrap/OOB",
        "seed": options.seed,
        "repeats": options.repeats,
        "training_or_method_selection_performed": False,
        "unknown_or_test_labels_used": False,
        "same_validation_source_indices": True,
        "four_epoch": {k: v for k, v in control.items() if k not in {"scores", "classes", "indices"}},
        "eight_epoch": {k: v for k, v in treatment.items() if k not in {"scores", "classes", "indices"}},
        "paired_threshold_delta_eight_minus_four": {
            "full_sample": treatment["exported_threshold"] - control["exported_threshold"],
            "bootstrap_mean": float(np.mean(threshold_delta)),
            "bootstrap_std": float(np.std(threshold_delta, ddof=1)),
            "q025": float(np.quantile(threshold_delta, 0.025)),
            "q975": float(np.quantile(threshold_delta, 0.975)),
        },
        "interpretation_limit": "Known-validation finite-sample stability only; does not estimate test coverage or unknown rejection.",
    }
    output = resolve(options.output) if options.output else ROOT / "analysis" / f"rejection_epoch_calibration_oob_s{options.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "control_oob": control["audit"]["out_of_bag_known_acceptance"], "treatment_oob": treatment["audit"]["out_of_bag_known_acceptance"], "paired_threshold_delta": result["paired_threshold_delta_eight_minus_four"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
