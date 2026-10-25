"""Audit known-validation calibration stability for a frozen fusion pair.

This is evaluation-only. It re-runs discovery on one frozen student checkpoint
solely to export the known-validation score arrays, then performs paired
known-only bootstrap/OOB analysis. It does not train, select a method, tune a
fusion weight, or use mixed-pool/test labels for any decision.
"""

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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_discover_args(config_path: Path, work_dir: Path, seed: int, device: str):
    values = read_json(config_path)
    args = parse_args(["discover"])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.device = device
    args.work_dir = str(work_dir)
    args.save_calibration_scores = True
    args.threshold_policy = "known_coverage"
    args.target_known_coverage = 0.95
    args.open_val_ratio = 0.0
    return args


def ensure_export(
    source_config: Path,
    output_dir: Path,
    split_path: Path,
    student_ckpt: Path,
    score_mode: str,
    seed: int,
    device: str,
) -> Path:
    score_path = output_dir / "calibration_scores.npz"
    if score_path.is_file():
        return score_path
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse incomplete stability output: {output_dir}"
        )
    args = load_discover_args(source_config, output_dir, seed, device)
    args.split_path = str(split_path)
    args.student_ckpt = str(student_ckpt)
    args.score_mode = score_mode
    args.independent_rejection_encoder = True
    args.independent_rejection_encoder_init = "shared_backbone"
    args.independent_rejection_input_norm = "none"
    args.skip_clustering = True
    print(f"[export] score_mode={score_mode} output={output_dir}", flush=True)
    discover(args)
    if not score_path.is_file():
        raise RuntimeError(f"calibration score export was not created: {score_path}")
    return score_path


def load_export(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"score", "predicted_class", "source_index"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        scores = np.asarray(payload["score"], dtype=np.float64)
        classes = np.asarray(payload["predicted_class"], dtype=np.int64)
        indices = np.asarray(payload["source_index"], dtype=np.int64)
    if scores.ndim != 1 or classes.shape != scores.shape or indices.shape != scores.shape:
        raise ValueError(f"unaligned score export: {path}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"source indices are not unique: {path}")
    metadata_path = path.with_name("calibration_report.json")
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    if metadata.get("score_export", {}).get("labels_exported", False):
        raise ValueError(f"labels were exported in calibration metadata: {path}")
    return scores, indices, metadata


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(",".join(map(str, indices.tolist())).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2089)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("splits_cifar100_semantic_hard_seed2089/cifar100_60_40_semantic_hard.json"),
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("runs/shared_rejection_fusion_fresh_s2089_baseline_discover/discover_config.json"),
    )
    parser.add_argument(
        "--student-ckpt",
        type=Path,
        default=Path("runs/shared_rejection_fusion_fresh_s2089_baseline/student.pt"),
    )
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--tag", default="shared_rejection_fusion_calibration_s2089")
    args = parser.parse_args()

    split_path = resolve(args.split_path)
    source_config = resolve(args.source_config)
    student_ckpt = resolve(args.student_ckpt)
    for path in (split_path, source_config, student_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = ROOT / "runs"
    control_export = ensure_export(
        source_config,
        base / f"{args.tag}_control",
        split_path,
        student_ckpt,
        "feature_rejector",
        args.seed,
        args.device,
    )
    fusion_export = ensure_export(
        source_config,
        base / f"{args.tag}_fusion",
        split_path,
        student_ckpt,
        "feature_rejector_uncertainty_fusion",
        args.seed,
        args.device,
    )

    control_scores, control_indices, control_meta = load_export(control_export)
    fusion_scores, fusion_indices, fusion_meta = load_export(fusion_export)
    if not np.array_equal(control_indices, fusion_indices):
        raise ValueError("control/fusion calibration exports are not source-index aligned")

    target = 0.95
    control_audit = audit(
        control_scores,
        target,
        args.repeats,
        seed=args.seed + 1000,
        predicted_classes=None,
    )
    fusion_audit = audit(
        fusion_scores,
        target,
        args.repeats,
        seed=args.seed + 1000,
        predicted_classes=None,
    )
    rng = np.random.default_rng(args.seed + 2000)
    deltas = np.empty(args.repeats, dtype=np.float64)
    n = len(control_scores)
    for repeat in range(args.repeats):
        sampled = rng.integers(0, n, size=n)
        deltas[repeat] = (
            np.quantile(fusion_scores[sampled], target, method="linear")
            - np.quantile(control_scores[sampled], target, method="linear")
        )

    result = {
        "question": "How stable are the known-validation q95 thresholds for fixed uncertainty fusion on a frozen seed-2089 checkpoint?",
        "seed": args.seed,
        "protocol": "known-validation-only paired bootstrap/OOB; evaluation-only",
        "test_or_mixed_pool_labels_used_for_decision": False,
        "fusion_weight_tuned": False,
        "source_index_sha256": index_hash(control_indices),
        "sample_count": int(n),
        "control_score_mode": "feature_rejector",
        "fusion_score_mode": "feature_rejector_uncertainty_fusion",
        "control_export": str(control_export),
        "fusion_export": str(fusion_export),
        "control_metadata": control_meta.get("score_export", {}),
        "fusion_metadata": fusion_meta.get("score_export", {}),
        "control_audit": control_audit,
        "fusion_audit": fusion_audit,
        "paired_threshold_delta_fusion_minus_control": {
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas, ddof=1)),
            "q025": float(np.quantile(deltas, 0.025)),
            "median": float(np.quantile(deltas, 0.5)),
            "q975": float(np.quantile(deltas, 0.975)),
        },
        "interpretation_limit": (
            "This measures finite-sample known-validation calibration stability only. "
            "It does not estimate test coverage, unknown rejection, or cross-split robustness."
        ),
    }
    output = ROOT / "analysis" / f"{args.tag}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sample_count": n}, ensure_ascii=False))


if __name__ == "__main__":
    main()
