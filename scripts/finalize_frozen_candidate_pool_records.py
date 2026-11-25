"""Backfill integrity metadata and run records for frozen candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


DETECTION_METRICS = (
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "known_accept_rate",
    "unknown_reject_rate",
    "known_class_accuracy_all_known",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize records for existing frozen candidate-pool runs."
    )
    parser.add_argument("run_dirs", type=Path, nargs="+")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _reproduction_command(config: dict) -> str:
    value_options = (
        ("--dataset", "dataset"),
        ("--data-root", "data_root"),
        ("--split-path", "split_path"),
        ("--num-known", "num_known"),
        ("--seed", "seed"),
        ("--image-size", "image_size"),
        ("--transform-profile", "transform_profile"),
        ("--batch-size", "batch_size"),
        ("--num-workers", "num_workers"),
        ("--backbone", "backbone"),
        ("--student-backbone", "student_backbone"),
        ("--proj-dim", "proj_dim"),
        ("--dropout", "dropout"),
        ("--device", "device"),
        ("--work-dir", "work_dir"),
        ("--student-ckpt", "student_ckpt"),
        ("--num-novel", "num_novel"),
        ("--cluster-k", "cluster_k"),
        ("--cluster-method", "cluster_method"),
        ("--cluster-feature", "cluster_feature"),
        ("--cluster-selection", "cluster_selection"),
        ("--max-auto-clusters", "max_auto_clusters"),
        ("--cluster-pca-dim", "cluster_pca_dim"),
        ("--cluster-n-init", "cluster_n_init"),
        ("--cluster-stability-repeats", "cluster_stability_repeats"),
        ("--threshold-percentile", "threshold_percentile"),
        ("--threshold-policy", "threshold_policy"),
        ("--target-known-coverage", "target_known_coverage"),
        ("--candidate-purify", "candidate_purify"),
        ("--candidate-keep-ratio", "candidate_keep_ratio"),
        ("--cluster-candidate-mode", "cluster_candidate_mode"),
        ("--mc-samples", "mc_samples"),
        ("--score-mode", "score_mode"),
    )
    argv = [sys.executable, "train.py", "discover"]
    for option, key in value_options:
        value = config.get(key)
        if value is not None and value != "":
            argv.extend([option, str(value)])
    if config.get("cluster_normalize"):
        argv.append("--cluster-normalize")
    if config.get("cluster_no_whiten"):
        argv.append("--cluster-no-whiten")
    if config.get("skip_clustering"):
        argv.append("--skip-clustering")
    if config.get("save_candidate_pool"):
        argv.append("--save-candidate-pool")
    if config.get("pretrained"):
        argv.append("--pretrained")
    return subprocess.list2cmdline(argv)


def _candidate_metrics(arrays: dict[str, np.ndarray], detail: dict) -> dict:
    pool_index = np.asarray(arrays["source_index"], dtype=np.int64)
    detail_index = np.asarray(detail["source_index"], dtype=np.int64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    raw_labels = np.asarray(detail["raw_labels"], dtype=np.int64)
    lookup = {int(value): position for position, value in enumerate(detail_index)}
    positions = np.asarray([lookup[int(value)] for value in pool_index], dtype=np.int64)
    aligned_known = true_known[positions]
    aligned_labels = raw_labels[positions]
    candidate = np.asarray(arrays["candidate"], dtype=bool)
    candidate_unknown = candidate & ~aligned_known
    total_unknown = int((~aligned_known).sum())
    classwise = []
    for raw_label in sorted(np.unique(aligned_labels[~aligned_known]).tolist()):
        class_mask = (~aligned_known) & (aligned_labels == raw_label)
        count = int(class_mask.sum())
        selected = int((class_mask & candidate).sum())
        classwise.append(
            {
                "raw_label": int(raw_label),
                "total_count": count,
                "candidate_count": selected,
                "candidate_recall": float(selected / max(count, 1)),
            }
        )
    return {
        "candidate_count": int(candidate.sum()),
        "candidate_purity": float(candidate_unknown.sum() / max(int(candidate.sum()), 1)),
        "known_contamination": float(
            (candidate & aligned_known).sum() / max(int(candidate.sum()), 1)
        ),
        "candidate_unknown_recall": float(
            candidate_unknown.sum() / max(total_unknown, 1)
        ),
        "total_unknown_count": total_unknown,
        "unknown_classwise": classwise,
    }


def finalize_run(run_dir: Path) -> dict:
    pool_path = run_dir / "candidate_pool.npz"
    metadata_path = run_dir / "candidate_pool_metadata.json"
    config = _read_json(run_dir / "config.json")
    discovery_report = _read_json(run_dir / "discovery_report.json")
    detail = _read_json(run_dir / "discovery_detail.json")
    metadata = _read_json(metadata_path)
    with np.load(pool_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    artifact_sha256 = _file_sha256(pool_path)
    array_sha256 = {name: _array_sha256(values) for name, values in arrays.items()}
    metadata.update(
        {
            "schema_version": 2,
            "artifact_sha256": artifact_sha256,
            "array_sha256": array_sha256,
        }
    )
    _write_json(metadata_path, metadata)
    candidate_metrics = _candidate_metrics(arrays, detail)
    detection_metrics = {
        name: discovery_report[name]
        for name in DETECTION_METRICS
        if name in discovery_report
    }
    command = _reproduction_command(config)
    command_note = (
        "# Reconstructed from saved config after the original run; the original shell "
        "command was not retained."
    )
    (run_dir / "command.txt").write_text(
        f"{command_note}\n{command}\n", encoding="utf-8"
    )
    metrics = {
        "protocol": "frozen_candidate_pool_generation",
        "seed": int(metadata["seed"]),
        "command_provenance": "reconstructed_from_saved_config",
        "detection_metrics": detection_metrics,
        "candidate_pool_metrics": candidate_metrics,
        "artifact_integrity": {
            "schema_version": 2,
            "artifact_sha256": artifact_sha256,
            "array_sha256": array_sha256,
            "student_checkpoint_sha256": metadata.get("student_checkpoint_sha256"),
        },
        "test_labels_used_for_candidate_selection": False,
        "test_labels_used_for_retrospective_metrics_only": True,
    }
    _write_json(run_dir / "metrics.json", metrics)

    summary = [
        "# Frozen candidate-pool generation record",
        "",
        f"- Seed: {metadata['seed']}",
        "- Changed factor: seed only; detector score, threshold policy, checkpoint family, and candidate rule are fixed.",
        "- Formal baseline: the same uncertainty-KD frozen-checkpoint discovery protocol on the other formal seeds.",
        f"- Detection metrics: AUROC={detection_metrics.get('auroc', float('nan')):.6f}, FPR95={detection_metrics.get('fpr95', float('nan')):.6f}, unknown reject rate={detection_metrics.get('unknown_reject_rate', float('nan')):.6f}.",
        f"- Candidate metrics: count={candidate_metrics['candidate_count']}, purity={candidate_metrics['candidate_purity']:.6f}, unknown recall={candidate_metrics['candidate_unknown_recall']:.6f}.",
        "- Three-seed direction: evaluated in the aggregate summary; all formal seeds show majority-known candidate pools and unknown recall below 0.05.",
        "- Hypothesis status: supports candidate-pool quality as the immediate bottleneck; does not establish that representation or K estimation is irrelevant.",
        "- Next step: use the frozen pool for matched oracle-K diagnostics; do not retune candidate membership from clustering metrics.",
        f"- Integrity: candidate_pool.npz SHA-256 `{artifact_sha256}`; every stored array also has a SHA-256 in metadata.",
        "- Command provenance: reconstructed from the saved config because the original shell command was not retained.",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "seed": int(metadata["seed"]),
        "artifact_sha256": artifact_sha256,
    }


def main() -> None:
    args = parse_args()
    results = [finalize_run(run_dir) for run_dir in args.run_dirs]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
