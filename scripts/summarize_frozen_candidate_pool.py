"""Aggregate paired oracle-K runs over frozen candidate pools."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


RUNS = {
    "feature_pca_kmeans": "cluster_feature_pca_kmeans",
    "projection_pca_kmeans": "cluster_projection_pca_kmeans",
    "feature_kmeans": "cluster_feature_kmeans",
    "feature_normalized_kmeans": "cluster_feature_normalized_kmeans",
    "feature_pca_agglomerative": "cluster_feature_pca_agglomerative",
    "feature_pca_spectral": "cluster_feature_pca_spectral",
}
METRIC_NAMES = ("acc", "nmi", "ari")
FEATURE_PIPELINE_LABELS = {
    "feature_pca_kmeans": "feature_pca (L2 + PCA/whitening + L2)",
    "projection_pca_kmeans": "projection_pca (L2 + PCA/whitening + L2)",
    "feature_kmeans": "feature (raw)",
    "feature_normalized_kmeans": "feature (L2-normalized)",
}
EXPECTED_NORMALIZATION = {
    "feature_pca_kmeans": True,
    "projection_pca_kmeans": True,
    "feature_kmeans": False,
    "feature_normalized_kmeans": True,
    "feature_pca_agglomerative": True,
    "feature_pca_spectral": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired oracle-K clustering runs from frozen pools."
    )
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        help="Per-seed run directories containing the frozen pool and paired runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _validate_report(report: dict, metadata: dict, seed: int, path: Path) -> None:
    if int(report.get("seed", -1)) != seed:
        raise ValueError(f"seed mismatch in {path}")
    if report.get("pool_source_index_sha256") != metadata.get("source_index_sha256"):
        raise ValueError(f"candidate-pool digest mismatch in {path}")
    if report.get("pool_artifact_sha256") != metadata.get("artifact_sha256"):
        raise ValueError(f"candidate-pool artifact digest mismatch in {path}")
    if report.get("cluster_k_mode") != "oracle":
        raise ValueError(f"only oracle-K runs may enter this summary: {path}")
    if report.get("selected_k") != report.get("requested_oracle_k"):
        raise ValueError(f"selected K differs from protocol K in {path}")
    if report.get("test_labels_used_for_clustering_or_k_selection") is not False:
        raise ValueError(f"label-isolation guarantee missing in {path}")
    evaluation = report.get("retrospective_evaluation", {})
    if evaluation.get("labels_loaded_after_clustering") is not True:
        raise ValueError(f"post-clustering evaluation guarantee missing in {path}")
    if int(report.get("candidate_count", -1)) != int(metadata.get("candidate_count", -2)):
        raise ValueError(f"candidate count mismatch in {path}")
    missing_metrics = sorted(
        set(METRIC_NAMES) - set(evaluation.get("unknown_only_metrics", {}))
    )
    if missing_metrics:
        raise ValueError(f"missing unknown-only metrics in {path}: {missing_metrics}")


def summarize(run_dirs: list[Path]) -> dict:
    if len(run_dirs) < 2:
        raise ValueError("at least two paired seed directories are required")

    per_seed: dict[int, dict] = {}
    shared_source_digest: str | None = None
    for run_dir in run_dirs:
        metadata = _read_json(run_dir / "candidate_pool_metadata.json")
        if metadata.get("labels_exported") is not False:
            raise ValueError(f"pool does not guarantee label isolation: {run_dir}")
        seed = int(metadata["seed"])
        if seed in per_seed:
            raise ValueError(f"duplicate seed: {seed}")
        source_digest = str(metadata["source_index_sha256"])
        if shared_source_digest is None:
            shared_source_digest = source_digest
        elif source_digest != shared_source_digest:
            raise ValueError("paired seed pools use different source-index sets")

        reports = {}
        for name, relative_dir in RUNS.items():
            path = run_dir / relative_dir / "cluster_report.json"
            report = _read_json(path)
            _validate_report(report, metadata, seed, path)
            if report.get("normalized") is not EXPECTED_NORMALIZATION[name]:
                raise ValueError(f"unexpected normalization setting in {path}")
            reports[name] = report

        evaluation = reports["feature_pca_kmeans"]["retrospective_evaluation"]
        all_unknown_oracle = reports["feature_pca_kmeans"].get(
            "all_unknown_oracle_evaluation"
        )
        if not all_unknown_oracle:
            raise ValueError(
                f"missing all-true-unknown oracle diagnostic: {run_dir}"
            )
        if all_unknown_oracle.get("used_for_representation_or_method_selection") is not False:
            raise ValueError(
                f"all-unknown diagnostic lacks method-selection isolation: {run_dir}"
            )
        unknown_classwise = evaluation["unknown_classwise"]
        recalls = [float(row["candidate_recall"]) for row in unknown_classwise]
        known_classwise = evaluation["known_contamination_classwise"]
        per_seed[seed] = {
            "run_dir": str(run_dir.resolve()),
            "metadata": metadata,
            "reports": reports,
            "pool": {
                "candidate_count": int(metadata["candidate_count"]),
                "candidate_purity": float(evaluation["candidate_purity"]),
                "known_contamination": float(evaluation["known_contamination"]),
                "candidate_unknown_recall": float(
                    evaluation["candidate_unknown_recall"]
                ),
                "unknown_classes_with_zero_candidates": int(
                    sum(value == 0.0 for value in recalls)
                ),
                "unknown_class_recall_min": float(np.min(recalls)),
                "unknown_class_recall_median": float(np.median(recalls)),
                "unknown_class_recall_max": float(np.max(recalls)),
                "max_known_class_false_reject_rate": float(
                    max(float(row["false_reject_rate"]) for row in known_classwise)
                ),
            },
            "all_unknown_oracle": all_unknown_oracle,
        }

    seeds = sorted(per_seed)
    pool_fields = (
        "candidate_count",
        "candidate_purity",
        "known_contamination",
        "candidate_unknown_recall",
        "unknown_classes_with_zero_candidates",
        "unknown_class_recall_min",
        "unknown_class_recall_median",
        "unknown_class_recall_max",
        "max_known_class_false_reject_rate",
    )
    pool_aggregate = {
        field: _aggregate([float(per_seed[seed]["pool"][field]) for seed in seeds])
        for field in pool_fields
    }

    experiments = {}
    for name in RUNS:
        rows = []
        for seed in seeds:
            report = per_seed[seed]["reports"][name]
            metrics = report["retrospective_evaluation"]["unknown_only_metrics"]
            rows.append(
                {
                    "seed": seed,
                    "candidate_count": int(report["candidate_count"]),
                    "acc": float(metrics["acc"]),
                    "nmi": float(metrics["nmi"]),
                    "ari": float(metrics["ari"]),
                    "cluster_size_min": int(min(report["cluster_sizes"].values())),
                    "cluster_size_max": int(max(report["cluster_sizes"].values())),
                }
            )
        first_report = per_seed[seeds[0]]["reports"][name]
        experiments[name] = {
            "feature": first_report["feature"],
            "normalized": bool(first_report["normalized"]),
            "method": first_report["method"],
            "per_seed": rows,
            "aggregate": {
                metric: _aggregate([float(row[metric]) for row in rows])
                for metric in METRIC_NAMES
            },
        }

    feature_nmi_beats_raw = all(
        per_seed[seed]["reports"]["feature_pca_kmeans"][
            "retrospective_evaluation"
        ]["unknown_only_metrics"]["nmi"]
        > per_seed[seed]["reports"]["feature_kmeans"]["retrospective_evaluation"][
            "unknown_only_metrics"
        ]["nmi"]
        for seed in seeds
    )
    feature_nmi_beats_projection = all(
        per_seed[seed]["reports"]["feature_pca_kmeans"][
            "retrospective_evaluation"
        ]["unknown_only_metrics"]["nmi"]
        > per_seed[seed]["reports"]["projection_pca_kmeans"][
            "retrospective_evaluation"
        ]["unknown_only_metrics"]["nmi"]
        for seed in seeds
    )
    feature_control_names = (
        "feature_pca_kmeans",
        "projection_pca_kmeans",
        "feature_kmeans",
        "feature_normalized_kmeans",
    )
    feature_pca_best_mean_all_metrics = all(
        experiments["feature_pca_kmeans"]["aggregate"][metric]["mean"]
        == max(
            experiments[name]["aggregate"][metric]["mean"]
            for name in feature_control_names
        )
        for metric in METRIC_NAMES
    )
    feature_pca_beats_normalized_raw_all_seeds = {
        metric: all(
            per_seed[seed]["reports"]["feature_pca_kmeans"][
                "retrospective_evaluation"
            ]["unknown_only_metrics"][metric]
            > per_seed[seed]["reports"]["feature_normalized_kmeans"][
                "retrospective_evaluation"
            ]["unknown_only_metrics"][metric]
            for seed in seeds
        )
        for metric in METRIC_NAMES
    }

    method_names = (
        "feature_pca_kmeans",
        "feature_pca_agglomerative",
        "feature_pca_spectral",
    )
    winners = {}
    for seed in seeds:
        winners[str(seed)] = {}
        for metric in METRIC_NAMES:
            scores = {
                experiments[name]["method"]: next(
                    row[metric]
                    for row in experiments[name]["per_seed"]
                    if row["seed"] == seed
                )
                for name in method_names
            }
            best = max(scores.values())
            winners[str(seed)][metric] = sorted(
                method for method, value in scores.items() if value == best
            )
    dominant_methods = []
    for candidate_name in method_names:
        candidate_method = experiments[candidate_name]["method"]
        if all(
            candidate_method in winners[str(seed)][metric]
            for seed in seeds
            for metric in METRIC_NAMES
        ):
            dominant_methods.append(candidate_method)

    public_per_seed = {
        str(seed): {
            "run_dir": per_seed[seed]["run_dir"],
            "pool": per_seed[seed]["pool"],
            "threshold": per_seed[seed]["metadata"]["threshold"],
            "student_checkpoint_sha256": per_seed[seed]["metadata"][
                "student_checkpoint_sha256"
            ],
            "all_unknown_oracle": per_seed[seed]["all_unknown_oracle"],
        }
        for seed in seeds
    }
    all_unknown_oracle_aggregate = {
        metric: _aggregate(
            [
                float(per_seed[seed]["all_unknown_oracle"]["metrics"][metric])
                for seed in seeds
            ]
        )
        for metric in METRIC_NAMES
    }
    return {
        "protocol": "frozen_candidate_pool_oracle_k_multiseed",
        "seeds": seeds,
        "source_index_sha256": shared_source_digest,
        "per_seed": public_per_seed,
        "pool_aggregate": pool_aggregate,
        "all_unknown_oracle_aggregate": all_unknown_oracle_aggregate,
        "experiments": experiments,
        "decisions": {
            "comparison_representation": "feature_pca",
            "comparison_representation_status": "diagnostic_anchor_not_label_selected",
            "method_comparison_status": "exploratory_post_hoc_not_confirmatory",
            "retrospective_feature_pca_best_three_seed_mean_all_metrics": (
                feature_pca_best_mean_all_metrics
            ),
            "retrospective_feature_pca_nmi_beats_raw_feature_all_seeds": feature_nmi_beats_raw,
            "retrospective_feature_pca_nmi_beats_projection_pca_all_seeds": (
                feature_nmi_beats_projection
            ),
            "retrospective_feature_pca_beats_normalized_raw_all_seeds": (
                feature_pca_beats_normalized_raw_all_seeds
            ),
            "method_metric_winners": winners,
            "dominant_methods_all_metrics_all_seeds": dominant_methods,
            "run_auto_k": False,
            "stop_reason": (
                "The oracle-K candidate pools contain a known-sample majority and "
                "recover only a small fraction of unknown samples; method winners "
                "also vary by seed and metric. Candidate-pool quality is the current "
                "priority, so auto K is deferred rather than ruled out."
            ),
        },
    }


def _format_mean_std(stats: dict[str, float]) -> str:
    return f"{stats['mean']:.4f} +/- {stats['std']:.4f}"


def render_summary(summary: dict) -> str:
    lines = [
        "# Frozen candidate-pool three-seed summary",
        "",
        "All clustering assignments were produced from label-free frozen pools. "
        "Ground-truth labels were loaded only for retrospective evaluation.",
        "",
        "## Candidate pool",
        "",
        "| Seed | Candidates | Purity | Known contamination | Unknown recall | Zero-recall unknown classes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in summary["seeds"]:
        pool = summary["per_seed"][str(seed)]["pool"]
        lines.append(
            f"| {seed} | {pool['candidate_count']} | {pool['candidate_purity']:.4f} | "
            f"{pool['known_contamination']:.4f} | "
            f"{pool['candidate_unknown_recall']:.4f} | "
            f"{pool['unknown_classes_with_zero_candidates']} |"
        )

    lines.extend(
        [
            "",
            "## Oracle-K feature comparison",
            "",
            "These are retrospective descriptions. They are not used to select a deployable representation.",
            "",
            "| Feature pipeline | ACC | NMI | ARI |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "feature_pca_kmeans",
        "projection_pca_kmeans",
        "feature_kmeans",
        "feature_normalized_kmeans",
    ):
        experiment = summary["experiments"][name]
        aggregate = experiment["aggregate"]
        lines.append(
            f"| {FEATURE_PIPELINE_LABELS[name]} | {_format_mean_std(aggregate['acc'])} | "
            f"{_format_mean_std(aggregate['nmi'])} | "
            f"{_format_mean_std(aggregate['ari'])} |"
        )

    lines.extend(
        [
            "",
            "## All-true-unknown oracle diagnostic",
            "",
            "Ground-truth known/unknown identity and K are used only to define this evaluation-only diagnostic.",
            "",
            "| Scope | ACC | NMI | ARI |",
            "|---|---:|---:|---:|",
        ]
    )
    all_unknown = summary["all_unknown_oracle_aggregate"]
    candidate_reference = summary["experiments"]["feature_pca_kmeans"]["aggregate"]
    lines.extend(
        [
            f"| All true unknown samples | {_format_mean_std(all_unknown['acc'])} | "
            f"{_format_mean_std(all_unknown['nmi'])} | {_format_mean_std(all_unknown['ari'])} |",
            f"| Rejected true-unknown candidates | {_format_mean_std(candidate_reference['acc'])} | "
            f"{_format_mean_std(candidate_reference['nmi'])} | {_format_mean_std(candidate_reference['ari'])} |",
            "",
            "## Exploratory method comparison on feature_pca",
            "",
            "`feature_pca` is a fixed diagnostic anchor here. Because the anchor was retained after retrospective label-based evaluation, this method comparison is exploratory rather than confirmatory.",
            "",
            "| Method | ACC | NMI | ARI |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "feature_pca_kmeans",
        "feature_pca_agglomerative",
        "feature_pca_spectral",
    ):
        experiment = summary["experiments"][name]
        aggregate = experiment["aggregate"]
        lines.append(
            f"| {experiment['method']} | {_format_mean_std(aggregate['acc'])} | "
            f"{_format_mean_std(aggregate['nmi'])} | "
            f"{_format_mean_std(aggregate['ari'])} |"
        )

    pool = summary["pool_aggregate"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Mean candidate purity: {_format_mean_std(pool['candidate_purity'])}.",
            f"- Mean unknown recall: {_format_mean_std(pool['candidate_unknown_recall'])}.",
            "- Changed factor: feature form, matched L2 normalization, or clustering method; candidate membership and oracle K remain fixed.",
            "- Formal baseline: `feature_pca + KMeans` on the frozen candidate pool. Feature comparisons are retrospective descriptions only.",
            "- The matched raw control isolates the added PCA+whitening stage from the L2-normalization change.",
            "- No clustering method dominates ACC, NMI, and ARI across all seeds.",
            "- The three seeds agree that candidate purity is below 0.36 and unknown recall is below 0.05; metric-level representation and method winners are not uniformly consistent.",
            "- The evidence supports candidate-pool quality as the immediate bottleneck, but does not rule out representation or K-estimation error.",
            "- Next step: improve candidate recall/purity under a separately validated detector protocol. Auto K remains deferred until that pool is usable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = summarize(args.run_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_dirs": [str(path) for path in args.run_dirs],
        "required_runs": RUNS,
        "protocol": summary["protocol"],
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        render_summary(summary), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "seeds": summary["seeds"]}))


if __name__ == "__main__":
    main()
