import json
from pathlib import Path

import pytest

from scripts.summarize_frozen_candidate_pool import RUNS, render_summary, summarize


def _write_seed(root: Path, seed: int, purity: float, recall: float) -> Path:
    run_dir = root / f"seed{seed}"
    run_dir.mkdir()
    metadata = {
        "seed": seed,
        "candidate_count": 4,
        "source_index_sha256": "shared",
        "artifact_sha256": f"artifact-{seed}",
        "labels_exported": False,
        "threshold": 1.5,
        "student_checkpoint_sha256": f"checkpoint-{seed}",
    }
    (run_dir / "candidate_pool_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    settings = {
        "feature_pca_kmeans": ("feature_pca", "kmeans", 0.6),
        "projection_pca_kmeans": ("projection_pca", "kmeans", 0.5),
        "feature_kmeans": ("feature", "kmeans", 0.4),
        "feature_normalized_kmeans": ("feature", "kmeans", 0.45),
        "feature_pca_agglomerative": ("feature_pca", "agglomerative", 0.55),
        "feature_pca_spectral": ("feature_pca", "spectral", 0.58),
    }
    for name, relative_dir in RUNS.items():
        feature, method, nmi = settings[name]
        report_dir = run_dir / relative_dir
        report_dir.mkdir()
        report = {
            "seed": seed,
            "pool_source_index_sha256": "shared",
            "pool_artifact_sha256": f"artifact-{seed}",
            "cluster_k_mode": "oracle",
            "selected_k": 2,
            "requested_oracle_k": 2,
            "test_labels_used_for_clustering_or_k_selection": False,
            "candidate_count": 4,
            "feature": feature,
            "normalized": name != "feature_kmeans",
            "method": method,
            "cluster_sizes": {"0": 2, "1": 2},
            "retrospective_evaluation": {
                "labels_loaded_after_clustering": True,
                "candidate_purity": purity,
                "known_contamination": 1.0 - purity,
                "candidate_unknown_recall": recall,
                "unknown_only_metrics": {"acc": 0.5, "nmi": nmi, "ari": 0.3},
                "unknown_classwise": [
                    {"candidate_recall": 0.0},
                    {"candidate_recall": recall * 2},
                ],
                "known_contamination_classwise": [
                    {"false_reject_rate": 0.1},
                    {"false_reject_rate": 0.2},
                ],
            },
            "all_unknown_oracle_evaluation": {
                "sample_count": 8,
                "oracle_k": 2,
                "metrics": {"acc": 0.75, "nmi": 0.65, "ari": 0.5},
                "used_for_representation_or_method_selection": False,
                "classwise": [],
            },
        }
        (report_dir / "cluster_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
    return run_dir


def test_summarize_paired_oracle_runs(tmp_path):
    run_dirs = [
        _write_seed(tmp_path, 42, 0.4, 0.1),
        _write_seed(tmp_path, 123, 0.6, 0.2),
    ]

    result = summarize(run_dirs)

    assert result["seeds"] == [42, 123]
    assert result["pool_aggregate"]["candidate_purity"]["mean"] == pytest.approx(0.5)
    assert result["pool_aggregate"]["candidate_purity"]["std"] == pytest.approx(
        2**0.5 / 10
    )
    assert result["decisions"]["retrospective_feature_pca_nmi_beats_raw_feature_all_seeds"]
    assert result["decisions"]["retrospective_feature_pca_nmi_beats_projection_pca_all_seeds"]
    assert result["decisions"]["retrospective_feature_pca_best_three_seed_mean_all_metrics"]
    assert result["decisions"]["method_comparison_status"] == "exploratory_post_hoc_not_confirmatory"
    assert result["all_unknown_oracle_aggregate"]["acc"]["mean"] == pytest.approx(0.75)
    assert result["decisions"]["run_auto_k"] is False
    rendered = render_summary(result)
    assert "Auto K remains deferred" in rendered
    assert "feature (raw)" in rendered
    assert "feature (L2-normalized)" in rendered
    assert result["experiments"]["feature_kmeans"]["normalized"] is False
    assert result["experiments"]["feature_normalized_kmeans"]["normalized"] is True


def test_summarize_rejects_auto_k_run(tmp_path):
    run_dirs = [
        _write_seed(tmp_path, 42, 0.4, 0.1),
        _write_seed(tmp_path, 123, 0.6, 0.2),
    ]
    path = run_dirs[0] / RUNS["feature_pca_kmeans"] / "cluster_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["cluster_k_mode"] = "auto"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="only oracle-K"):
        summarize(run_dirs)
