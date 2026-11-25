import hashlib
import json
import sys

import numpy as np
import pytest

from scripts.analyze_frozen_candidate_pool import (
    cluster_unknown_samples,
    evaluate_all_unknown_oracle,
    evaluate_after_clustering,
    load_pool,
    main,
)


def _write_pool(tmp_path, **extra_arrays):
    source_index = np.array([1, 2, 4], dtype=np.int64)
    arrays = {
        "source_index": source_index,
        "classification_feature": np.zeros((3, 4), dtype=np.float32),
        "projection_feature": np.zeros((3, 2), dtype=np.float32),
        "risk_score": np.ones(3, dtype=np.float64),
        "candidate": np.ones(3, dtype=np.uint8),
        "candidate_weight": np.ones(3, dtype=np.float32),
        **extra_arrays,
    }
    pool_path = tmp_path / "candidate_pool.npz"
    np.savez_compressed(pool_path, **arrays)
    exported_source_index = np.asarray(arrays["source_index"], dtype=np.int64)
    metadata = {
        "schema_version": 2,
        "labels_exported": False,
        "source_index_sha256": hashlib.sha256(
            np.ascontiguousarray(exported_source_index).tobytes()
        ).hexdigest(),
        "artifact_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "array_sha256": {
            name: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            for name, values in arrays.items()
        },
    }
    (tmp_path / "candidate_pool_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return pool_path


def test_load_pool_rejects_label_arrays(tmp_path):
    pool_path = _write_pool(tmp_path, raw_labels=np.array([1, 80, 81]))

    with pytest.raises(ValueError, match="forbidden evaluation arrays"):
        load_pool(pool_path)


def test_load_pool_rejects_modified_artifact(tmp_path):
    pool_path = _write_pool(tmp_path)
    with np.load(pool_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["candidate"] = np.zeros(3, dtype=np.uint8)
    np.savez_compressed(pool_path, **arrays)

    with pytest.raises(ValueError, match="artifact digest"):
        load_pool(pool_path)


def test_evaluation_reports_recall_contamination_and_classwise_metrics(tmp_path):
    detail_path = tmp_path / "discovery_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "source_index": [0, 1, 2, 3, 4, 5],
                "raw_labels": [0, 1, 80, 80, 81, 81],
                "true_known": [1, 1, 0, 0, 0, 0],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_after_clustering(
        detail_path,
        source_index=np.array([1, 2, 4], dtype=np.int64),
        predictions=np.array([2, 0, 1], dtype=np.int64),
    )

    assert report["candidate_purity"] == pytest.approx(2 / 3)
    assert report["candidate_unknown_recall"] == pytest.approx(0.5)
    assert report["missed_unknown_count"] == 2
    assert report["true_unknown_k"] == 2
    assert [row["candidate_recall"] for row in report["unknown_classwise"]] == [
        pytest.approx(0.5),
        pytest.approx(0.5),
    ]
    assert [
        row["hungarian_matched_recall"] for row in report["unknown_classwise"]
    ] == [pytest.approx(1.0), pytest.approx(1.0)]
    known_rows = {
        row["raw_label"]: row for row in report["known_contamination_classwise"]
    }
    assert known_rows[0]["candidate_count"] == 0
    assert known_rows[1]["candidate_count"] == 1
    assert known_rows[1]["false_reject_rate"] == pytest.approx(1.0)


def test_evaluation_rejects_duplicate_detail_indices(tmp_path):
    detail_path = tmp_path / "discovery_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "source_index": [1, 1],
                "raw_labels": [80, 81],
                "true_known": [0, 0],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one-dimensional and unique"):
        evaluate_after_clustering(
            detail_path,
            source_index=np.array([1], dtype=np.int64),
            predictions=np.array([0], dtype=np.int64),
        )


def test_all_unknown_oracle_uses_full_unknown_subset_for_diagnostics(tmp_path):
    detail_path = tmp_path / "discovery_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "source_index": [0, 1, 2, 3, 4, 5],
                "raw_labels": [0, 1, 80, 80, 81, 81],
                "true_known": [1, 1, 0, 0, 0, 0],
            }
        ),
        encoding="utf-8",
    )
    features = np.array(
        [[0, 0], [0, 1], [-2, -2], [-2.1, -1.9], [2, 2], [2.1, 1.9]],
        dtype=np.float32,
    )

    report, source_index, predictions = evaluate_all_unknown_oracle(
        detail_path,
        np.arange(6, dtype=np.int64),
        features,
        use_pca=False,
        pca_dim=2,
        whiten=True,
        normalize=False,
        method="kmeans",
        n_init=10,
        seed=42,
    )

    assert report["sample_count"] == 4
    assert report["oracle_k"] == 2
    assert report["metrics"]["acc"] == pytest.approx(1.0)
    assert report["uses_ground_truth_unknown_mask"] is True
    assert report["used_for_representation_or_method_selection"] is False
    assert source_index.tolist() == [2, 3, 4, 5]
    assert predictions.shape == (4,)


def test_cli_writes_reproducible_run_artifacts(tmp_path, monkeypatch):
    source_index = np.arange(6, dtype=np.int64)
    features = np.array(
        [
            [-2.0, -2.0],
            [-2.1, -1.9],
            [-1.9, -2.1],
            [2.0, 2.0],
            [2.1, 1.9],
            [1.9, 2.1],
        ],
        dtype=np.float32,
    )
    pool_path = _write_pool(
        tmp_path,
        source_index=source_index,
        classification_feature=features,
        projection_feature=features,
        risk_score=np.ones(6, dtype=np.float64),
        candidate=np.ones(6, dtype=np.uint8),
        candidate_weight=np.ones(6, dtype=np.float32),
    )
    detail_path = tmp_path / "discovery_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "source_index": source_index.tolist(),
                "raw_labels": [80, 80, 80, 81, 81, 81],
                "true_known": [0, 0, 0, 0, 0, 0],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_frozen_candidate_pool.py",
            str(pool_path),
            "--output-dir",
            str(output_dir),
            "--feature",
            "feature",
            "--method",
            "kmeans",
            "--cluster-k",
            "oracle",
            "--num-novel",
            "2",
            "--evaluation-detail",
            str(detail_path),
        ],
    )

    main()

    expected_files = {
        "cluster_assignments.npz",
        "all_unknown_oracle_assignments.npz",
        "cluster_report.json",
        "metrics.json",
        "config.json",
        "command.txt",
        "summary.md",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    report = json.loads((output_dir / "cluster_report.json").read_text(encoding="utf-8"))
    assert report["selected_k"] == 2
    assert report["k_absolute_error"] == 0
    assert report["retrospective_evaluation"]["unknown_only_metrics"]["acc"] == pytest.approx(1.0)
    assert report["all_unknown_oracle_evaluation"]["metrics"]["acc"] == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["kmeans", "agglomerative", "spectral"])
def test_supported_clustering_methods_run_on_same_frozen_features(method):
    features = np.array(
        [
            [-2.0, -2.0],
            [-2.1, -1.9],
            [-1.9, -2.1],
            [2.0, 2.0],
            [2.1, 1.9],
            [1.9, 2.1],
        ],
        dtype=np.float32,
    )

    predictions = cluster_unknown_samples(features, 2, method=method, random_state=42)

    assert predictions.shape == (6,)
    assert len(np.unique(predictions)) == 2
