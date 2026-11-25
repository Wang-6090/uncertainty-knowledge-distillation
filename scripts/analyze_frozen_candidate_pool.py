"""Run clustering ablations on one frozen, label-free candidate-pool artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.metrics import clustering_report


REQUIRED_ARRAYS = {
    "source_index",
    "classification_feature",
    "projection_feature",
    "risk_score",
    "candidate",
    "candidate_weight",
}
FORBIDDEN_LABEL_ARRAYS = {
    "label",
    "labels",
    "raw_label",
    "raw_labels",
    "is_known",
    "true_known",
    "true_label",
}


def prepare_cluster_features(
    features: np.ndarray,
    use_pca: bool = False,
    pca_dim: int = 32,
    whiten: bool = True,
) -> np.ndarray:
    """Prepare a frozen feature matrix without consulting evaluation labels."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        return values
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    if use_pca and len(values) > 2 and values.shape[1] > 1:
        n_components = min(int(pca_dim), values.shape[1], len(values) - 1)
        if n_components >= 2:
            values = PCA(
                n_components=n_components,
                whiten=whiten,
                random_state=42,
            ).fit_transform(values)
            values = values / np.maximum(
                np.linalg.norm(values, axis=1, keepdims=True), 1e-12
            )
    return values


def cluster_unknown_samples(
    features: np.ndarray,
    num_clusters: int,
    method: str = "kmeans",
    n_init: int = 10,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Cluster frozen features with a selectable label-free method."""
    values = np.asarray(features, dtype=np.float32)
    num_clusters = int(max(1, min(num_clusters, len(values))))
    if num_clusters == 1 or len(values) <= 1:
        return np.zeros(len(values), dtype=np.int64)
    if method == "kmeans":
        model = KMeans(
            n_clusters=num_clusters,
            n_init=max(1, int(n_init)),
            random_state=random_state,
        )
        model.fit(values, sample_weight=sample_weight)
        return model.labels_
    if sample_weight is not None and not np.allclose(sample_weight, 1.0):
        raise ValueError("sample_weight is currently supported only for kmeans")
    if method == "agglomerative":
        return AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage="ward",
        ).fit_predict(values)
    if method == "spectral":
        neighbors = max(1, min(10, len(values) - 1))
        return SpectralClustering(
            n_clusters=num_clusters,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            assign_labels="kmeans",
            n_init=max(1, int(n_init)),
            random_state=random_state,
        ).fit_predict(values)
    raise ValueError(f"unsupported cluster method: {method}")


def _scale_metric(values: list[float], higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not higher_is_better:
        array = -array
    if np.ptp(array) < 1e-12:
        return np.ones_like(array)
    return (array - array.min()) / (array.max() - array.min())


def evaluate_cluster_candidates(
    features: np.ndarray,
    max_clusters: int,
    method: str = "kmeans",
    selection: str = "silhouette",
    n_init: int = 10,
    stability_repeats: int = 5,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
) -> tuple[int, list[dict]]:
    """Select K from frozen-pool geometry without using unknown labels."""
    n = len(features)
    if n < 4:
        return max(1, n), []
    upper = min(int(max_clusters), n - 1)
    if upper < 2:
        return 1, []
    values = np.asarray(features, dtype=np.float32)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
    if weights is not None and len(weights) != n:
        raise ValueError("sample_weight must align with clustering features")
    eval_values = values
    eval_indices = np.arange(n)
    if n > 2000:
        rng = np.random.default_rng(random_state)
        eval_indices = rng.choice(n, 2000, replace=False)
        eval_values = values[eval_indices]

    rows: list[dict] = []
    for k in range(2, upper + 1):
        labels = cluster_unknown_samples(
            values,
            k,
            method,
            n_init,
            random_state,
            sample_weight=weights,
        )
        eval_labels = labels[eval_indices]
        if len(np.unique(eval_labels)) < 2:
            continue
        row = {
            "k": int(k),
            "silhouette": float(silhouette_score(eval_values, eval_labels)),
            "calinski_harabasz": float(
                calinski_harabasz_score(eval_values, eval_labels)
            ),
            "davies_bouldin": float(davies_bouldin_score(eval_values, eval_labels)),
        }
        if selection == "gmm_bic":
            mixture = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-4,
                n_init=1,
                random_state=random_state,
            ).fit(eval_values)
            row["gmm_bic"] = float(mixture.bic(eval_values))
        if selection == "stability":
            repeat_labels = [
                cluster_unknown_samples(
                    values,
                    k,
                    method,
                    n_init,
                    random_state + repeat + 1,
                    sample_weight=weights,
                )[eval_indices]
                for repeat in range(max(2, int(stability_repeats)))
            ]
            stability_values = [
                normalized_mutual_info_score(repeat_labels[i], repeat_labels[j])
                for i in range(len(repeat_labels))
                for j in range(i + 1, len(repeat_labels))
            ]
            row["stability_nmi"] = (
                float(np.mean(stability_values)) if stability_values else 1.0
            )
        if selection == "bootstrap_stability":
            if method != "kmeans":
                raise ValueError("bootstrap_stability requires method='kmeans'")
            bootstrap_rng = np.random.default_rng(random_state + 10000 + int(k))
            sample_size = max(int(k) * 3, int(np.ceil(0.8 * len(eval_values))))
            sample_size = min(len(eval_values), sample_size)
            bootstrap_labels = []
            for repeat in range(max(3, int(stability_repeats))):
                subset = bootstrap_rng.choice(
                    len(eval_values), size=sample_size, replace=False
                )
                model = KMeans(
                    n_clusters=int(k),
                    n_init=max(1, int(n_init)),
                    random_state=random_state + 20000 + repeat,
                ).fit(
                    eval_values[subset],
                    sample_weight=(
                        None
                        if weights is None
                        else weights[eval_indices][subset]
                    ),
                )
                bootstrap_labels.append(model.predict(eval_values))
            stability_values = [
                normalized_mutual_info_score(
                    bootstrap_labels[i], bootstrap_labels[j]
                )
                for i in range(len(bootstrap_labels))
                for j in range(i + 1, len(bootstrap_labels))
            ]
            row["bootstrap_stability_nmi"] = (
                float(np.mean(stability_values)) if stability_values else 1.0
            )
        rows.append(row)

    if not rows:
        return 1, []
    silhouette = _scale_metric([row["silhouette"] for row in rows], True)
    ch = _scale_metric([row["calinski_harabasz"] for row in rows], True)
    db = _scale_metric([row["davies_bouldin"] for row in rows], False)
    internal = 0.5 * silhouette + 0.25 * ch + 0.25 * db
    if selection == "silhouette":
        combined = silhouette
    elif selection == "stability":
        stability = np.asarray([row["stability_nmi"] for row in rows], dtype=float)
        combined = 0.5 * internal + 0.5 * stability
    elif selection == "bootstrap_stability":
        stability = np.asarray(
            [row["bootstrap_stability_nmi"] for row in rows], dtype=float
        )
        combined = 0.5 * internal + 0.5 * stability
    elif selection == "composite":
        combined = internal
    elif selection == "gmm_bic":
        combined = _scale_metric(
            [row["gmm_bic"] for row in rows], higher_is_better=False
        )
    else:
        raise ValueError(f"unsupported cluster selection: {selection}")
    for row, internal_value, value in zip(rows, internal, combined):
        row["internal_score"] = float(internal_value)
        row["selection_score"] = float(value)
    return int(rows[int(np.argmax(combined))]["k"]), rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster a previously exported candidate_pool.npz without rerunning "
            "the model or changing candidate membership."
        )
    )
    parser.add_argument("pool", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature",
        choices=["feature", "feature_pca", "projection", "projection_pca"],
        default="feature_pca",
    )
    parser.add_argument(
        "--method", choices=["kmeans", "agglomerative", "spectral"], default="kmeans"
    )
    parser.add_argument("--cluster-k", choices=["oracle", "auto"], default="oracle")
    parser.add_argument(
        "--num-novel",
        type=int,
        default=40,
        help="Protocol-defined K for --cluster-k oracle; labels are not read to derive it.",
    )
    parser.add_argument("--max-auto-clusters", type=int, default=50)
    parser.add_argument(
        "--selection",
        choices=[
            "silhouette",
            "composite",
            "stability",
            "bootstrap_stability",
            "gmm_bic",
        ],
        default="silhouette",
    )
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--no-whiten", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--stability-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-all-unknown-oracle",
        action="store_true",
        help="Skip the evaluation-only all-true-unknown oracle diagnostic.",
    )
    parser.add_argument(
        "--evaluation-detail",
        type=Path,
        help=(
            "Optional discovery_detail.json. Ground-truth fields are loaded only "
            "after assignments have been produced and are used only for reporting."
        ),
    )
    return parser.parse_args()


def _index_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pool(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(REQUIRED_ARRAYS - set(payload.files))
        if missing:
            raise ValueError(f"candidate pool is missing arrays: {missing}")
        forbidden = sorted(FORBIDDEN_LABEL_ARRAYS & set(payload.files))
        if forbidden:
            raise ValueError(
                f"candidate pool contains forbidden evaluation arrays: {forbidden}"
            )
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    sample_count = len(arrays["source_index"])
    if sample_count == 0:
        raise ValueError("candidate pool is empty")
    if arrays["source_index"].ndim != 1:
        raise ValueError("source_index must be one-dimensional")
    for name, values in arrays.items():
        if len(values) != sample_count:
            raise ValueError(f"array {name!r} is not row-aligned")
    if len(np.unique(arrays["source_index"])) != sample_count:
        raise ValueError("source_index must uniquely identify every exported sample")

    metadata_path = path.with_name("candidate_pool_metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing candidate-pool metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("labels_exported") is not False:
        raise ValueError("candidate-pool metadata does not guarantee label isolation")
    digest = _index_digest(arrays["source_index"])
    if metadata.get("source_index_sha256") != digest:
        raise ValueError("candidate-pool source-index digest does not match metadata")
    if int(metadata.get("schema_version", 1)) >= 2:
        if metadata.get("artifact_sha256") != _file_digest(path):
            raise ValueError("candidate-pool artifact digest does not match metadata")
        expected_array_digests = metadata.get("array_sha256")
        if not isinstance(expected_array_digests, dict):
            raise ValueError("candidate-pool schema v2 metadata is missing array digests")
        if set(expected_array_digests) != set(arrays):
            raise ValueError("candidate-pool array digest inventory does not match artifact")
        for name, values in arrays.items():
            if expected_array_digests.get(name) != _array_digest(values):
                raise ValueError(f"candidate-pool array digest mismatch: {name}")
    return arrays, metadata


def evaluate_after_clustering(
    detail_path: Path,
    source_index: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    source_index = np.asarray(source_index, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if source_index.ndim != 1 or predictions.ndim != 1:
        raise ValueError("candidate source indices and predictions must be one-dimensional")
    if len(source_index) != len(predictions):
        raise ValueError("candidate source indices and predictions must be row-aligned")
    if len(np.unique(source_index)) != len(source_index):
        raise ValueError("candidate source indices must be unique")
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    required = {"source_index", "raw_labels", "true_known"}
    missing = sorted(required - detail.keys())
    if missing:
        raise ValueError(f"evaluation detail is missing fields: {missing}")
    detail_index = np.asarray(detail["source_index"], dtype=np.int64)
    raw_labels = np.asarray(detail["raw_labels"], dtype=np.int64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    if detail_index.ndim != 1 or raw_labels.ndim != 1 or true_known.ndim != 1:
        raise ValueError("evaluation detail arrays must be one-dimensional")
    if not (len(detail_index) == len(raw_labels) == len(true_known)):
        raise ValueError("evaluation detail arrays are not row-aligned")
    if len(np.unique(detail_index)) != len(detail_index):
        raise ValueError("evaluation detail source_index must be one-dimensional and unique")
    lookup = {int(value): position for position, value in enumerate(detail_index)}
    try:
        positions = np.asarray([lookup[int(value)] for value in source_index], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"candidate source index missing from evaluation detail: {exc}") from exc
    candidate_labels = raw_labels[positions]
    candidate_known = true_known[positions]
    unknown_mask = ~candidate_known
    all_unknown = ~true_known
    candidate_index_set = set(int(value) for value in source_index)
    selected_in_detail = np.asarray(
        [int(value) in candidate_index_set for value in detail_index], dtype=bool
    )
    true_unknown_k = int(np.unique(raw_labels[all_unknown]).size)

    result = {
        "evaluation_only": True,
        "labels_loaded_after_clustering": True,
        "candidate_purity": float(unknown_mask.mean()),
        "known_contamination": float(candidate_known.mean()),
        "true_unknown_count": int(unknown_mask.sum()),
        "false_reject_count": int(candidate_known.sum()),
        "total_unknown_count": int(all_unknown.sum()),
        "missed_unknown_count": int((all_unknown & ~selected_in_detail).sum()),
        "candidate_unknown_recall": float(
            (all_unknown & selected_in_detail).sum() / max(int(all_unknown.sum()), 1)
        ),
        "true_unknown_k": true_unknown_k,
    }
    if len(predictions) > 1:
        result["all_candidate_metrics"] = clustering_report(candidate_labels, predictions)
    if int(unknown_mask.sum()) > 1:
        result["unknown_only_metrics"] = clustering_report(
            candidate_labels[unknown_mask], predictions[unknown_mask]
        )

    predicted_to_raw_label: dict[int, int] = {}
    if int(unknown_mask.sum()) > 0:
        unknown_labels = candidate_labels[unknown_mask]
        unknown_predictions = predictions[unknown_mask]
        label_values = np.unique(unknown_labels)
        cluster_values = np.unique(unknown_predictions)
        contingency = np.zeros((len(label_values), len(cluster_values)), dtype=np.int64)
        for label_position, raw_label in enumerate(label_values):
            for cluster_position, cluster_id in enumerate(cluster_values):
                contingency[label_position, cluster_position] = int(
                    np.sum(
                        (unknown_labels == raw_label)
                        & (unknown_predictions == cluster_id)
                    )
                )
        row_indices, column_indices = linear_sum_assignment(
            contingency.max(initial=0) - contingency
        )
        predicted_to_raw_label = {
            int(cluster_values[column]): int(label_values[row])
            for row, column in zip(row_indices, column_indices)
        }

    classwise = []
    for raw_label in sorted(np.unique(raw_labels[all_unknown]).tolist()):
        class_mask = unknown_mask & (candidate_labels == raw_label)
        cluster_ids, counts = np.unique(predictions[class_mask], return_counts=True)
        dominant_count = int(counts.max()) if len(counts) else 0
        total_class_count = int(np.sum(all_unknown & (raw_labels == raw_label)))
        candidate_count = int(class_mask.sum())
        matched_count = int(
            sum(
                count
                for cluster_id, count in zip(cluster_ids, counts)
                if predicted_to_raw_label.get(int(cluster_id)) == int(raw_label)
            )
        )
        classwise.append(
            {
                "raw_label": int(raw_label),
                "total_count": total_class_count,
                "candidate_count": candidate_count,
                "candidate_recall": float(candidate_count / max(total_class_count, 1)),
                "occupied_clusters": int(len(cluster_ids)),
                "dominant_cluster_fraction": float(
                    dominant_count / max(candidate_count, 1)
                ),
                "hungarian_matched_count": matched_count,
                "hungarian_matched_recall": float(
                    matched_count / max(candidate_count, 1)
                ),
            }
        )
    result["unknown_classwise"] = classwise
    known_classwise = []
    for raw_label in sorted(np.unique(raw_labels[true_known]).tolist()):
        full_class_mask = true_known & (raw_labels == raw_label)
        candidate_count = int((full_class_mask & selected_in_detail).sum())
        total_class_count = int(full_class_mask.sum())
        known_classwise.append(
            {
                "raw_label": int(raw_label),
                "total_count": total_class_count,
                "candidate_count": candidate_count,
                "false_reject_rate": float(
                    candidate_count / max(total_class_count, 1)
                ),
            }
        )
    result["known_contamination_classwise"] = known_classwise
    return result


def evaluate_all_unknown_oracle(
    detail_path: Path,
    source_index: np.ndarray,
    features: np.ndarray,
    *,
    use_pca: bool,
    pca_dim: int,
    whiten: bool,
    normalize: bool,
    method: str,
    n_init: int,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run an evaluation-only oracle diagnostic on every true unknown sample."""
    source_index = np.asarray(source_index, dtype=np.int64)
    features = np.asarray(features, dtype=np.float32)
    if len(source_index) != len(features):
        raise ValueError("full-pool source indices and features must be row-aligned")

    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    required = {"source_index", "raw_labels", "true_known"}
    missing = sorted(required - detail.keys())
    if missing:
        raise ValueError(f"evaluation detail is missing fields: {missing}")
    detail_index = np.asarray(detail["source_index"], dtype=np.int64)
    raw_labels = np.asarray(detail["raw_labels"], dtype=np.int64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    if not (len(detail_index) == len(raw_labels) == len(true_known)):
        raise ValueError("evaluation detail arrays are not row-aligned")
    if len(np.unique(detail_index)) != len(detail_index):
        raise ValueError("evaluation detail source_index must be one-dimensional and unique")
    lookup = {int(value): position for position, value in enumerate(detail_index)}
    try:
        positions = np.asarray([lookup[int(value)] for value in source_index], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"pool source index missing from evaluation detail: {exc}") from exc

    aligned_labels = raw_labels[positions]
    aligned_known = true_known[positions]
    unknown_mask = ~aligned_known
    unknown_labels = aligned_labels[unknown_mask]
    unknown_source_index = source_index[unknown_mask]
    unknown_features = features[unknown_mask]
    oracle_k = int(np.unique(unknown_labels).size)
    if len(unknown_features) < 2 or oracle_k < 1:
        raise ValueError("all-unknown oracle diagnostic requires at least two unknown samples")
    if normalize or use_pca:
        unknown_features = prepare_cluster_features(
            unknown_features,
            use_pca=use_pca,
            pca_dim=pca_dim,
            whiten=whiten,
        )
    predictions = cluster_unknown_samples(
        unknown_features,
        num_clusters=min(oracle_k, len(unknown_features)),
        method=method,
        n_init=n_init,
        random_state=seed,
    )

    label_values = np.unique(unknown_labels)
    cluster_values = np.unique(predictions)
    contingency = np.zeros((len(label_values), len(cluster_values)), dtype=np.int64)
    for label_position, raw_label in enumerate(label_values):
        for cluster_position, cluster_id in enumerate(cluster_values):
            contingency[label_position, cluster_position] = int(
                np.sum((unknown_labels == raw_label) & (predictions == cluster_id))
            )
    row_indices, column_indices = linear_sum_assignment(
        contingency.max(initial=0) - contingency
    )
    predicted_to_raw_label = {
        int(cluster_values[column]): int(label_values[row])
        for row, column in zip(row_indices, column_indices)
    }
    classwise = []
    for raw_label in sorted(label_values.tolist()):
        class_mask = unknown_labels == raw_label
        class_clusters, counts = np.unique(predictions[class_mask], return_counts=True)
        matched_count = int(
            sum(
                count
                for cluster_id, count in zip(class_clusters, counts)
                if predicted_to_raw_label.get(int(cluster_id)) == int(raw_label)
            )
        )
        class_count = int(class_mask.sum())
        classwise.append(
            {
                "raw_label": int(raw_label),
                "sample_count": class_count,
                "occupied_clusters": int(len(class_clusters)),
                "dominant_cluster_fraction": float(
                    counts.max(initial=0) / max(class_count, 1)
                ),
                "hungarian_matched_count": matched_count,
                "hungarian_matched_accuracy": float(
                    matched_count / max(class_count, 1)
                ),
            }
        )
    cluster_ids, cluster_counts = np.unique(predictions, return_counts=True)
    report = {
        "evaluation_only": True,
        "uses_ground_truth_unknown_mask": True,
        "uses_ground_truth_k": True,
        "used_for_candidate_selection": False,
        "used_for_representation_or_method_selection": False,
        "sample_count": int(len(unknown_labels)),
        "oracle_k": oracle_k,
        "metrics": clustering_report(unknown_labels, predictions),
        "cluster_sizes": {
            str(int(cluster_id)): int(count)
            for cluster_id, count in zip(cluster_ids, cluster_counts)
        },
        "classwise": classwise,
    }
    return report, unknown_source_index, np.asarray(predictions, dtype=np.int64)


def _write_run_manifest(output_dir: Path, args: argparse.Namespace, report: dict) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    evaluation = report.get("retrospective_evaluation", {})
    metrics = evaluation.get("unknown_only_metrics", {})
    summary_lines = [
        "# Frozen candidate-pool clustering run",
        "",
        f"- Candidate pool: `{report['pool']}`",
        f"- Changed factor: feature={report['feature']}, method={report['method']}, K mode={report['cluster_k_mode']}",
        f"- Selected K: {report['selected_k']}",
        f"- Candidate count: {report['candidate_count']}",
    ]
    if evaluation:
        summary_lines.extend(
            [
                f"- True unknown K (evaluation only): {evaluation['true_unknown_k']}",
                f"- K absolute error: {report['k_absolute_error']}",
                f"- Candidate purity: {evaluation['candidate_purity']:.6f}",
                f"- Candidate unknown recall: {evaluation['candidate_unknown_recall']:.6f}",
            ]
        )
    if metrics:
        summary_lines.extend(
            [
                f"- Unknown-only ACC: {metrics['acc']:.6f}",
                f"- Unknown-only NMI: {metrics['nmi']:.6f}",
                f"- Unknown-only ARI: {metrics['ari']:.6f}",
            ]
        )
    all_unknown = report.get("all_unknown_oracle_evaluation", {})
    if all_unknown:
        all_unknown_metrics = all_unknown["metrics"]
        summary_lines.extend(
            [
                "",
                "Evaluation-only all-true-unknown oracle diagnostic:",
                f"- Samples: {all_unknown['sample_count']}",
                f"- Oracle K: {all_unknown['oracle_k']}",
                f"- ACC: {all_unknown_metrics['acc']:.6f}",
                f"- NMI: {all_unknown_metrics['nmi']:.6f}",
                f"- ARI: {all_unknown_metrics['ari']:.6f}",
                "- Ground truth defines this diagnostic subset and K; it does not alter the frozen candidate run.",
            ]
        )
    summary_lines.extend(
        [
            "",
            "Conclusion: compare this run with the paired single-factor controls before attributing the bottleneck.",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    arrays, metadata = load_pool(args.pool)
    candidate_mask = np.asarray(arrays["candidate"], dtype=bool)
    if int(candidate_mask.sum()) < 2:
        raise ValueError("at least two frozen candidates are required for clustering")

    feature_key = (
        "classification_feature" if args.feature.startswith("feature") else "projection_feature"
    )
    candidate_features = np.asarray(arrays[feature_key][candidate_mask], dtype=np.float32)
    use_pca = args.feature.endswith("_pca")
    if args.normalize or use_pca:
        candidate_features = prepare_cluster_features(
            candidate_features,
            use_pca=use_pca,
            pca_dim=args.pca_dim,
            whiten=not args.no_whiten,
        )
    candidate_weights = np.asarray(
        arrays["candidate_weight"][candidate_mask], dtype=np.float32
    )

    diagnostics = []
    selected_k = int(args.num_novel)
    if args.cluster_k == "auto":
        selected_k, diagnostics = evaluate_cluster_candidates(
            candidate_features,
            max_clusters=args.max_auto_clusters,
            method=args.method,
            selection=args.selection,
            n_init=args.n_init,
            stability_repeats=args.stability_repeats,
            random_state=args.seed,
            sample_weight=candidate_weights,
        )
    if selected_k < 1 or selected_k > int(candidate_mask.sum()):
        raise ValueError(
            f"selected K={selected_k} must be between 1 and the frozen candidate "
            f"count ({int(candidate_mask.sum())})"
        )
    predictions = cluster_unknown_samples(
        candidate_features,
        num_clusters=selected_k,
        method=args.method,
        n_init=args.n_init,
        random_state=args.seed,
        sample_weight=candidate_weights,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_source_index = np.asarray(arrays["source_index"][candidate_mask], dtype=np.int64)
    assignment_path = args.output_dir / "cluster_assignments.npz"
    np.savez_compressed(
        assignment_path,
        source_index=candidate_source_index,
        pred_cluster=np.asarray(predictions, dtype=np.int64),
    )
    cluster_ids, cluster_counts = np.unique(predictions, return_counts=True)
    report = {
        "protocol": "frozen_label_free_candidate_pool",
        "pool": str(args.pool.resolve()),
        "pool_source_index_sha256": metadata["source_index_sha256"],
        "pool_artifact_sha256": metadata.get("artifact_sha256"),
        "candidate_count": int(candidate_mask.sum()),
        "feature": args.feature,
        "method": args.method,
        "cluster_k_mode": args.cluster_k,
        "selected_k": int(selected_k),
        "requested_oracle_k": int(args.num_novel) if args.cluster_k == "oracle" else None,
        "selection": args.selection if args.cluster_k == "auto" else "oracle",
        "pca_dim": int(args.pca_dim),
        "whiten": bool(not args.no_whiten),
        "normalized": bool(args.normalize or use_pca),
        "n_init": int(args.n_init),
        "seed": int(args.seed),
        "cluster_sizes": {
            str(int(cluster_id)): int(count)
            for cluster_id, count in zip(cluster_ids, cluster_counts)
        },
        "auto_k_diagnostics": diagnostics,
        "assignments": assignment_path.name,
        "test_labels_used_for_clustering_or_k_selection": False,
    }
    if args.evaluation_detail is not None:
        evaluation = evaluate_after_clustering(
            args.evaluation_detail,
            candidate_source_index,
            predictions,
        )
        report["retrospective_evaluation"] = evaluation
        report["true_unknown_k"] = int(evaluation["true_unknown_k"])
        report["k_absolute_error"] = abs(
            int(selected_k) - int(evaluation["true_unknown_k"])
        )
        if args.cluster_k == "oracle" and not args.skip_all_unknown_oracle:
            all_unknown, unknown_source_index, unknown_predictions = (
                evaluate_all_unknown_oracle(
                    args.evaluation_detail,
                    np.asarray(arrays["source_index"], dtype=np.int64),
                    np.asarray(arrays[feature_key], dtype=np.float32),
                    use_pca=use_pca,
                    pca_dim=args.pca_dim,
                    whiten=not args.no_whiten,
                    normalize=args.normalize,
                    method=args.method,
                    n_init=args.n_init,
                    seed=args.seed,
                )
            )
            all_unknown_assignment_path = (
                args.output_dir / "all_unknown_oracle_assignments.npz"
            )
            np.savez_compressed(
                all_unknown_assignment_path,
                source_index=unknown_source_index,
                pred_cluster=unknown_predictions,
            )
            all_unknown["assignments"] = all_unknown_assignment_path.name
            report["all_unknown_oracle_evaluation"] = all_unknown
    report_path = args.output_dir / "cluster_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_run_manifest(args.output_dir, args, report)
    print(json.dumps({"report": str(report_path), "selected_k": selected_k}, indent=2))


if __name__ == "__main__":
    main()
