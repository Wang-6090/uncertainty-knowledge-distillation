"""Verify that a saved feature-rejector artifact reproduces its scores.

This audit is intentionally read-only.  It loads the fitted rejector, the
frozen model-output arrays, and (when required) the saved class-support
boundary.  It never refits a rejector and never uses labels to change scores.

Usage:
    python scripts/audit_feature_rejector_artifact.py path/to/run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# When invoked as ``python scripts/audit_feature_rejector_artifact.py`` from
# the repository root, Python places ``scripts`` rather than the repository
# root on sys.path.  Add the project root so the read-only audit command works
# exactly as documented without requiring PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel_discovery.pipeline import (
    attach_feature_rejector_score,
    attach_known_support_score,
)


SUPPORT_MODES = {
    "support_augmented",
    "predicted_support_augmented",
    "support_uncertainty_augmented",
    "rejection_support_uncertainty_augmented",
    "rejection_support_augmented",
    "rejection_balanced_support_augmented",
    "rejection_support_logit_augmented",
    "prototype_support_augmented",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_support_model(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"classes", "centers", "radii", "quantile"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"support model is missing arrays: {', '.join(sorted(missing))}"
            )
        return {
            "classes": np.asarray(archive["classes"], dtype=np.int64),
            "centers": np.asarray(archive["centers"], dtype=np.float32),
            "radii": np.asarray(archive["radii"], dtype=np.float32),
            "quantile": float(np.asarray(archive["quantile"]).reshape(())),
        }


def _load_outputs(archive, prefix: str) -> dict:
    marker = f"{prefix}_"
    outputs = {}
    for key in archive.files:
        if key.startswith(marker):
            outputs[key[len(marker) :]] = np.asarray(archive[key])
    if "features" not in outputs:
        raise ValueError(f"recompute input archive has no {prefix}_features array")
    return outputs


def audit_run(run_dir: Path) -> dict:
    metadata_path = run_dir / "feature_rejector.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing artifact metadata: {metadata_path}")
    metadata = _read_json(metadata_path)
    model_path = run_dir / str(metadata["model_file"])
    input_name = metadata.get("recompute_input_file")
    if not input_name:
        raise ValueError(
            "artifact has no frozen recompute inputs; this is a legacy artifact "
            "and cannot be strictly audited"
        )
    score_name = metadata.get("score_file")
    if not score_name:
        raise ValueError("artifact metadata has no score_file; it is not recompute-ready")
    score_path = run_dir / str(score_name)
    input_path = run_dir / str(input_name)
    if not model_path.exists() or not score_path.exists() or not input_path.exists():
        raise FileNotFoundError("artifact is incomplete: model, scores, or inputs missing")
    if metadata.get("model_sha256") and _sha256(model_path) != metadata["model_sha256"]:
        raise ValueError("feature_rejector.pkl hash does not match metadata")
    if metadata.get("score_sha256") and _sha256(score_path) != metadata["score_sha256"]:
        raise ValueError("feature_rejector_scores.npz hash does not match metadata")
    if metadata.get("recompute_input_sha256") and _sha256(input_path) != metadata[
        "recompute_input_sha256"
    ]:
        raise ValueError("feature_rejector_inputs.npz hash does not match metadata")

    support_model = None
    support_name = metadata.get("support_model_file")
    if support_name:
        support_path = run_dir / str(support_name)
        if not support_path.exists():
            raise FileNotFoundError(f"missing support model: {support_path}")
        expected_hash = metadata.get("support_model_sha256")
        if expected_hash and _sha256(support_path) != expected_hash:
            raise ValueError("support model hash does not match metadata")
        support_model = _load_support_model(support_path)
    feature_mode = str(metadata.get("feature_mode", "embedding"))
    if feature_mode in SUPPORT_MODES and support_model is None:
        raise ValueError(
            f"feature mode {feature_mode!r} requires a saved support model"
        )

    with model_path.open("rb") as handle:
        rejector = pickle.load(handle)
    with np.load(score_path, allow_pickle=False) as scores_archive, np.load(
        input_path, allow_pickle=False
    ) as inputs_archive:
        expected_score_arrays = metadata.get("score_arrays")
        if expected_score_arrays is not None and sorted(scores_archive.files) != sorted(
            expected_score_arrays
        ):
            raise ValueError("score archive arrays do not match metadata")
        expected_input_arrays = metadata.get("recompute_input_arrays")
        if expected_input_arrays is not None and sorted(inputs_archive.files) != sorted(
            expected_input_arrays
        ):
            raise ValueError("recompute input archive arrays do not match metadata")
        score_keys = [
            key
            for key in scores_archive.files
            if key.endswith("_feature_rejector_score")
        ]
        if not score_keys:
            raise ValueError("score archive has no feature_rejector_score arrays")
        results = []
        for score_key in sorted(score_keys):
            prefix = score_key[: -len("_feature_rejector_score")]
            outputs = _load_outputs(inputs_archive, prefix)
            if support_model is not None:
                attach_known_support_score(outputs, support_model)
            attach_feature_rejector_score(
                outputs, rejector, feature_mode=feature_mode
            )
            expected = np.asarray(scores_archive[score_key], dtype=np.float64).reshape(-1)
            actual = np.asarray(outputs["feature_rejector_score"], dtype=np.float64).reshape(-1)
            if expected.shape != actual.shape:
                raise ValueError(
                    f"{prefix}: score shape changed from {expected.shape} to {actual.shape}"
                )
            delta = np.abs(expected - actual)
            result = {
                "split": prefix,
                "samples": int(len(expected)),
                "max_abs_error": float(delta.max(initial=0.0)),
                "mean_abs_error": float(delta.mean()) if len(delta) else 0.0,
                "allclose": bool(np.allclose(expected, actual, rtol=1e-6, atol=1e-8)),
            }
            results.append(result)
    return {
        "run_dir": str(run_dir),
        "feature_mode": feature_mode,
        "support_model_loaded": support_model is not None,
        "results": results,
        "passed": all(item["allclose"] for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = audit_run(args.run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
