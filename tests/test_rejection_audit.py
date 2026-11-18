from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from novel_discovery.models import build_model
from novel_discovery.pipeline import attach_feature_rejector_score, fit_feature_rejector
from novel_discovery.rejection_audit import (
    batchnorm_snapshot,
    build_rejection_only_optimizer,
    compare_state_dicts_exact,
    compare_parameter_snapshots,
    parameter_snapshot,
    select_state_dict_prefixes,
    state_dict_structure,
    state_dict_sha256,
    sync_shared_backbone_rejection_encoder,
)
from scripts.audit_feature_rejector_artifact import audit_run
from train import load_checkpoint, save_checkpoint, save_feature_rejector_artifact


class RejectionAuditTest(unittest.TestCase):
    def make_model(self):
        return build_model(
            3,
            backbone="resnet18",
            rejection_feature_dim=8,
            independent_rejection_encoder=True,
            independent_rejection_encoder_init="shared_backbone",
            cifar_stem=True,
        )

    def test_shared_copy_is_equal_but_not_aliased(self):
        model = self.make_model()
        report = sync_shared_backbone_rejection_encoder(model)
        self.assertTrue(report["equal_after_copy"])
        self.assertFalse(report["parameter_storage_aliased"])

    def test_state_dict_comparison_distinguishes_exact_and_changed_values(self):
        model = self.make_model()
        classification = select_state_dict_prefixes(
            model.state_dict(), ("encoder", "classifier", "projector", "uncertainty_head")
        )
        same = {name: value.clone() for name, value in classification.items()}
        self.assertTrue(compare_state_dicts_exact(classification, same)["exact"])
        self.assertEqual(state_dict_sha256(classification), state_dict_sha256(same))
        changed = {name: value.clone() for name, value in classification.items()}
        first_name = next(iter(changed))
        changed[first_name].view(-1)[0] += 1.0
        report = compare_state_dicts_exact(classification, changed)
        self.assertFalse(report["exact"])
        self.assertIn(first_name, report["value_mismatch_names"])

    def test_state_dict_structure_records_same_architecture(self):
        model = self.make_model()
        rejection = select_state_dict_prefixes(
            model.state_dict(),
            ("rejection_encoder", "rejection_projector", "rejection_classifier"),
        )
        structure = state_dict_structure(rejection)
        self.assertTrue(structure)
        self.assertEqual(structure, sorted(structure, key=lambda item: item["name"]))

    def test_rejection_only_optimizer_excludes_classification_parameters(self):
        model = self.make_model()
        optimizer, report = build_rejection_only_optimizer(model, lr=1e-3)
        self.assertEqual(report["optimizer"]["classification_parameter_names"], [])
        self.assertEqual(report["optimizer"]["unexpected_parameter_names"], [])
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertGreater(len(optimizer.param_groups[0]["params"]), 0)

    def test_snapshot_detects_only_changed_path(self):
        model = self.make_model()
        before = parameter_snapshot(model)
        with torch.no_grad():
            next(model.rejection_encoder.parameters()).add_(1.0)
        report = compare_parameter_snapshots(before, parameter_snapshot(model))
        self.assertGreater(report["changed_count"], 0)
        self.assertTrue(all(name.startswith("rejection_encoder.") for name in report["changed_names"]))

    def test_classification_only_checkpoint_load_preserves_rejection_initialization(self):
        source = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pt"
            save_checkpoint(source, path, extra={"known_classes": [0, 1, 2]})
            target = self.make_model()
            rejection_before = parameter_snapshot(target)
            ckpt = load_checkpoint(
                target,
                path,
                torch.device("cpu"),
                expected_known_classes=[0, 1, 2],
                load_mode="classification_only",
            )
            self.assertEqual(ckpt["_load_audit"]["mode"], "classification_only")
            rejection_after = parameter_snapshot(target)
            rejection_names = {
                name
                for name in rejection_before
                if name == "rejection_encoder"
                or name.startswith("rejection_encoder.")
                or name == "rejection_input_norm"
                or name.startswith("rejection_input_norm.")
                or name == "rejection_projector"
                or name.startswith("rejection_projector.")
                or name == "rejection_classifier"
                or name.startswith("rejection_classifier.")
                or name == "rejection_uncertainty_head"
                or name.startswith("rejection_uncertainty_head.")
            }
            rejection_report = compare_parameter_snapshots(
                {name: rejection_before[name] for name in rejection_names},
                {name: rejection_after[name] for name in rejection_names},
            )
            self.assertEqual(rejection_report["changed_count"], 0)

    def test_batchnorm_snapshot_is_available_for_protocol_audit(self):
        snapshot = batchnorm_snapshot(self.make_model())
        self.assertTrue(snapshot)

    @staticmethod
    def make_artifact_args(feature_mode="embedding"):
        return SimpleNamespace(
            rejector_feature_mode=feature_mode,
            score_mode="feature_rejector",
            rejector_training="hard",
            discovery_pool_mode="unknown",
            rejector_known_prior=0.5,
            rejector_nnpu_risk="nnpu",
            rejector_nnpu_model="logistic",
        )

    @staticmethod
    def make_artifact_outputs(rejector):
        rng = np.random.default_rng(20261114)
        outputs = []
        for count in (8, 6, 5):
            features = rng.normal(size=(count, 4)).astype(np.float32)
            current = {"features": features}
            attach_feature_rejector_score(current, rejector, feature_mode="embedding")
            outputs.append(current)
        return outputs

    def test_embedding_artifact_is_strictly_recomputable(self):
        rng = np.random.default_rng(7)
        known = {"features": rng.normal(loc=-1.0, size=(12, 4)).astype(np.float32)}
        unknown = {"features": rng.normal(loc=1.0, size=(12, 4)).astype(np.float32)}
        rejector = fit_feature_rejector(
            known, unknown, seed=7, feature_mode="embedding", model_type="logistic"
        )
        outputs_test, outputs_val, outputs_open_val = self.make_artifact_outputs(rejector)
        with tempfile.TemporaryDirectory() as directory:
            result = save_feature_rejector_artifact(
                directory,
                rejector,
                self.make_artifact_args(),
                outputs_test,
                outputs_val,
                outputs_open_val,
            )
            self.assertTrue(result["restorable"])
            report = audit_run(Path(directory))
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["results"]), 3)
            self.assertTrue(all(item["allclose"] for item in report["results"]))

    def test_artifact_requires_scores_for_every_persisted_split(self):
        rejector = fit_feature_rejector(
            {"features": np.zeros((4, 4), dtype=np.float32)},
            {"features": np.ones((4, 4), dtype=np.float32)},
            seed=3,
            feature_mode="embedding",
            model_type="logistic",
        )
        outputs = {"features": np.zeros((2, 3), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing feature_rejector_score"):
                save_feature_rejector_artifact(
                    directory,
                    rejector,
                    self.make_artifact_args(),
                    outputs,
                    None,
                    None,
                )

    def test_support_augmented_artifact_requires_support_model(self):
        rejector = fit_feature_rejector(
            {"features": np.zeros((4, 3), dtype=np.float32)},
            {"features": np.ones((4, 3), dtype=np.float32)},
            seed=4,
            feature_mode="embedding",
            model_type="logistic",
        )
        outputs = {
            "features": np.zeros((2, 3), dtype=np.float32),
            "logits": np.zeros((2, 2), dtype=np.float32),
            "probs": np.full((2, 2), 0.5, dtype=np.float32),
            "entropy": np.zeros(2, dtype=np.float32),
            "head_uncertainty": np.zeros(2, dtype=np.float32),
        }
        attach_feature_rejector_score(outputs, rejector, feature_mode="embedding")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires a support model"):
                save_feature_rejector_artifact(
                    directory,
                    rejector,
                    self.make_artifact_args("support_augmented"),
                    outputs,
                    None,
                    None,
                )

    def test_artifact_hashes_detect_input_or_score_tampering(self):
        rejector = fit_feature_rejector(
            {"features": np.zeros((4, 4), dtype=np.float32)},
            {"features": np.ones((4, 4), dtype=np.float32)},
            seed=5,
            feature_mode="embedding",
            model_type="logistic",
        )
        outputs = self.make_artifact_outputs(rejector)[0]
        with tempfile.TemporaryDirectory() as directory:
            save_feature_rejector_artifact(
                directory,
                rejector,
                self.make_artifact_args(),
                outputs,
                None,
                None,
            )
            scores_path = Path(directory) / "feature_rejector_scores.npz"
            with np.load(scores_path, allow_pickle=False) as archive:
                payload = {key: np.asarray(archive[key]) for key in archive.files}
            first_key = sorted(payload)[0]
            payload[first_key] = payload[first_key].copy()
            if payload[first_key].dtype.kind in "fc":
                payload[first_key].flat[0] += 1.0
            else:
                payload[first_key].flat[0] += 1
            np.savez_compressed(scores_path, **payload)
            with self.assertRaisesRegex(ValueError, "scores.npz hash"):
                audit_run(Path(directory))

    def test_legacy_artifact_is_reported_as_not_strictly_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "feature_rejector.json"
            metadata_path.write_text(
                '{"artifact_version": 2, "model_file": "feature_rejector.pkl"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "legacy artifact"):
                audit_run(Path(directory))


if __name__ == "__main__":
    unittest.main()
