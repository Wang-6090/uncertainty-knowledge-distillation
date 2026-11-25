import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from novel_discovery.losses import classification_view_consistency_loss
from novel_discovery.pipeline import train_one_epoch_student


def test_classification_view_consistency_is_finite_and_updates_student_features():
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    second = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    loss = classification_view_consistency_loss(first, second)
    loss.backward()

    assert torch.isfinite(loss)
    assert first.grad is not None
    assert torch.linalg.vector_norm(first.grad) > 0.0
    assert not second.requires_grad


def test_classification_view_consistency_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        classification_view_consistency_loss(torch.zeros(2, 3), torch.zeros(2, 4))


def test_classification_view_consistency_zero_batch_is_finite():
    first = torch.zeros(0, 3, requires_grad=True)
    second = torch.zeros(0, 3)
    loss = classification_view_consistency_loss(first, second)

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


class _TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 5)
        self.classifier = torch.nn.Linear(5, 2)
        self.projector = torch.nn.Linear(5, 4)
        self.uncertainty_head = torch.nn.Linear(5, 1)
        self.dropout_p = 0.0

    def forward(self, images, stochastic=False):
        features = self.encoder(images.mean(dim=(2, 3)))
        return {
            "logits": self.classifier(features),
            "features": features,
            "proj": F.normalize(self.projector(features), dim=-1),
            "uncertainty": torch.sigmoid(self.uncertainty_head(features)).squeeze(-1),
        }


def test_paired_training_path_records_active_student_view_consistency_gradient():
    torch.manual_seed(12)
    student = _TinyStudent()
    teacher = _TinyStudent()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    first = torch.rand(4, 3, 8, 8)
    second = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    loader = [(first, second, labels, labels, torch.ones(4, dtype=torch.bool))]

    stats = train_one_epoch_student(
        student,
        teacher,
        loader,
        optimizer,
        torch.device("cpu"),
        alpha_unc=0.0,
        alpha_kd=0.0,
        alpha_feat_kd=0.0,
        alpha_supcon=0.0,
        alpha_proto=0.0,
        alpha_classification_view_consistency=0.5,
        paired_classification_views=True,
        audit_loss_gradients=True,
    )

    assert stats["classification_view_consistency_applied"] is True
    assert stats["paired_classification_views"] is True
    assert stats["classification_view_consistency"] > 0.0
    assert stats["gradient_audit_grad_norm_classification_view_consistency_term"] > 0.0


def test_zero_coefficient_keeps_paired_consistency_inactive():
    student = _TinyStudent()
    teacher = _TinyStudent()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    images = torch.rand(2, 3, 8, 8)
    labels = torch.tensor([0, 1])
    loader = [(images, images.clone(), labels, labels, torch.ones(2, dtype=torch.bool))]

    stats = train_one_epoch_student(
        student,
        teacher,
        loader,
        optimizer,
        torch.device("cpu"),
        alpha_unc=0.0,
        alpha_kd=0.0,
        alpha_feat_kd=0.0,
        alpha_supcon=0.0,
        alpha_proto=0.0,
        paired_classification_views=True,
    )

    assert stats["classification_view_consistency_applied"] is False
    assert stats["classification_view_consistency"] == 0.0
