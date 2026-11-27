import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel_discovery.losses import feature_variance_floor_loss


def test_variance_floor_penalizes_collapsed_rejection_features():
    collapsed = torch.ones(8, 4)
    diverse = torch.eye(4).repeat(2, 1)

    collapsed_loss = feature_variance_floor_loss(collapsed, floor=0.05)
    diverse_loss = feature_variance_floor_loss(diverse, floor=0.05)

    assert collapsed_loss.item() > diverse_loss.item()
    assert collapsed_loss.item() > 0.0


def test_variance_floor_is_invariant_to_projector_scale():
    features = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )

    first = feature_variance_floor_loss(features, floor=0.2)
    scaled = feature_variance_floor_loss(features * 17.0, floor=0.2)

    assert torch.allclose(first, scaled, atol=1e-6)
