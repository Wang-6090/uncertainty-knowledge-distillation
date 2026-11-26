import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_srd_scale_geometry import summarize


def test_geometry_summary_reports_class_compactness_and_global_scatter():
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    result = summarize(features, labels, classes=2)

    assert result["samples"] == 6
    assert result["normalized_within_cosine_distance_mean"] > 0.0
    assert result["normalized_nearest_center_cosine_distance_mean"] > 0.0
    assert result["normalized_between_total_scatter_ratio"] > 0.0
