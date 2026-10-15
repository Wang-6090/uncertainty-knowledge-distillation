"""Run a post-hoc fusion pilot for the independent rejection branch.

The student checkpoint and all data/calibration settings are fixed.  Only the
fusion base score and the normalized rejector weight change, so this is a
detector ablation rather than a training comparison.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--student-ckpt",
        default=None,
        help="Fixed student checkpoint; defaults to the seed-123 margin-only pilot.",
    )
    args = parser.parse_args()
    seed = args.seed
    split = ROOT / "splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json"
    student = Path(args.student_ckpt) if args.student_ckpt else ROOT / (
        f"runs/rej_uniform_lr01_openval_s{seed}_margin_only_student/student.pt"
    )
    if not student.is_file():
        raise FileNotFoundError(student)

    common = [
        sys.executable,
        "train.py",
        "discover",
        "--dataset",
        "cifar100",
        "--data-root",
        ".\\data",
        "--num-known",
        "60",
        "--seed",
        str(seed),
        "--split-path",
        str(split),
        "--known-split-mode",
        "random",
        "--image-size",
        "64",
        "--batch-size",
        "64",
        "--num-workers",
        "0",
        "--device",
        "auto",
        "--open-val-ratio",
        "0.1",
        "--calibration-ratio",
        "0.25",
        "--discovery-pool-mode",
        "mixed",
        "--matched-discovery-pool",
        "--matched-pool-size",
        "5400",
        "--matched-known-prior",
        "0.2",
        "--backbone",
        "resnet18",
        "--student-backbone",
        "resnet18",
        "--rejection-feature-dim",
        "128",
        "--num-novel",
        "40",
        "--mc-samples",
        "4",
        "--knn-ood",
        "--score-mode",
        "feature_rejector_fusion",
        "--rejector-training",
        "nnpu",
        "--rejector-nnpu-risk",
        "nu_corrected",
        "--rejector-nnpu-model",
        "linear",
        "--rejector-known-prior",
        "0.2",
        "--rejector-feature-mode",
        "support_augmented",
        "--threshold-policy",
        "known_coverage",
        "--target-known-coverage",
        "0.95",
        "--skip-clustering",
        "--student-ckpt",
        str(student),
    ]

    trials = [
        ("knn_w025", "normalized_entropy_min_class_knn", "0.25"),
        ("knn_w050", "normalized_entropy_min_class_knn", "0.50"),
        ("knn_w075", "normalized_entropy_min_class_knn", "0.75"),
        ("proto_w050", "normalized_entropy_proto", "0.50"),
    ]
    for tag, base_score, weight in trials:
        output = ROOT / f"runs/rejection_fusion_s{seed}_{tag}_discover"
        cmd = [
            *common,
            "--fusion-base-score",
            base_score,
            "--fusion-rejector-weight",
            weight,
            "--work-dir",
            str(output),
        ]
        run(cmd)


if __name__ == "__main__":
    main()
