"""Paired pilot for a rejection-side class-conditional support boundary.

The baseline student is the previously audited margin-only checkpoint.  The
treatment changes exactly one student objective: a class-wise kNN support
boundary in the independent rejection embedding.  The discovery detector and
coverage calibration are identical in both arms.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    seed = args.seed
    split = ROOT / "splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json"
    teacher = ROOT / f"runs/semantic_isolated_full_s{seed}_disjointcal_teacher/teacher.pt"
    baseline_student = ROOT / f"runs/rej_uniform_lr01_openval_s{seed}_margin_only_student/student.pt"
    if not teacher.is_file():
        raise FileNotFoundError(teacher)
    if not baseline_student.is_file():
        raise FileNotFoundError(baseline_student)

    train_dir = ROOT / f"runs/rejection_local_boundary_s{seed}_student"
    student = train_dir / "student.pt"
    if not student.is_file():
        run([
            sys.executable, "train.py", "train_student",
            "--dataset", "cifar100", "--data-root", ".\\data",
            "--num-known", "60", "--seed", str(seed), "--split-path", str(split),
            "--known-split-mode", "random", "--image-size", "64",
            "--batch-size", "64", "--num-workers", "0", "--device", "auto",
            "--open-val-ratio", "0.1", "--backbone", "resnet18",
            "--teacher-backbone", "resnet34", "--student-backbone", "resnet18",
            "--pretrained", "--epochs", "5", "--alpha-unc", "0.1",
            "--alpha-kd", "1.0", "--kd-mode", "uncertainty", "--temperature", "2.0",
            "--uncertainty-weight-mode", "raw", "--alpha-feat-kd", "0.1",
            "--alpha-proto", "0.1", "--alpha-supcon", "0.1",
            "--alpha-discovery-uncertainty-pu", "0.1",
            "--discovery-uncertainty-pu-risk", "nu_corrected",
            "--discovery-uncertainty-known-prior", "0.2", "--discovery-pool",
            "--discovery-pool-mode", "mixed", "--matched-discovery-pool",
            "--matched-pool-size", "5400", "--matched-known-prior", "0.2",
            "--rejection-feature-dim", "128", "--alpha-rejection-known-ce", "1.0",
            "--encoder-lr-scale", "0.1", "--alpha-outlier-rejection-feature-margin", "0.05",
            "--outlier-rejection-feature-margin", "0.2",
            "--outlier-dataset", "cifar10", "--outlier-data-root", ".\\data",
            "--alpha-discovery-rejection-knn-boundary", str(args.alpha),
            "--discovery-feature-candidate-gating", "--discovery-cross-view-gating",
            "--discovery-select-ratio", "0.25", "--discovery-select-mode", "consensus",
            "--discovery-knn-k", "5", "--discovery-knn-quantile", "0.95",
            "--discovery-knn-margin", "0.02", "--teacher-ckpt", str(teacher),
            "--work-dir", str(train_dir), "--student-ckpt", str(student),
        ])
    else:
        print(f"Reusing {student}")

    common = [
        sys.executable, "train.py", "discover", "--dataset", "cifar100",
        "--data-root", ".\\data", "--num-known", "60", "--seed", str(seed),
        "--split-path", str(split), "--known-split-mode", "random", "--image-size", "64",
        "--batch-size", "64", "--num-workers", "0", "--device", "auto",
        "--open-val-ratio", "0.1", "--calibration-ratio", "0.25",
        "--discovery-pool-mode", "mixed", "--matched-discovery-pool",
        "--matched-pool-size", "5400", "--matched-known-prior", "0.2",
        "--backbone", "resnet18", "--student-backbone", "resnet18",
        "--rejection-feature-dim", "128", "--num-novel", "40", "--mc-samples", "4",
        "--score-mode", "feature_rejector", "--rejector-training", "nnpu",
        "--rejector-nnpu-risk", "nu_corrected", "--rejector-nnpu-model", "linear",
        "--rejector-known-prior", "0.2", "--rejector-feature-mode", "support_augmented",
        "--threshold-policy", "known_coverage", "--target-known-coverage", "0.95",
        "--skip-clustering",
    ]
    for tag, checkpoint in (
        ("baseline", baseline_student),
        ("treatment", student),
    ):
        output = ROOT / f"runs/rejection_local_boundary_s{seed}_{tag}_discover"
        report = output / "discovery_report.json"
        if report.is_file():
            print(f"Reusing {report}")
            continue
        run([*common, "--work-dir", str(output), "--student-ckpt", str(checkpoint)])


if __name__ == "__main__":
    main()
