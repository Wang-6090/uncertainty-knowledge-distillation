"""Matched two-arm rejection-branch OE replication."""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()
    seed = args.seed
    teacher = ROOT / f"runs/semantic_isolated_full_s{seed}_disjointcal_teacher/teacher.pt"
    split = ROOT / "splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json"
    if not teacher.is_file():
        raise FileNotFoundError(teacher)

    common_train = [
        "train.py", "train_student", "--dataset", "cifar100", "--data-root", ".\\data",
        "--num-known", "60", "--seed", str(seed), "--split-path", str(split),
        "--known-split-mode", "random", "--image-size", "64", "--batch-size", "64",
        "--num-workers", "0", "--device", "auto", "--open-val-ratio", "0.1",
        "--backbone", "resnet18", "--teacher-backbone", "resnet34",
        "--student-backbone", "resnet18", "--pretrained", "--epochs", "5",
        "--alpha-unc", "0.1", "--alpha-kd", "1.0", "--kd-mode", "uncertainty",
        "--temperature", "2.0", "--uncertainty-weight-mode", "raw", "--alpha-feat-kd", "0.1",
        "--alpha-proto", "0.1", "--alpha-supcon", "0.1",
        "--alpha-discovery-uncertainty-pu", "0.1", "--discovery-uncertainty-pu-risk", "nu_corrected",
        "--discovery-uncertainty-known-prior", "0.2", "--discovery-pool", "--discovery-pool-mode", "mixed",
        "--matched-discovery-pool", "--matched-pool-size", "5400", "--matched-known-prior", "0.2",
        "--rejection-feature-dim", "128", "--alpha-rejection-known-ce", "1.0",
        "--outlier-dataset", "cifar10", "--outlier-data-root", ".\\data",
        "--alpha-outlier-rejection-feature-margin", "0.05", "--outlier-rejection-feature-margin", "0.2",
        "--teacher-ckpt", str(teacher),
    ]
    common_discover = [
        "train.py", "discover", "--dataset", "cifar100", "--data-root", ".\\data",
        "--num-known", "60", "--seed", str(seed), "--split-path", str(split),
        "--known-split-mode", "random", "--image-size", "64", "--batch-size", "64",
        "--num-workers", "0", "--device", "auto", "--open-val-ratio", "0.1",
        "--calibration-ratio", "0.25", "--discovery-pool-mode", "mixed",
        "--matched-discovery-pool", "--matched-pool-size", "5400", "--matched-known-prior", "0.2",
        "--backbone", "resnet18", "--student-backbone", "resnet18", "--rejection-feature-dim", "128",
        "--num-novel", "40", "--mc-samples", "4", "--score-mode", "feature_rejector",
        "--rejector-training", "nnpu", "--rejector-nnpu-risk", "nu_corrected", "--rejector-nnpu-model", "linear",
        "--rejector-known-prior", "0.2", "--rejector-feature-mode", "support_augmented",
        "--threshold-policy", "known_coverage", "--target-known-coverage", "0.95", "--skip-clustering",
    ]

    for arm, uniform in (("margin_only", "0.0"), ("margin_plus_uniform", "0.05")):
        tag = f"rej_uniform_lr01_openval_s{seed}_{arm}"
        student_dir = ROOT / "runs" / f"{tag}_student"
        discover_dir = ROOT / "runs" / f"{tag}_discover"
        student = student_dir / "student.pt"
        if not student.is_file():
            run([sys.executable, *common_train, "--encoder-lr-scale", "0.1",
                 "--alpha-outlier-rejection-uniform", uniform,
                 "--work-dir", str(student_dir), "--student-ckpt", str(student)])
        else:
            print(f"Reusing {student}")
        run([sys.executable, *common_discover, "--work-dir", str(discover_dir),
             "--student-ckpt", str(student)])


if __name__ == "__main__":
    main()
