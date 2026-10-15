"""Paired pilot for shared versus rejection-space candidate gating.

Both arms use the same teacher, seed, mixed discovery pool, rejection kNN
boundary loss, optimizer and training budget.  The only method variable is
the representation used to select the candidate samples on which that loss
acts.  This is intentionally a short diagnostic pilot; it is not a final
hyperparameter selection.
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    seed = int(args.seed)
    split = ROOT / "splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json"
    teacher = ROOT / f"runs/semantic_isolated_full_s{seed}_disjointcal_teacher/teacher.pt"
    if not teacher.is_file():
        raise FileNotFoundError(teacher)

    common = [
        sys.executable,
        "train.py",
        "train_student",
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
        str(args.device),
        "--open-val-ratio",
        "0.1",
        "--backbone",
        "resnet18",
        "--teacher-backbone",
        "resnet34",
        "--student-backbone",
        "resnet18",
        "--pretrained",
        "--epochs",
        str(args.epochs),
        "--alpha-unc",
        "0.1",
        "--alpha-kd",
        "1.0",
        "--kd-mode",
        "uncertainty",
        "--temperature",
        "2.0",
        "--uncertainty-weight-mode",
        "raw",
        "--alpha-feat-kd",
        "0.1",
        "--alpha-proto",
        "0.1",
        "--alpha-supcon",
        "0.1",
        "--alpha-discovery-uncertainty-pu",
        "0.1",
        "--discovery-uncertainty-pu-risk",
        "nu_corrected",
        "--discovery-uncertainty-known-prior",
        "0.2",
        "--discovery-pool",
        "--discovery-pool-mode",
        "mixed",
        "--matched-discovery-pool",
        "--matched-pool-size",
        "5400",
        "--matched-known-prior",
        "0.2",
        "--rejection-feature-dim",
        "128",
        "--alpha-rejection-known-ce",
        "1.0",
        "--encoder-lr-scale",
        "0.1",
        "--alpha-outlier-rejection-feature-margin",
        "0.05",
        "--outlier-rejection-feature-margin",
        "0.2",
        "--outlier-dataset",
        "cifar10",
        "--outlier-data-root",
        ".\\data",
        "--alpha-discovery-rejection-knn-boundary",
        "0.1",
        "--discovery-feature-candidate-gating",
        "--discovery-cross-view-gating",
        "--discovery-select-ratio",
        "0.25",
        "--discovery-select-mode",
        "consensus",
        "--discovery-knn-k",
        "5",
        "--discovery-knn-quantile",
        "0.95",
        "--discovery-knn-margin",
        "0.02",
        "--teacher-ckpt",
        str(teacher),
    ]
    for space in ("shared", "rejection_support"):
        tag = f"rejection_candidate_space_s{seed}_e{int(args.epochs)}_{space}"
        work_dir = ROOT / "runs" / tag
        student = work_dir / "student.pt"
        if student.is_file():
            print(f"Reusing {student}")
            continue
        run(
            [
                *common,
                "--discovery-feature-candidate-space",
                space,
                "--work-dir",
                str(work_dir),
                "--student-ckpt",
                str(student),
            ]
        )


if __name__ == "__main__":
    main()
