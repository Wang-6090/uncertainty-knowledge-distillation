"""Compare linear and MLP nnPU rejectors on one fixed student checkpoint."""

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
    args = parser.parse_args()
    split = ROOT / "splits_cifar100_protocols/cifar100_60_40_semantic_isolated.json"
    student = ROOT / f"runs/rej_uniform_lr01_openval_s{args.seed}_margin_only_student/student.pt"
    if not student.is_file():
        raise FileNotFoundError(student)
    common = [
        sys.executable, "train.py", "discover", "--dataset", "cifar100",
        "--data-root", ".\\data", "--num-known", "60", "--seed", str(args.seed),
        "--split-path", str(split), "--known-split-mode", "random", "--image-size", "64",
        "--batch-size", "64", "--num-workers", "0", "--device", "auto",
        "--open-val-ratio", "0.1", "--calibration-ratio", "0.25",
        "--discovery-pool-mode", "mixed", "--matched-discovery-pool",
        "--matched-pool-size", "5400", "--matched-known-prior", "0.2",
        "--backbone", "resnet18", "--student-backbone", "resnet18",
        "--rejection-feature-dim", "128", "--num-novel", "40", "--mc-samples", "4",
        "--score-mode", "feature_rejector", "--rejector-training", "nnpu",
        "--rejector-nnpu-risk", "nu_corrected", "--rejector-known-prior", "0.2",
        "--rejector-feature-mode", "support_augmented", "--threshold-policy",
        "known_coverage", "--target-known-coverage", "0.95", "--skip-clustering",
        "--student-ckpt", str(student),
    ]
    for model in ("linear", "mlp"):
        output = ROOT / f"runs/rejection_{model}_s{args.seed}_discover"
        run([*common, "--rejector-nnpu-model", model, "--work-dir", str(output)])


if __name__ == "__main__":
    main()
