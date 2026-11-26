"""Run a known-only teacher training-budget pilot.

The only treatment factor is teacher epochs (5 vs 15). The protocol matches
the formal Team A teacher recipe and uses a fresh semantic-hard split. This
runner never trains a student or evaluates unknown/test samples.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_split(path: Path, seed: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "semantic_hard":
        raise ValueError("pilot requires a semantic_hard split")
    if payload.get("metadata", {}).get("seed") != seed:
        raise ValueError("split metadata seed does not match requested seed")
    if len(payload.get("known_classes", [])) != 60:
        raise ValueError("pilot requires a 60/40 split")


def run_arm(seed: int, split_path: Path, root: Path, epochs: int, device: str) -> None:
    work_dir = root / f"teacher_quality_budget_s{seed}_e{epochs}"
    if work_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {work_dir}")
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "train_teacher",
        "--dataset",
        "cifar100",
        "--data-root",
        "data",
        "--split-path",
        str(split_path),
        "--num-known",
        "60",
        "--seed",
        str(seed),
        "--image-size",
        "64",
        "--batch-size",
        "64",
        "--num-workers",
        "0",
        "--backbone",
        "resnet34",
        "--pretrained",
        "--transform-profile",
        "cifar",
        "--epochs",
        str(epochs),
        "--lr",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--alpha-unc",
        "0.1",
        "--alpha-proto",
        "0.0",
        "--alpha-pseudo",
        "0.0",
        "--work-dir",
        str(work_dir),
        "--teacher-ckpt",
        str(work_dir / "teacher.pt"),
        "--device",
        device,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2182)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("splits/semantic_hard/seed2182/cifar100_60_40_semantic_hard.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    split_path = (ROOT / args.split_path).resolve() if not args.split_path.is_absolute() else args.split_path.resolve()
    output_root = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    load_split(split_path, args.seed)
    output_root.mkdir(parents=True, exist_ok=True)
    for epochs in (5, 15):
        run_arm(args.seed, split_path, output_root, epochs, args.device)
    print(json.dumps({"seed": args.seed, "epochs": [5, 15], "split": str(split_path)}, indent=2))


if __name__ == "__main__":
    main()
