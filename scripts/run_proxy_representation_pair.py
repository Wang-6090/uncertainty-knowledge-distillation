"""Paired full-data proxy-contrastive representation pilot.

Regenerates baseline and treatment students from the saved semantic-isolated
recipes for seeds 44/123. The sole treatment is alpha_proxy=0.05. Each arm is
evaluated with the same saved discovery protocol and its own paired checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args


RUNS = ROOT / "runs"


def load_args(config_path: Path, command: str, work_dir: Path, seed: int):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = seed
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def run_seed(seed: int, alpha_proxy: float) -> None:
    baseline_source = RUNS / f"semantic_isolated_full_s{seed}_disjointcal_lr01_student"
    baseline_discovery_source = RUNS / f"semantic_isolated_full_s{seed}_disjointcal_lr01_discover"
    if not (baseline_source / "train_student_config.json").is_file():
        raise FileNotFoundError(baseline_source / "train_student_config.json")
    if not (baseline_discovery_source / "discover_config.json").is_file():
        raise FileNotFoundError(baseline_discovery_source / "discover_config.json")

    teacher_ckpt = Path(json.loads((baseline_source / "train_student_config.json").read_text(encoding="utf-8"))["teacher_ckpt"])
    if not teacher_ckpt.is_absolute():
        teacher_ckpt = ROOT / teacher_ckpt
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(teacher_ckpt)

    for arm, weight in (("baseline", 0.0), ("proxy005", alpha_proxy)):
        tag = f"proxy_repr_s{seed}_{arm}"
        student_dir = RUNS / tag
        student_dir.mkdir(parents=True, exist_ok=True)
        train_args = load_args(
            baseline_source / "train_student_config.json", "train_student", student_dir, seed
        )
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(student_dir / "student.pt")
        train_args.alpha_proxy = weight
        fit_student(train_args)

        discovery_dir = RUNS / f"{tag}_discover"
        discovery_args = load_args(
            baseline_discovery_source / "discover_config.json",
            "discover",
            discovery_dir,
            seed,
        )
        discovery_args.student_ckpt = train_args.student_ckpt
        discover(discovery_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 44])
    parser.add_argument("--alpha-proxy", type=float, default=0.05)
    args = parser.parse_args()
    for seed in args.seeds:
        run_seed(seed, args.alpha_proxy)


if __name__ == "__main__":
    main()
