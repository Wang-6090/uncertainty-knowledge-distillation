"""Paired full-data nnPU vs nnPU+uncertainty-feature-margin experiment."""

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


def run_seed(seed: int, margin_weight: float, margin: float) -> None:
    source = RUNS / f"semantic_isolated_full_s{seed}_disjointcal_lr01_student"
    discovery_source = RUNS / f"semantic_isolated_full_s{seed}_disjointcal_lr01_discover"
    train_config = source / "train_student_config.json"
    discovery_config = discovery_source / "discover_config.json"
    if not train_config.is_file() or not discovery_config.is_file():
        raise FileNotFoundError(f"Missing paired recipe for seed {seed}: {source}")

    source_config = json.loads(train_config.read_text(encoding="utf-8"))
    teacher_ckpt = Path(source_config["teacher_ckpt"])
    if not teacher_ckpt.is_absolute():
        teacher_ckpt = ROOT / teacher_ckpt
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(teacher_ckpt)

    for arm, weight in (("baseline", 0.0), ("margin005", margin_weight)):
        tag = f"nnpu_margin_pair_s{seed}_{arm}"
        student_dir = RUNS / tag
        student_dir.mkdir(parents=True, exist_ok=True)
        train_args = load_args(train_config, "train_student", student_dir, seed)
        train_args.teacher_ckpt = str(teacher_ckpt)
        train_args.student_ckpt = str(student_dir / "student.pt")
        train_args.alpha_discovery_uncertainty_feature_margin = weight
        train_args.discovery_uncertainty_feature_margin = margin
        fit_student(train_args)

        discovery_dir = RUNS / f"{tag}_discover"
        discovery_args = load_args(discovery_config, "discover", discovery_dir, seed)
        discovery_args.student_ckpt = train_args.student_ckpt
        discover(discovery_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 123])
    parser.add_argument("--margin-weight", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()
    for seed in args.seeds:
        run_seed(seed, args.margin_weight, args.margin)


if __name__ == "__main__":
    main()
