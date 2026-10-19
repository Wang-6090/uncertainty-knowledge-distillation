"""Paired, short training audit of VOS source-feature gradient routing.

This is a mechanism probe, not a model-performance experiment. The two arms
share the audited seed-123 recipe and differ only in whether synthesized
inter-class virtual features retain gradients to their source features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import fit_student, parse_args  # noqa: E402


def run_arm(source_config: Path, work_dir: Path, source_gradient: bool) -> None:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    args = parse_args(["train_student"])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = 123
    args.model_seed = None
    args.epochs = 1
    args.limit_train = 512
    args.audit_loss_gradients = True
    args.vos_mode = "interclass_mix"
    args.vos_interclass_pairing = "nearest_class"
    args.vos_boundary_fraction = 0.25
    args.alpha_vos = 0.01
    args.alpha_vos_rejection = 0.0
    args.vos_source_gradient = bool(source_gradient)
    args.work_dir = str(work_dir)
    # The source experiment config contains an absolute student checkpoint
    # path. Always redirect output before fitting, or an audit could overwrite
    # the archived experiment being used as its source.
    args.student_ckpt = str(work_dir / "student.pt")
    work_dir.mkdir(parents=True, exist_ok=True)
    fit_student(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=ROOT
        / "runs/vos_boundary_replicate_s123_e4_interclass_mix/train_student_config.json",
    )
    parser.add_argument(
        "--tag", default="vos_gradient_path_audit_s123_512_e1"
    )
    args = parser.parse_args()
    if not args.source_config.is_file():
        raise FileNotFoundError(args.source_config)
    for name, source_gradient in (("detached", False), ("source_gradient", True)):
        run_arm(
            args.source_config,
            ROOT / "runs" / f"{args.tag}_{name}",
            source_gradient,
        )


if __name__ == "__main__":
    main()
