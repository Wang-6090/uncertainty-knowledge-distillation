"""Run independent-rejection-encoder treatments against fixed matched controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import discover, fit_student, parse_args  # noqa: E402


def load_args(path: Path, command: str, work_dir: Path, seed: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.work_dir = str(work_dir)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(2040, 2041))
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    options = parser.parse_args()

    control_dir = options.control_dir.resolve()
    source_train = control_dir / "train_student_config.json"
    source_discover = control_dir.parent / f"{control_dir.name}_discover" / "discover_config.json"
    control_ckpt = control_dir / "student.pt"
    control_report = source_discover.parent / "discovery_report.json"
    for required in (source_train, source_discover, control_ckpt, control_report):
        if not required.is_file():
            raise FileNotFoundError(required)

    train_config = json.loads(source_train.read_text(encoding="utf-8"))
    if train_config.get("seed") != options.seed:
        raise ValueError("control seed does not match requested seed")
    if train_config.get("alpha_vos", 0.0) != 0.0:
        raise ValueError("control must be the no-VOS baseline for this pair")
    if train_config.get("independent_rejection_encoder", False):
        raise ValueError("control unexpectedly enables independent rejection encoder")

    treatment_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_treatment"
    treatment_discover_dir = ROOT / "runs" / f"{options.tag}_s{options.seed}_treatment_discover"
    if treatment_discover_dir.exists():
        raise FileExistsError("refusing to overwrite an existing treatment discovery output")
    treatment_ckpt = treatment_dir / "student.pt"
    if treatment_dir.exists() and not treatment_ckpt.is_file():
        raise FileExistsError(
            "treatment output directory exists without a complete checkpoint; "
            "inspect it manually before retrying"
        )

    treatment_args = load_args(source_train, "train_student", treatment_dir, options.seed)
    treatment_args.independent_rejection_encoder = True
    treatment_args.student_ckpt = str(treatment_ckpt)
    if treatment_args.alpha_vos_rejection > 0.0:
        raise ValueError("independent encoder is incompatible with VOS rejection BCE")

    # Compare the fully materialized training arguments, masking only output
    # location/checkpoint and the single treatment switch.
    control_args = load_args(source_train, "train_student", control_dir, options.seed)
    control_args.student_ckpt = str(control_ckpt)
    ignored = {"work_dir", "student_ckpt", "independent_rejection_encoder"}
    differences = {
        key: (getattr(control_args, key), getattr(treatment_args, key))
        for key in vars(control_args)
        if key not in ignored and getattr(control_args, key) != getattr(treatment_args, key)
    }
    if differences:
        raise ValueError(f"unexpected train config differences: {differences}")

    if not treatment_ckpt.is_file():
        fit_student(treatment_args)
    discover_args = load_args(
        source_discover, "discover", treatment_discover_dir, options.seed
    )
    discover_args.independent_rejection_encoder = True
    discover_args.student_ckpt = treatment_args.student_ckpt
    discover(discover_args)
    print(json.dumps({"seed": options.seed, "control": str(control_dir),
                      "treatment": str(treatment_dir),
                      "control_discovery": str(control_report),
                      "treatment_discovery": str(treatment_discover_dir / "discovery_report.json")},
                     indent=2))


if __name__ == "__main__":
    main()
