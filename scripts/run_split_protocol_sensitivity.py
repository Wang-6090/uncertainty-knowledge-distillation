"""Run a matched CIFAR-100 split-protocol sensitivity pilot.

The only planned experimental factor is the known/novel class split protocol
(semantic_isolated versus semantic_hard). Both arms start from the same saved
training/discovery argument templates, seed, and epoch budget, and use separate
output directories so no existing checkpoint can be overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import discover, fit_student, fit_teacher, parse_args  # noqa: E402


PROTOCOLS = ("semantic_isolated", "semantic_hard")
SPLIT_DIR = ROOT / "splits_cifar100_protocols"
METRIC_KEYS = (
    "auroc",
    "aupr",
    "fpr95",
    "oscr",
    "known_accept_rate",
    "unknown_reject_rate",
    "known_class_accuracy_after_accept",
    "unknown_false_accept",
    "known_rejected",
)


def _load_args(path: Path, command: str, *, seed: int, epochs: int):
    values = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args([command])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.seed = int(seed)
    args.model_seed = None
    args.epochs = int(epochs)
    return args


def _protocol_metadata(path: Path) -> dict:
    split = json.loads(path.read_text(encoding="utf-8"))
    fine_to_coarse = split["fine_to_coarse"]
    known_coarse = {fine_to_coarse[str(label)] for label in split["known_classes"]}
    novel_coarse = {fine_to_coarse[str(label)] for label in split["novel_classes"]}
    return {
        "protocol": split["protocol"],
        "known_class_count": len(split["known_classes"]),
        "novel_class_count": len(split["novel_classes"]),
        "known_coarse_superclass_count": len(known_coarse),
        "novel_coarse_superclass_count": len(novel_coarse),
        "coarse_superclasses_shared": len(known_coarse & novel_coarse),
        "metadata": split.get("metadata", {}),
    }


def _assert_checkpoint_split_compatible(checkpoint_path: str, split_path: Path) -> None:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required initialization checkpoint not found: {checkpoint}")
    # Check metadata before any training starts. This catches split-conditioned
    # teachers (class-head ordering and prototypes) that cannot be reused across
    # semantic protocols.
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if list(payload.get("known_classes", [])) != list(split["known_classes"]):
        raise ValueError(
            "Initialization checkpoint known_classes do not match protocol "
            f"{split['protocol']}: checkpoint={payload.get('known_classes')} "
            f"split={split['known_classes']}"
        )
    if list(payload.get("novel_classes", [])) != list(split["novel_classes"]):
        raise ValueError(
            "Initialization checkpoint novel_classes do not match protocol "
            f"{split['protocol']}: checkpoint={payload.get('novel_classes')} "
            f"split={split['novel_classes']}"
        )


def _comparable_config(args) -> dict:
    ignored = {
        "command",
        "split_path",
        "work_dir",
        "student_ckpt",
        "teacher_ckpt",
    }
    return {key: value for key, value in vars(args).items() if key not in ignored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-teacher",
        type=Path,
        default=ROOT
        / "runs/semantic_isolated_full_s44_disjointcal_teacher/train_teacher_config.json",
    )
    parser.add_argument(
        "--source-train",
        type=Path,
        default=ROOT
        / "runs/vos_source_gradient_compare_s44_e4_20261018_detached/train_student_config.json",
    )
    parser.add_argument(
        "--source-discover",
        type=Path,
        default=ROOT
        / "runs/vos_source_gradient_compare_s44_e4_20261018_detached_discover/discover_config.json",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--output-prefix", default="split_protocol_matched_teacher_s44_e4_20261018_v2"
    )
    args = parser.parse_args()

    teacher_source = args.source_teacher.resolve()
    train_source = args.source_train.resolve()
    discover_source = args.source_discover.resolve()
    if (
        not teacher_source.is_file()
        or not train_source.is_file()
        or not discover_source.is_file()
    ):
        raise FileNotFoundError("Source teacher/train/discover configs must exist")

    output_dirs = {}
    for protocol in PROTOCOLS:
        split_path = (SPLIT_DIR / f"cifar100_60_40_{protocol}.json").resolve()
        teacher_dir = ROOT / "runs" / f"{args.output_prefix}_{protocol}_teacher"
        student_dir = ROOT / "runs" / f"{args.output_prefix}_{protocol}_student"
        discover_dir = ROOT / "runs" / f"{args.output_prefix}_{protocol}_discover"
        output_dirs[protocol] = (split_path, teacher_dir, student_dir, discover_dir)
        for output_dir in (teacher_dir, student_dir, discover_dir):
            if output_dir.exists():
                raise FileExistsError(
                    f"Refusing to reuse existing output directory: {output_dir}"
                )

    teacher_configs = []
    student_configs = []
    discover_configs = []
    for protocol in PROTOCOLS:
        split_path, teacher_dir, student_dir, discover_dir = output_dirs[protocol]

        teacher_args = _load_args(
            teacher_source,
            "train_teacher",
            seed=args.seed,
            epochs=args.teacher_epochs,
        )
        teacher_args.split_path = str(split_path)
        teacher_args.work_dir = str(teacher_dir)
        teacher_args.teacher_ckpt = str(teacher_dir / "teacher.pt")
        teacher_configs.append(_comparable_config(teacher_args))

        student_args = _load_args(
            train_source, "train_student", seed=args.seed, epochs=args.epochs
        )
        student_args.split_path = str(split_path)
        student_args.work_dir = str(student_dir)
        student_args.student_ckpt = str(student_dir / "student.pt")
        student_args.teacher_ckpt = str(teacher_dir / "teacher.pt")
        student_configs.append(_comparable_config(student_args))

        discover_args = _load_args(
            discover_source, "discover", seed=args.seed, epochs=args.epochs
        )
        discover_args.split_path = str(split_path)
        discover_args.work_dir = str(discover_dir)
        discover_args.student_ckpt = str(student_dir / "student.pt")
        discover_configs.append(_comparable_config(discover_args))

    if teacher_configs[0] != teacher_configs[1]:
        raise AssertionError("Non-split teacher arguments differ between arms")
    if student_configs[0] != student_configs[1]:
        raise AssertionError("Non-split student arguments differ between arms")
    if discover_configs[0] != discover_configs[1]:
        raise AssertionError("Non-split discovery arguments differ between arms")

    summary = {
        "question": "How sensitive are known/unknown results to CIFAR-100 split semantics?",
        "single_changed_factor": "known_novel_split_protocol_with_protocol_matched_teacher",
        "estimand": "end-to-end protocol sensitivity under matched teacher/student recipes; not split-only causal effect",
        "seed": int(args.seed),
        "teacher_epochs": int(args.teacher_epochs),
        "epochs": int(args.epochs),
        "test_labels_used_for_training_selection_or_calibration": False,
        "source_teacher_config": str(teacher_source),
        "source_train_config": str(train_source),
        "source_discover_config": str(discover_source),
        "arms": {},
    }

    for protocol in PROTOCOLS:
        split_path, teacher_dir, _, _ = output_dirs[protocol]
        teacher_args = _load_args(
            teacher_source,
            "train_teacher",
            seed=args.seed,
            epochs=args.teacher_epochs,
        )
        teacher_args.split_path = str(split_path)
        teacher_args.work_dir = str(teacher_dir)
        teacher_args.teacher_ckpt = str(teacher_dir / "teacher.pt")
        print(f"[teacher] protocol={protocol} output={teacher_dir}", flush=True)
        fit_teacher(teacher_args)
        _assert_checkpoint_split_compatible(str(teacher_dir / "teacher.pt"), split_path)

    for protocol in PROTOCOLS:
        split_path, teacher_dir, train_dir, discover_dir = output_dirs[protocol]

        train_args = _load_args(
            train_source, "train_student", seed=args.seed, epochs=args.epochs
        )
        train_args.split_path = str(split_path)
        train_args.work_dir = str(train_dir)
        train_args.student_ckpt = str(train_dir / "student.pt")
        train_args.teacher_ckpt = str(teacher_dir / "teacher.pt")

        print(f"[train] protocol={protocol} output={train_dir}", flush=True)
        fit_student(train_args)

        discover_args = _load_args(
            discover_source, "discover", seed=args.seed, epochs=args.epochs
        )
        discover_args.split_path = str(split_path)
        discover_args.work_dir = str(discover_dir)
        discover_args.student_ckpt = str(train_dir / "student.pt")
        discover(discover_args)

        report_path = discover_dir / "discovery_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary["arms"][protocol] = {
            "split": _protocol_metadata(split_path),
            "train_dir": str(train_dir),
            "discover_dir": str(discover_dir),
            "metrics": {key: report.get(key) for key in METRIC_KEYS},
            "score_distribution": report.get("score_distribution", {}),
            "threshold_policy": report.get("threshold_policy"),
            "target_known_coverage": report.get("target_known_coverage"),
            "calibration_known_accept_rate": report.get(
                "calibration_known_accept_rate"
            ),
        }

    summary["non_split_teacher_student_and_discovery_config_match"] = True
    output_path = ROOT / "analysis" / f"{args.output_prefix}_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
