"""把逐 epoch 快照还原成训练正常结束时会产出的 student.pt。

背景：``train_student`` 只在训练完全结束时写 ``student.pt``，中途崩溃（例如显卡
VIDEO_TDR_FAILURE 蓝屏）就什么都没有。加上 ``--save-epoch-checkpoints`` 后每个
epoch 会落一份 ``student_epoch_<n>.pt``（eval 模型快照），本脚本按训练日志里的
``checkpoint_selection`` 选出最佳 epoch，把它复制成 ``student.pt``，并写一份
``promote_report.json`` 说明来源，避免事后无法解释用的到底哪个 epoch。

这是崩溃后的补救路径，不是常规流程：常规流程直接跑满 epochs 由 train.py 自己选。

用法：

    python scripts/promote_epoch_checkpoints.py --run-root runs/team_b_rejection/formal_20261124
    python scripts/promote_epoch_checkpoints.py --run-root <dir> --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

EPOCH_LINE = re.compile(r"\[student\]\[(\d+)/(\d+)\]")
SELECTION = re.compile(r"checkpoint_selection=([-0-9.eE+]+)")

STUDENT_DIR = re.compile(r"^s(?P<seed>-?\d+)_(?P<mode>[a-z0-9_]+)_student$")


def parse_log(log_path: Path) -> tuple[dict[int, float], int | None]:
    """Return {epoch: selection_score} and the total epoch budget seen in the log."""
    scores: dict[int, float] = {}
    total: int | None = None
    if not log_path.is_file():
        return scores, total
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "[student][" not in line:
                continue
            epoch_match = EPOCH_LINE.search(line)
            score_match = SELECTION.search(line)
            if not epoch_match or not score_match:
                continue
            scores[int(epoch_match.group(1))] = float(score_match.group(1))
            total = int(epoch_match.group(2))
    return scores, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing student.pt")
    args = parser.parse_args()

    run_root = args.run_root
    if not run_root.is_dir():
        raise SystemExit(f"run root not found: {run_root}")

    promoted: list[dict] = []
    missing: list[dict] = []

    for student_dir in sorted(run_root.iterdir()):
        if not student_dir.is_dir() or not STUDENT_DIR.match(student_dir.name):
            continue
        match = STUDENT_DIR.match(student_dir.name)
        seed, mode = int(match.group("seed")), match.group("mode")
        target = student_dir / "student.pt"
        scores, total = parse_log(student_dir / "train.log")
        checkpoints = {
            int(path.stem.replace("student_epoch_", "")): path
            for path in student_dir.glob("student_epoch_*.pt")
        }

        if not checkpoints:
            missing.append({"config": student_dir.name, "seed": seed, "mode": mode,
                            "reason": "no epoch checkpoints"})
            continue
        if target.exists() and not args.force:
            promoted.append({"config": student_dir.name, "seed": seed, "mode": mode,
                             "skipped": "student.pt already exists"})
            continue

        available = sorted(checkpoints)
        # Prefer the best selection score; fall back to the newest checkpoint.
        scored = {epoch: score for epoch, score in scores.items() if epoch in checkpoints}
        if scored:
            best_epoch, best_score = max(scored.items(), key=lambda item: item[1])
            selection_reason = "max checkpoint_selection"
        else:
            best_epoch, best_score = available[-1], None
            selection_reason = "latest checkpoint (no selection scores in log)"

        record = {
            "config": student_dir.name,
            "seed": seed,
            "mode": mode,
            "source_checkpoint": checkpoints[best_epoch].name,
            "source_epoch": best_epoch,
            "epochs_logged": len(scores),
            "epoch_budget_in_log": total,
            "checkpoints_available": available,
            "selection_reason": selection_reason,
            "checkpoint_selection": best_score,
            "all_selection_scores": {str(k): v for k, v in sorted(scores.items())},
            "reconstructed": True,
            "note": (
                "student.pt was reconstructed from an epoch snapshot because training "
                "was interrupted before train.py could write its own end-of-run checkpoint."
            ),
        }
        if not args.dry_run:
            shutil.copy2(checkpoints[best_epoch], target)
            (student_dir / "promote_report.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        promoted.append(record)
        print(
            f"{student_dir.name}: epoch {best_epoch}/{total} "
            f"(selection={best_score}) -> student.pt"
        )

    print(json.dumps({"promoted": promoted, "missing": missing},
                     indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
