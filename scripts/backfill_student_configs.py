"""为中断过的学生训练目录补齐 config.json（第 9 节统一记录格式）。

背景：``train_student`` 只在训练结束时写 ``config.json`` / ``train_history.json``。
本轮因为显卡 TDR 蓝屏中断，五个学生目录是从 epoch 快照还原出来的，只有
``command.txt``。本脚本用同一批次里正常结束的目录作为模板，按 ``command.txt``
改正 seed / 模式 / 路径 / epoch 数，补一份 ``config.json``，并显式标注
``reconstructed: true``，避免事后被误当成原始运行记录。

``train_history.json`` 不补：那是训练过程的原始统计，伪造比缺失更糟；
每个 epoch 的选择分在 ``promote_report.json`` 与 ``train.log`` 里都有。

用法：
    python scripts/backfill_student_configs.py --run-root runs/team_b_rejection/formal_20261124
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STUDENT_DIR = re.compile(r"^s(?P<seed>-?\d+)_(?P<mode>[a-z0-9_]+)_student$")
ARG = re.compile(r"--(?P<key>[a-z0-9-]+)(?:[ =](?P<value>[^ ]+))?")


def parse_command(path: Path) -> dict[str, str | bool]:
    """Very small --key value parser; handles the flags this runner emits."""
    values: dict[str, str | bool] = {}
    if not path.is_file():
        return values
    text = path.read_text(encoding="utf-8").strip()
    tokens = text.split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                key, _, value = key.partition("=")
                values[key] = value
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                values[key] = tokens[index + 1]
                index += 1
            else:
                values[key] = True
        index += 1
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=None,
                        help="a completed run directory to copy the schema from; "
                             "defaults to the first student dir that has config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="rewrite config.json even if it already exists")
    args = parser.parse_args()

    run_root = args.run_root
    student_dirs = sorted(
        path for path in run_root.iterdir()
        if path.is_dir() and STUDENT_DIR.match(path.name)
    )
    template_path = args.template
    if template_path is None:
        for candidate in student_dirs:
            if (candidate / "config.json").is_file():
                template_path = candidate / "config.json"
                break
    if template_path is None or not Path(template_path).is_file():
        raise SystemExit("no template config.json found; pass --template")
    template = json.loads(Path(template_path).read_text(encoding="utf-8"))

    written = []
    for student_dir in student_dirs:
        config_path = student_dir / "config.json"
        if config_path.is_file() and not args.force:
            continue
        match = STUDENT_DIR.match(student_dir.name)
        seed, mode = int(match.group("seed")), match.group("mode")
        command = parse_command(student_dir / "command.txt")
        config = dict(template)
        config.update(
            {
                "command": "train_student",
                "seed": seed,
                "work_dir": str(student_dir),
                "independent_rejection_encoder": True,
                "independent_rejection_encoder_init": mode,
                "epochs": int(command.get("epochs", template.get("epochs", 0))),
                "reconstructed": True,
                "reconstruction_note": (
                    "config.json was rebuilt from command.txt after the machine crashed "
                    "with VIDEO_TDR_FAILURE; the original file is only written when "
                    "train_student finishes. train_history.json is intentionally absent."
                ),
            }
        )
        # Record what actually happened, not only what was requested.
        promote_path = student_dir / "promote_report.json"
        if promote_path.is_file():
            promote = json.loads(promote_path.read_text(encoding="utf-8"))
            config["selected_epoch"] = promote.get("source_epoch")
            config["checkpoints_available"] = promote.get("checkpoints_available")
            config["effective_epochs_trained"] = (
                max(promote.get("checkpoints_available") or [0]) or None
            )
            config["interrupted"] = True
            config["interruption_note"] = (
                "training stopped before the requested epoch budget; student.pt was "
                "reconstructed from the best epoch snapshot"
            )
        if "teacher-ckpt" in command:
            config["teacher_ckpt"] = command["teacher-ckpt"]
        if "image-size" in command:
            config["image_size"] = int(str(command["image-size"]))
        if "mc-samples" in command:
            config["mc_samples"] = int(str(command["mc-samples"]))
        payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        if not args.dry_run:
            config_path.write_text(payload, encoding="utf-8")
            config_path.with_name("train_student_config.json").write_text(payload, encoding="utf-8")
        written.append(student_dir.name)
        print(f"backfilled {student_dir.name} (seed={seed}, mode={mode}, "
              f"epochs={config['epochs']})")

    if not written:
        print("nothing to do: every student directory already has config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
