"""Validate the saved Team A reproducibility configurations without training.

The repository stores JSON files produced by previous runs.  They are useful
as experiment specifications, but they are not accepted directly by
``train.py`` as a ``--config`` argument.  This script checks the parts that
must be correct before writing a command wrapper or starting a smoke test:

* stage and required fields;
* relative-path resolution from the repository root and from another cwd;
* teacher -> student -> discovery checkpoint dependencies;
* whether a path is an existing reusable artifact or an expected future
  output of the reproducibility run.

It is intentionally read-only with respect to runs and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "reproducibility" / "team_a"


@dataclass(frozen=True)
class StageSpec:
    name: str
    filename: str
    required: tuple[str, ...]
    checkpoint_fields: tuple[str, ...]


STAGES = (
    StageSpec(
        "teacher",
        "teacher.json",
        ("command", "dataset", "data_root", "split_path", "num_known", "seed", "work_dir"),
        ("teacher_ckpt",),
    ),
    StageSpec(
        "student",
        "student.json",
        (
            "command",
            "dataset",
            "data_root",
            "split_path",
            "num_known",
            "seed",
            "work_dir",
            "teacher_ckpt",
            "student_ckpt",
        ),
        ("teacher_ckpt", "student_ckpt"),
    ),
    StageSpec(
        "discover",
        "discover.json",
        (
            "command",
            "dataset",
            "data_root",
            "split_path",
            "num_known",
            "seed",
            "work_dir",
            "student_ckpt",
        ),
        ("student_ckpt",),
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def resolve(value: Any, base: Path) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def path_status(value: Any, *, root: Path, alternate_cwd: Path) -> dict[str, Any]:
    if value in (None, ""):
        return {
            "value": value,
            "relative": False,
            "root_resolution": None,
            "root_exists": False,
            "alternate_cwd_resolution": None,
            "alternate_cwd_exists": False,
        }
    raw = Path(str(value))
    root_path = resolve(value, root)
    alternate_path = resolve(value, alternate_cwd)
    return {
        "value": str(value),
        "relative": not raw.is_absolute(),
        "root_resolution": str(root_path),
        "root_exists": bool(root_path and root_path.exists()),
        "alternate_cwd_resolution": str(alternate_path),
        "alternate_cwd_exists": bool(alternate_path and alternate_path.exists()),
    }


def classify_artifact(path: Path | None, *, is_output: bool) -> str:
    if path is None:
        return "未提供"
    if path.is_file():
        return "已存在文件，可复用"
    if path.is_dir():
        return "已存在目录，需检查内部产物"
    if is_output:
        return "尚未生成的预期输出"
    return "缺失依赖"


def validate_stage(spec: StageSpec, config: dict[str, Any], alternate_cwd: Path) -> dict[str, Any]:
    missing = [key for key in spec.required if key not in config]
    command_ok = config.get("command") == (
        "train_teacher" if spec.name == "teacher" else
        "train_student" if spec.name == "student" else "discover"
    )
    paths: dict[str, Any] = {}
    for key in ("data_root", "split_path", "work_dir", *spec.checkpoint_fields):
        if key in config:
            status = path_status(config[key], root=ROOT, alternate_cwd=alternate_cwd)
            resolved = resolve(config[key], ROOT)
            status["artifact_class"] = classify_artifact(
                resolved,
                is_output=key == "work_dir" or key == "teacher_ckpt" and spec.name == "teacher" or key == "student_ckpt" and spec.name == "student",
            )
            paths[key] = status

    return {
        "stage": spec.name,
        "config_file": str(CONFIG_DIR / spec.filename),
        "checkpoint_fields": list(spec.checkpoint_fields),
        "command": config.get("command"),
        "command_matches_stage": command_ok,
        "required_fields_missing": missing,
        "data_protocol": {
            "dataset": config.get("dataset"),
            "num_known": config.get("num_known"),
            "seed": config.get("seed"),
            "transform_profile": config.get("transform_profile"),
            "discovery_pool_mode": config.get("discovery_pool_mode"),
            "open_val_ratio": config.get("open_val_ratio"),
            "calibration_ratio": config.get("calibration_ratio"),
        },
        "paths": paths,
    }


def build_report(alternate_cwd: Path) -> dict[str, Any]:
    configs = {spec.name: load_json(CONFIG_DIR / spec.filename) for spec in STAGES}
    stages = [validate_stage(spec, configs[spec.name], alternate_cwd) for spec in STAGES]

    teacher_ckpt = resolve(configs["teacher"].get("teacher_ckpt"), ROOT)
    student_teacher_ckpt = resolve(configs["student"].get("teacher_ckpt"), ROOT)
    student_ckpt = resolve(configs["student"].get("student_ckpt"), ROOT)
    discover_student_ckpt = resolve(configs["discover"].get("student_ckpt"), ROOT)

    dependency_checks = [
        {
            "name": "student uses teacher output path",
            "same_path": teacher_ckpt == student_teacher_ckpt,
            "teacher_path": str(teacher_ckpt) if teacher_ckpt else None,
            "student_reference": str(student_teacher_ckpt) if student_teacher_ckpt else None,
            "target_exists": bool(teacher_ckpt and teacher_ckpt.is_file()),
        },
        {
            "name": "discover uses student output path",
            "same_path": student_ckpt == discover_student_ckpt,
            "student_path": str(student_ckpt) if student_ckpt else None,
            "discover_reference": str(discover_student_ckpt) if discover_student_ckpt else None,
            "target_exists": bool(student_ckpt and student_ckpt.is_file()),
        },
    ]

    root_ok = all(
        stage["paths"].get(key, {}).get("root_exists", False)
        for stage in stages
        for key in ("data_root", "split_path")
    )
    alternate_relative_failures = [
        {
            "stage": stage["stage"],
            "key": key,
            "value": status["value"],
            "root_resolution": status["root_resolution"],
            "alternate_cwd_resolution": status["alternate_cwd_resolution"],
        }
        for stage in stages
        for key, status in stage["paths"].items()
        if status["relative"] and status["root_exists"] and not status["alternate_cwd_exists"]
    ]

    return {
        "repository_root": str(ROOT),
        "config_directory": str(CONFIG_DIR),
        "alternate_cwd": str(alternate_cwd),
        "read_only": True,
        "stages": stages,
        "dependency_checks": dependency_checks,
        "summary": {
            "all_required_fields_present": all(not stage["required_fields_missing"] for stage in stages),
            "all_commands_match_stage": all(stage["command_matches_stage"] for stage in stages),
            "data_and_split_exist_from_root": root_ok,
            "alternate_cwd_relative_path_failures": len(alternate_relative_failures),
            "teacher_output_exists": bool(teacher_ckpt and teacher_ckpt.is_file()),
            "student_output_exists": bool(student_ckpt and student_ckpt.is_file()),
            "discover_input_exists": bool(discover_student_ckpt and discover_student_ckpt.is_file()),
            "can_start_from_saved_team_a_paths": bool(
                root_ok
                and all(not stage["required_fields_missing"] for stage in stages)
                and all(item["same_path"] for item in dependency_checks)
                and bool(teacher_ckpt and teacher_ckpt.is_file())
                and bool(student_ckpt and student_ckpt.is_file())
            ),
        },
        "alternate_relative_path_failures": alternate_relative_failures,
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Team A 复现配置预检",
        "",
        "本检查只读取 `configs/reproducibility/team_a/` 和路径元信息，不启动训练、不修改 runs、不使用测试标签选择参数。",
        "",
        "## 结论",
        "",
        f"- 必需字段完整：{'是' if summary['all_required_fields_present'] else '否'}。",
        f"- 三个阶段的 command 与阶段名称一致：{'是' if summary['all_commands_match_stage'] else '否'}。",
        f"- 从仓库根目录解析 data_root 和 split_path：{'正常' if summary['data_and_split_exist_from_root'] else '失败'}。",
        f"- 从其他工作目录启动时会失效的相对路径：{summary['alternate_cwd_relative_path_failures']} 条。",
        f"- 配置声明的 teacher/student checkpoint 已存在：{'是' if summary['teacher_output_exists'] and summary['student_output_exists'] else '否'}。",
        "",
        "当前 JSON 可以作为复现参数的保存记录，但不能直接写成 `train.py --config ...`；正式复现仍需要把 JSON 字段转换为对应的子命令参数，并在仓库根目录或显式指定根目录的入口下执行。",
        "",
        "## 阶段与路径",
        "",
        "| 阶段 | 数据路径 | 划分文件 | 工作目录 | 关键 checkpoint |",
        "|---|---|---|---|---|",
    ]
    for stage in report["stages"]:
        paths = stage["paths"]
        def cell(key: str) -> str:
            item = paths.get(key)
            if not item:
                return "未提供"
            return f"{item['value']}（{item['artifact_class']}）"
        lines.append(
            f"| {stage['stage']} | {cell('data_root')} | {cell('split_path')} | {cell('work_dir')} | "
            f"{'; '.join(f'{key}={cell(key)}' for key in stage['checkpoint_fields']) or '无'} |"
        )
    lines.extend(["", "## 阶段依赖", ""])
    for item in report["dependency_checks"]:
        lines.append(
            f"- {item['name']}：路径一致性={'是' if item['same_path'] else '否'}，"
            f"目标存在={'是' if item['target_exists'] else '否'}。"
        )
    lines.extend(["", "## 需要处理的复现入口问题", ""])
    lines.append("1. `train.py` 当前只有 `train_teacher`、`train_student`、`discover` 等子命令，没有通用 `--config` 参数；三份 JSON 不能直接作为命令输入。")
    lines.append("2. 配置中的相对路径以当前工作目录为基准。仓库根目录运行时 data 和 split 可解析；从其他目录运行时会出现路径失效，因此入口脚本需要显式固定仓库根目录。")
    lines.append("3. `runs/reproducibility/team_a/...` 是配置声明的预期输出路径；本次预检不把已有的 `runs/team_a_rejection_init_s2074_*` 历史结果冒充为这些路径下的复现产物。")
    lines.extend(["", "## 下一步", "", "先增加独立的 JSON 转命令 dry-run 入口，再用现有 checkpoint 在临时目录执行小规模 smoke test；通过后才考虑正式复现。"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alternate-cwd", type=Path, default=Path("D:/"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    alternate_cwd = args.alternate_cwd.resolve()
    report = build_report(alternate_cwd)
    output_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output_md = markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output_json, encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(output_md, encoding="utf-8")
    print(output_md, end="")


if __name__ == "__main__":
    main()
