"""Convert a saved reproducibility JSON into a checked train.py command.

This is intentionally a command builder first.  ``--dry-run`` never trains and
prints the exact argv that would be executed.  The script treats saved-run
metadata and post-run measurements differently from actual parser parameters;
they are reported instead of being silently discarded.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "reproducibility" / "team_a"


def ensure_repo_import_path() -> None:
    """Make repository modules importable when this script is run by path.

    Python puts ``scripts/`` (the script's directory) at ``sys.path[0]``.
    Changing the working directory does not update that entry, so importing
    the repository's ``train.py`` would otherwise fail outside an editable
    install.
    """
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

STAGE_COMMAND = {
    "teacher": "train_teacher",
    "student": "train_student",
    "discover": "discover",
}

# Fields written by train.py after a run, or fields describing the saved run.
# They must be visible in the report but must never be passed back as CLI args.
SAVED_ONLY_FIELDS = {
    "command",
    "transform_profile_requested",
    "best_epoch",
    "best_val_known_acc",
    "best_checkpoint_selection_score",
    "best_open_checkpoint_stats",
    "prototype_init_stats",
    "prototype_refresh_history",
    "discovery_pool_semantics",
    "discovery_pool_known_proportion",
    "discovery_uncertainty_known_prior_resolved",
    "actual_discovery_pool_size",
    "actual_discovery_pool_known_fraction",
}

# The saved configs use descriptive names for two parser parameters.  These
# aliases are explicit so a spelling change cannot silently change the protocol.
ALIASES = {
    "matched_discovery_pool_size": "matched_pool_size",
    "matched_discovery_known_prior": "matched_known_prior",
}

# Some historical records contain protocol fields on stages where the current
# command does not expose them.  Preserve them in the report as inactive rather
# than passing them to a different stage by accident.
STAGE_INACTIVE_FIELDS = {
    "teacher": {"open_val_ratio"},
}


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def parser_dests(stage: str) -> set[str]:
    # Import only after fixing the cwd: train.py resolves its default cache from
    # the current directory at import time.
    os.chdir(ROOT)
    ensure_repo_import_path()
    import train  # pylint: disable=import-outside-toplevel

    parser = train.build_parser()
    command_parser = parser._subparsers._group_actions[0].choices[STAGE_COMMAND[stage]]
    return {action.dest for action in command_parser._actions}


def cli_name(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def value_to_args(dest: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [cli_name(dest)] if value else []
    if isinstance(value, (str, int, float)):
        return [cli_name(dest), str(value)]
    raise TypeError(f"unsupported CLI value for {dest}: {value!r}")


def resolve_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def validate_paths(stage: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for field in ("data_root", "split_path", "teacher_ckpt", "student_ckpt"):
        if field not in config:
            continue
        resolved = resolve_path(config[field])
        is_output = (
            (stage == "teacher" and field == "teacher_ckpt")
            or (stage == "student" and field == "student_ckpt")
        )
        checks.append(
            {
                "field": field,
                "value": config[field],
                "resolved": str(resolved) if resolved else None,
                "exists": bool(resolved and resolved.exists()),
                "required_existing_input": not is_output,
                "expected_output": is_output,
            }
        )
    return checks


def build(stage: str, config: dict[str, Any]) -> dict[str, Any]:
    if stage not in STAGE_COMMAND:
        raise ValueError(f"unknown stage: {stage}")
    expected_command = STAGE_COMMAND[stage]
    if config.get("command") != expected_command:
        raise ValueError(
            f"config command={config.get('command')!r} does not match stage {stage!r}"
        )

    dests = parser_dests(stage)
    args: list[str] = [sys.executable, "train.py", expected_command]
    passed: dict[str, str] = {}
    aliases_used: dict[str, str] = {}
    saved_only: dict[str, Any] = {}
    inactive: dict[str, Any] = {}
    unknown: dict[str, Any] = {}

    for raw_key, value in config.items():
        if raw_key in SAVED_ONLY_FIELDS:
            saved_only[raw_key] = value
            continue
        if raw_key in STAGE_INACTIVE_FIELDS.get(stage, set()):
            inactive[raw_key] = value
            continue
        key = ALIASES.get(raw_key, raw_key)
        if key != raw_key:
            aliases_used[raw_key] = key
        if key not in dests:
            unknown[raw_key] = value
            continue
        if value is None:
            # None means the saved run used the parser default.  Omitting it is
            # safer than serializing a literal "None".
            continue
        args.extend(value_to_args(key, value))
        passed[key] = value

    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"{stage} contains fields not accepted by train.py: {names}. "
            "Add an explicit alias or saved-only classification before running."
        )

    # Resolve all path arguments to the repository root so execution does not
    # depend on the caller's current directory.
    path_dests = {"data_root", "split_path", "work_dir", "teacher_ckpt", "student_ckpt", "outlier_data_root"}
    normalized: list[str] = [args[0], str(ROOT / "train.py"), args[2]]
    i = 3
    while i < len(args):
        token = args[i]
        normalized.append(token)
        dest = token[2:].replace("-", "_") if token.startswith("--") else None
        if dest in path_dests and i + 1 < len(args) and not args[i + 1].startswith("--"):
            raw = Path(args[i + 1])
            normalized.append(str(raw if raw.is_absolute() else (ROOT / raw).resolve()))
            i += 2
        else:
            i += 1

    return {
        "stage": stage,
        "command": normalized,
        "command_shell": shlex.join(normalized),
        "passed_fields": passed,
        "aliases_used": aliases_used,
        "saved_only_fields": saved_only,
        "inactive_fields": inactive,
        "path_checks": validate_paths(stage, config),
    }


def validate_generated_argv(command: list[str]) -> dict[str, Any]:
    """Run argparse only, without entering train.py's main dispatch."""
    os.chdir(ROOT)
    ensure_repo_import_path()
    import train  # pylint: disable=import-outside-toplevel

    # ``command`` is ``[python, train.py, subcommand, ...]``; train.parse_args
    # expects the same argv slice that would follow ``train.py``.
    parsed = train.parse_args(command[2:])
    return {
        "accepted": True,
        "resolved_command": parsed.command,
        "transform_profile": parsed.transform_profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_COMMAND), required=True)
    parser.add_argument("--dry-run", action="store_true", help="print only; never execute train.py")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path.resolve())
    report = build(args.stage, config)

    report["parser_validation"] = validate_generated_argv(report["command"])
    required_missing = [
        item["field"]
        for item in report["path_checks"]
        if item["required_existing_input"] and not item["exists"]
    ]
    report["missing_existing_inputs"] = required_missing
    if required_missing and not args.dry_run:
        raise FileNotFoundError(
            f"{args.stage} has missing existing inputs: {', '.join(required_missing)}"
        )

    print(f"stage: {report['stage']}")
    print(f"command: {report['command_shell']}")
    print(f"aliases: {json.dumps(report['aliases_used'], ensure_ascii=False)}")
    print(f"saved-only fields: {', '.join(sorted(report['saved_only_fields'])) or 'none'}")
    print(f"inactive fields: {', '.join(sorted(report['inactive_fields'])) or 'none'}")
    print("path checks:")
    for item in report["path_checks"]:
        state = "exists" if item["exists"] else ("expected output" if item["expected_output"] else "MISSING")
        print(f"  - {item['field']}: {state} -> {item['resolved']}")
    if required_missing:
        print(f"missing existing inputs (execution blocked): {', '.join(required_missing)}")

    if args.output_json:
        output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.dry_run:
        if any(item["expected_output"] and item["exists"] for item in report["path_checks"]):
            raise FileExistsError(
                "refusing to run because a declared stage output already exists; "
                "choose a new work_dir/checkpoint path explicitly"
            )
        completed = subprocess.run(report["command"], cwd=ROOT, check=False)
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
