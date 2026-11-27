"""Audit whether two saved training runs are a valid paired comparison.

The project has accumulated many experimental flags.  Comparing only the
intended flag is unsafe when an old command omitted newer options, because the
saved configuration then contains different defaults.  This small utility
prints every configuration difference and can optionally ignore bookkeeping
keys such as output paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_IGNORED = {
    "command",
    "work_dir",
    "student_ckpt",
    "teacher_ckpt",
    "best_epoch",
    "best_val_known_acc",
    "prototype_init_stats",
    "prototype_refresh_history",
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compare(left: dict[str, Any], right: dict[str, Any], ignored: set[str]) -> list[tuple[str, Any, Any]]:
    keys = sorted(set(left) | set(right))
    return [
        (key, left.get(key, "<MISSING>"), right.get(key, "<MISSING>"))
        for key in keys
        if key not in ignored and left.get(key, "<MISSING>") != right.get(key, "<MISSING>")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="first run directory or config JSON")
    parser.add_argument("right", type=Path, help="second run directory or config JSON")
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="additional configuration key to ignore; may be repeated",
    )
    args = parser.parse_args()

    def config_path(value: Path) -> Path:
        return value if value.suffix == ".json" else value / "train_student_config.json"

    left_path = config_path(args.left)
    right_path = config_path(args.right)
    left = _load(left_path)
    right = _load(right_path)
    differences = compare(left, right, DEFAULT_IGNORED | set(args.ignore))

    print(f"left:  {left_path}")
    print(f"right: {right_path}")
    if not differences:
        print("PAIR_STATUS: strict (no non-ignored configuration differences)")
        return 0

    print(f"PAIR_STATUS: non-strict ({len(differences)} configuration differences)")
    for key, left_value, right_value in differences:
        print(f"- {key}: left={_render(left_value)} | right={_render(right_value)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
