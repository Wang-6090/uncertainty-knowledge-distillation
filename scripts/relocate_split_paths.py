"""Move split references to the consolidated ``splits/`` directory.

This maintenance script is intentionally limited to UTF-8 text artifacts. It
does not touch checkpoints, NumPy arrays, images, Word documents, or Git
metadata. It is safe to run repeatedly because the replacements are
idempotent.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py",
    ".ps1",
    ".md",
    ".json",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".log",
    ".err",
}


def rewrite(text: str) -> str:
    # Keep these replacements textual and idempotent so the script can also
    # repair old commands/configs created before the directory consolidation.
    text = text.replace("splits_cifar100_protocols", "splits/protocols")
    # Replace the prefix as well as concrete ``seed2040`` names.  The prefix
    # form is needed for f-strings such as ``seed{seed}`` in experiment
    # runners, which cannot be handled by a digits-only regular expression.
    text = text.replace(
        "splits_cifar100_semantic_hard_seed",
        "splits/semantic_hard/seed",
    )
    text = re.sub(
        r"splits_cifar100_semantic_hard_seed([0-9]+)",
        r"splits/semantic_hard/seed\1",
        text,
    )
    text = text.replace("splits_cifar100_60_40.json", "splits/cifar100_60_40.json")
    text = text.replace("splits_toy_10_90.json", "splits/toy_10_90.json")
    text = re.sub(r"(?<![A-Za-z0-9_])splits\.json", "splits/default.json", text)
    return text


def main() -> None:
    changed: list[Path] = []
    skipped: list[Path] = []
    script_path = Path(__file__).resolve()
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == script_path:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            old = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped.append(path)
            continue
        new = rewrite(old)
        if new != old:
            path.write_text(new, encoding="utf-8", newline="")
            changed.append(path)
    print(f"updated_text_files={len(changed)}")
    for path in changed:
        print(path.relative_to(ROOT))
    print(f"skipped_non_utf8_files={len(skipped)}")


if __name__ == "__main__":
    main()
