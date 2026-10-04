"""Generate reproducible CIFAR-100 open-set class protocols.

The script reads the official CIFAR-100 coarse labels from the downloaded
``cifar-100-python`` archive. It creates three protocols with the same 60/40
class count:

* ``random``: a seeded random baseline;
* ``semantic_hard``: every coarse superclass contributes 3 known and 2 novel
  fine classes, so known and novel classes are semantically close;
* ``semantic_isolated``: 12 complete coarse superclasses are known and the
  remaining 8 are novel, producing a coarse-semantic separation protocol.

The generated JSON files are directly accepted by ``train.py --split-path``.
No image data or model weights are written by this script.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_cifar100_taxonomy(data_root: str | Path) -> tuple[list[str], list[str], dict[int, int]]:
    """Read fine/coarse names and the official fine->coarse mapping."""
    root = Path(data_root) / "cifar-100-python"
    train_file = root / "train"
    meta_file = root / "meta"
    if not train_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"CIFAR-100 raw files not found under {root}. "
            "Run train.py inspect_data --download first."
        )

    with train_file.open("rb") as handle:
        train = pickle.load(handle, encoding="bytes")
    with meta_file.open("rb") as handle:
        meta = pickle.load(handle, encoding="bytes")

    fine_names = [_decode(name) for name in meta[b"fine_label_names"]]
    coarse_names = [_decode(name) for name in meta[b"coarse_label_names"]]
    mapping: dict[int, int] = {}
    for fine_id, coarse_id in zip(train[b"fine_labels"], train[b"coarse_labels"]):
        fine_id = int(fine_id)
        coarse_id = int(coarse_id)
        previous = mapping.setdefault(fine_id, coarse_id)
        if previous != coarse_id:
            raise ValueError(f"Fine class {fine_id} maps to multiple coarse classes")
    if set(mapping) != set(range(len(fine_names))):
        raise ValueError("CIFAR-100 taxonomy is incomplete")
    return fine_names, coarse_names, mapping


def build_protocols(
    fine_names: list[str],
    coarse_names: list[str],
    fine_to_coarse: dict[int, int],
    seed: int,
    num_known: int = 60,
) -> dict[str, dict[str, Any]]:
    if len(fine_names) != 100 or len(coarse_names) != 20 or num_known != 60:
        raise ValueError("This protocol generator currently targets CIFAR-100 60/40")

    all_classes = list(range(len(fine_names)))
    rng = random.Random(seed)
    random_known = sorted(rng.sample(all_classes, num_known))

    by_coarse: dict[int, list[int]] = {coarse_id: [] for coarse_id in range(20)}
    for fine_id in all_classes:
        by_coarse[fine_to_coarse[fine_id]].append(fine_id)
    for fine_ids in by_coarse.values():
        fine_ids.sort()

    hard_known = sorted(
        fine_id
        for fine_ids in by_coarse.values()
        for fine_id in fine_ids[:3]
    )
    isolated_coarse = sorted(rng.sample(list(range(20)), 12))
    isolated_known = sorted(
        fine_id
        for coarse_id in isolated_coarse
        for fine_id in by_coarse[coarse_id]
    )

    protocols = {
        "random": {"known_classes": random_known, "metadata": {"seed": seed}},
        "semantic_hard": {
            "known_classes": hard_known,
            "metadata": {"known_per_coarse": 3, "novel_per_coarse": 2},
        },
        "semantic_isolated": {
            "known_classes": isolated_known,
            "metadata": {
                "known_coarse_classes": isolated_coarse,
                "novel_coarse_classes": sorted(set(range(20)) - set(isolated_coarse)),
            },
        },
    }
    for name, protocol in protocols.items():
        known = protocol["known_classes"]
        novel = sorted(set(all_classes) - set(known))
        protocol["novel_classes"] = novel
        protocol["protocol"] = name
        protocol["fine_class_names"] = fine_names
        protocol["coarse_class_names"] = coarse_names
        protocol["fine_to_coarse"] = {str(k): int(v) for k, v in fine_to_coarse.items()}
    return protocols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./splits_cifar100_protocols")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fine_names, coarse_names, fine_to_coarse = read_cifar100_taxonomy(args.data_root)
    protocols = build_protocols(fine_names, coarse_names, fine_to_coarse, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, protocol in protocols.items():
        path = output_dir / f"cifar100_60_40_{name}.json"
        path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        print(f"  known={len(protocol['known_classes'])} novel={len(protocol['novel_classes'])}")


if __name__ == "__main__":
    main()
