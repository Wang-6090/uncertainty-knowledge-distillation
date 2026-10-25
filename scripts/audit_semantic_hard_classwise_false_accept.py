"""Post-hoc classwise false-accept audit for frozen semantic-hard runs.

This diagnostic must not be used to select checkpoints, tune thresholds,
configure training, or derive class-specific weights. It only summarizes the
already-completed open-set evaluation outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def audit_classwise_false_accept(
    run_root: Path,
    split_root: Path,
    seeds: tuple[int, ...] = (2045, 2046),
    arms: tuple[str, ...] = ("shared_backbone", "random_backbone"),
    run_tag: str = "rejection_encoder_same_arch_init",
) -> dict:
    """Summarize per-novel-class and coarse-class false-accept rates."""
    per_run: dict[str, dict] = {}
    class_rates: dict[tuple[int, str], dict[int, float]] = {}
    expected_novel: dict[int, set[int]] = {}

    for seed in seeds:
        split_path = split_root / f"splits_cifar100_semantic_hard_seed{seed}" / "cifar100_60_40_semantic_hard.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        novel_classes = {int(value) for value in split["novel_classes"]}
        expected_novel[seed] = novel_classes
        names = split["fine_class_names"]
        coarse_names = split["coarse_class_names"]
        fine_to_coarse = {int(key): int(value) for key, value in split["fine_to_coarse"].items()}

        for arm in arms:
            detail_path = run_root / f"{run_tag}_s{seed}_{arm}_discover" / "discovery_detail.json"
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
            raw_labels = np.asarray(detail["raw_labels"], dtype=np.int64)
            true_known = np.asarray(detail["true_known"], dtype=bool)
            pred_known = np.asarray(detail["pred_known"], dtype=bool)
            if not (raw_labels.ndim == true_known.ndim == pred_known.ndim == 1):
                raise ValueError(f"malformed per-example arrays in {detail_path}")
            if not (len(raw_labels) == len(true_known) == len(pred_known)):
                raise ValueError(f"misaligned per-example arrays in {detail_path}")

            class_rows = []
            for class_id in sorted(novel_classes):
                mask = (raw_labels == class_id) & ~true_known
                count = int(mask.sum())
                if count == 0:
                    raise ValueError(f"no novel test examples for class {class_id} in {detail_path}")
                accepted = int((pred_known & mask).sum())
                class_rows.append({
                    "class_id": class_id,
                    "fine_class": names[class_id],
                    "coarse_class": coarse_names[fine_to_coarse[class_id]],
                    "count": count,
                    "false_accepts": accepted,
                    "false_accept_rate": accepted / count,
                    "wilson_95_ci": wilson_interval(accepted, count),
                })
            novel_mask = ~true_known
            false_accepts = int((pred_known & novel_mask).sum())
            coarse_rows = []
            for coarse_id, coarse_name in enumerate(coarse_names):
                coarse_classes = {
                    class_id for class_id in novel_classes
                    if fine_to_coarse[class_id] == coarse_id
                }
                if not coarse_classes:
                    continue
                mask = np.isin(raw_labels, list(coarse_classes)) & novel_mask
                count = int(mask.sum())
                accepted = int((pred_known & mask).sum())
                coarse_rows.append({
                    "coarse_class": coarse_name,
                    "novel_fine_classes": sorted(coarse_classes),
                    "count": count,
                    "false_accepts": accepted,
                    "false_accept_rate": accepted / count,
                })

            key = f"seed_{seed}_{arm}"
            per_run[key] = {
                "seed": seed,
                "arm": arm,
                "novel_count": int(novel_mask.sum()),
                "false_accepts": false_accepts,
                "false_accept_rate": false_accepts / max(int(novel_mask.sum()), 1),
                "per_novel_class": class_rows,
                "per_coarse_class": coarse_rows,
            }
            class_rates[(seed, arm)] = {
                row["class_id"]: row["false_accept_rate"] for row in class_rows
            }

    cross_split = {}
    if len(seeds) >= 2:
        common_novel = sorted(set.intersection(*(expected_novel[seed] for seed in seeds)))
        for arm in arms:
            paired = [
                (class_rates[(seeds[0], arm)][class_id], class_rates[(seeds[1], arm)][class_id])
                for class_id in common_novel
            ]
            left = np.asarray([pair[0] for pair in paired], dtype=np.float64)
            right = np.asarray([pair[1] for pair in paired], dtype=np.float64)
            correlation = (
                float(np.corrcoef(left, right)[0, 1])
                if len(paired) >= 3 and np.std(left) > 0 and np.std(right) > 0
                else None
            )
            cross_split[arm] = {
                "common_novel_class_count": len(common_novel),
                "common_novel_classes": common_novel,
                "classwise_false_accept_rate_pearson_r": correlation,
                "interpretation_guard": "descriptive only; do not derive training weights or tune from these test outcomes",
            }

    paired_arms = {}
    if len(arms) == 2:
        left_arm, right_arm = arms
        for seed in seeds:
            common_novel = sorted(expected_novel[seed])
            deltas = {
                class_id: class_rates[(seed, right_arm)][class_id]
                - class_rates[(seed, left_arm)][class_id]
                for class_id in common_novel
            }
            values = np.asarray(list(deltas.values()), dtype=np.float64)
            paired_arms[f"seed_{seed}"] = {
                "left_arm": left_arm,
                "right_arm": right_arm,
                "classwise_delta_definition": f"{right_arm} minus {left_arm}; positive means more novel false accepts",
                "per_class_delta": [
                    {
                        "class_id": class_id,
                        "fine_class": next(
                            row["fine_class"]
                            for row in per_run[f"seed_{seed}_{left_arm}"]["per_novel_class"]
                            if row["class_id"] == class_id
                        ),
                        "delta_false_accept_rate": float(deltas[class_id]),
                    }
                    for class_id in common_novel
                ],
                "classes_improved_for_right_arm": int((values < 0).sum()),
                "classes_worsened_for_right_arm": int((values > 0).sum()),
                "classes_tied": int((values == 0).sum()),
                "median_classwise_delta": float(np.median(values)),
                "mean_classwise_delta": float(np.mean(values)),
                "interpretation_guard": "single-split test description only; not for class-specific tuning or training weights",
            }

    return {
        "protocol": {
            "purpose": "post-hoc description of frozen open-set test outputs",
            "seeds": list(seeds),
            "arms": list(arms),
            "selection_or_training_used_test_labels": False,
            "checkpoint_or_threshold_changed": False,
        },
        "runs": per_run,
        "cross_split_common_novel_classes": cross_split,
        "paired_arm_classwise_comparison": paired_arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--split-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="rejection_encoder_same_arch_init")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2045, 2046])
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["shared_backbone", "random_backbone"],
    )
    args = parser.parse_args()
    result = audit_classwise_false_accept(
        args.run_root,
        args.split_root,
        seeds=tuple(args.seeds),
        arms=tuple(args.arms),
        run_tag=args.tag,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
