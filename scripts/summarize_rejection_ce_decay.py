"""Summarize fixed late rejection-CE decay paired result JSON files.

The script is intentionally read-only with respect to experiment artifacts:
it loads completed paired JSON files and writes a separate summary.  It never
uses test labels to select a schedule or a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = [
    ("AUROC", "auroc", False),
    ("AUPR", "aupr", False),
    ("FPR95", "fpr95", False),
    ("OSCR", "oscr", False),
    ("known acceptance", "known_accept_rate", False),
    ("accepted-known accuracy", "known_class_accuracy_after_accept", False),
    ("unknown rejection", "unknown_reject_rate", False),
    ("unknown below known q95", "overlap_unknown_below_known_q95", False),
    ("score KS", "ks_distance", False),
    ("candidate purity", "cluster_candidate_purity", False),
    ("unknown-only NMI", "cluster_unknown_only_nmi", False),
    ("unknown-only ARI", "cluster_unknown_only_ari", False),
    ("automatic-K absolute error", "cluster_k_abs_error", False),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_metric(report: dict[str, Any], key: str) -> float | None:
    if key in report:
        value = report[key]
    elif key == "overlap_unknown_below_known_q95":
        value = report.get("score_distribution", {}).get("overlap_unknown_below_known_q95")
    elif key == "ks_distance":
        value = report.get("score_distribution", {}).get("ks_distance")
    else:
        value = None
    return None if value is None else float(value)


def resolve_arms(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = result.get("arms", {})
    control = arms.get("static_control") or arms.get("control")
    treatment = arms.get("late_decay") or arms.get("treatment")
    if not control or not treatment:
        raise ValueError(f"cannot find paired arms in {result}")
    return control.get("metrics", control), treatment.get("metrics", treatment)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    options = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in options.results:
        result = read_json(path)
        control, treatment = resolve_arms(result)
        seed = int(result.get("seed", -1))
        row: dict[str, Any] = {"seed": seed, "source": str(path)}
        for label, key, _ in METRICS:
            control_value = get_metric(control, key)
            treatment_value = get_metric(treatment, key)
            delta = None
            if control_value is not None and treatment_value is not None:
                delta = treatment_value - control_value
            row[key] = {"control": control_value, "treatment": treatment_value, "delta": delta}
        rows.append(row)
    rows.sort(key=lambda row: row["seed"])

    means: dict[str, Any] = {"seed": "mean", "source": "paired arithmetic mean"}
    directions: dict[str, dict[str, int]] = {}
    for _, key, _ in METRICS:
        values = [row[key]["delta"] for row in rows if row[key]["delta"] is not None]
        means[key] = {
            "control": sum(row[key]["control"] for row in rows if row[key]["control"] is not None) / len(
                [row for row in rows if row[key]["control"] is not None]
            )
            if any(row[key]["control"] is not None for row in rows)
            else None,
            "treatment": sum(row[key]["treatment"] for row in rows if row[key]["treatment"] is not None) / len(
                [row for row in rows if row[key]["treatment"] is not None]
            )
            if any(row[key]["treatment"] is not None for row in rows)
            else None,
            "delta": sum(values) / len(values) if values else None,
        }
        directions[key] = {
            "positive": sum(value > 0 for value in values),
            "negative": sum(value < 0 for value in values),
            "zero": sum(value == 0 for value in values),
        }

    summary = {
        "protocol": "fixed late rejection-CE decay",
        "schedule": {"start_epoch": 3, "decay_epochs": 2, "final_scale": 0.25},
        "rows": rows,
        "mean": means,
        "direction_counts": directions,
    }
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    options.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Fixed late rejection-CE decay: machine-generated paired summary",
        "",
        "Schedule: epochs 1–2 scale `1.0`, epoch 3 scale `0.625`, epoch 4 scale `0.25`.",
        "Every row is treatment minus its paired static-control run; values below show raw control → treatment (delta).",
        "",
        "| seed | " + " | ".join(label for label, _, _ in METRICS) + " |",
        "|---:|" + "---:|" * len(METRICS),
    ]
    for row in rows + [means]:
        cells = []
        for _, key, _ in METRICS:
            item = row[key]
            cells.append(f"{fmt(item['control'])} → {fmt(item['treatment'])} ({item['delta']:+.6f})" if item["delta"] is not None else "NA")
        lines.append(f"| {row['seed']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Direction counts", ""])
    for label, key, _ in METRICS:
        counts = directions[key]
        lines.append(f"- {label}: positive {counts['positive']}, negative {counts['negative']}, zero {counts['zero']}")
    options.output_md.parent.mkdir(parents=True, exist_ok=True)
    options.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(options.output_json)
    print(options.output_md)


if __name__ == "__main__":
    main()
