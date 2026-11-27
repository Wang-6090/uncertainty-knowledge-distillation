"""Summarize known-validation calibration stability across frozen fusion splits."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("test_or_mixed_pool_labels_used_for_decision") is not False:
        raise ValueError(f"label-use audit failed: {path}")
    return value


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2084, 2085, 2086, 2089])
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        path = args.analysis_dir / f"shared_rejection_fusion_calibration_s{seed}.json"
        value = read(path)
        if value["control_audit"]["bootstrap_repeats"] != value["fusion_audit"]["bootstrap_repeats"]:
            raise ValueError(f"control/fusion bootstrap repeat mismatch in {path}")
        if value["control_metadata"].get("contains_test_samples") is not False:
            raise ValueError(f"control export metadata failed: {path}")
        if value["fusion_metadata"].get("contains_test_samples") is not False:
            raise ValueError(f"fusion export metadata failed: {path}")
        c = value["control_audit"]["out_of_bag_known_acceptance"]
        f = value["fusion_audit"]["out_of_bag_known_acceptance"]
        rows.append(
            {
                "seed": seed,
                "sample_count": value["sample_count"],
                "control_oob": c,
                "fusion_oob": f,
                "control_threshold_std": value["control_audit"]["bootstrap_threshold"]["std"],
                "fusion_threshold_std": value["fusion_audit"]["bootstrap_threshold"]["std"],
                "paired_threshold_delta": value["paired_threshold_delta_fusion_minus_control"],
            }
        )

    summary = {
        "protocol": "known-validation-only calibration stability for fixed fusion",
        "seeds": args.seeds,
        "bootstrap_repeats_by_seed": {
            str(row["seed"]): read(
                args.analysis_dir / f"shared_rejection_fusion_calibration_s{row['seed']}.json"
            )["control_audit"]["bootstrap_repeats"]
            for row in rows
        },
        "test_or_mixed_pool_labels_used_for_decision": False,
        "rows": rows,
        "aggregate": {
            "sample_count": stats([float(row["sample_count"]) for row in rows]),
            "control_oob_mean": stats([row["control_oob"]["mean"] for row in rows]),
            "fusion_oob_mean": stats([row["fusion_oob"]["mean"] for row in rows]),
            "control_oob_interval_width": stats(
                [row["control_oob"]["q975"] - row["control_oob"]["q025"] for row in rows]
            ),
            "fusion_oob_interval_width": stats(
                [row["fusion_oob"]["q975"] - row["fusion_oob"]["q025"] for row in rows]
            ),
            "control_threshold_std": stats([row["control_threshold_std"] for row in rows]),
            "fusion_threshold_std": stats([row["fusion_threshold_std"] for row in rows]),
        },
        "interpretation_limit": (
            "Known-validation finite-sample calibration stability only; this does not "
            "estimate test coverage, unknown rejection, or cross-split performance."
        ),
    }
    out_json = args.analysis_dir / "shared_rejection_fusion_calibration_2084_2089_summary.json"
    out_md = args.analysis_dir / "shared_rejection_fusion_calibration_2084_2089_summary_20261024.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Fixed uncertainty-fusion calibration stability summary",
        "",
        "This is a known-validation-only bootstrap/OOB audit of frozen checkpoints.",
        "It does not estimate test coverage, unknown rejection, or cross-split performance.",
        "",
        "| seed | n | control OOB mean [q2.5, q97.5] | fusion OOB mean [q2.5, q97.5] |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        c, f = row["control_oob"], row["fusion_oob"]
        lines.append(
            f"| {row['seed']} | {row['sample_count']} | "
            f"{c['mean']:.4f} [{c['q025']:.4f}, {c['q975']:.4f}] | "
            f"{f['mean']:.4f} [{f['q025']:.4f}, {f['q975']:.4f}] |"
        )
    a = summary["aggregate"]
    lines += [
        "",
        "## Aggregate diagnostics",
        "",
        f"Mean OOB known acceptance: control `{a['control_oob_mean']['mean']:.4f}`, fusion `{a['fusion_oob_mean']['mean']:.4f}`.",
        f"Mean OOB interval width: control `{a['control_oob_interval_width']['mean']:.4f}`, fusion `{a['fusion_oob_interval_width']['mean']:.4f}`.",
        f"Mean within-score bootstrap threshold standard deviation: control `{a['control_threshold_std']['mean']:.4f}`, fusion `{a['fusion_threshold_std']['mean']:.4f}`.",
        "Absolute threshold scales are not compared across score modes; each score is assessed within its own scale.",
        "",
        "## Decision",
        "",
        "The OOB means remain close to the requested 95% validation coverage, but the intervals remain materially wide across all fresh splits. This is a calibration-risk finding, not evidence against the fusion performance result and not evidence that test coverage transfers reliably. The fusion remains an explicit opt-in candidate rather than the default.",
        "",
        "Machine records: `shared_rejection_fusion_calibration_s2084.json`, `s2085`, `s2086`, and `s2089`.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
