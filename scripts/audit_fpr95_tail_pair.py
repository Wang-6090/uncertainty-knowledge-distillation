"""Audit FPR95 score tails for a paired detector comparison.

This is a retrospective diagnostic. Test labels are read only to explain the
already-computed metrics and are never used for training, calibration,
checkpoint selection, or method selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_detail(path: Path) -> tuple[np.ndarray, np.ndarray]:
    detail = json.loads(path.read_text(encoding="utf-8"))
    scores = np.asarray(detail["score"], dtype=np.float64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    if scores.ndim != 1 or true_known.shape != scores.shape:
        raise ValueError(f"invalid score/label shape in {path}")
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite scores in {path}")
    return scores, true_known


def quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float]:
    return {
        f"q{int(probability * 100):02d}": float(np.quantile(values, probability))
        for probability in probabilities
    }


def fpr95_prefix(scores: np.ndarray, true_known: np.ndarray) -> dict[str, float | int]:
    is_unknown = ~true_known
    order = np.argsort(scores)[::-1]
    cumulative_unknown = np.cumsum(is_unknown[order], dtype=np.int64)
    target = 0.95 * int(is_unknown.sum())
    reached = np.flatnonzero(cumulative_unknown >= target)
    index = int(reached[0]) if len(reached) else len(order) - 1
    prefix = order[: index + 1]
    unknown_recalled = int(is_unknown[prefix].sum())
    known_false_rejects = int(true_known[prefix].sum())
    threshold = float(scores[order[index]])
    return {
        "fpr95": float(known_false_rejects / int(true_known.sum())),
        "threshold_score_at_fpr95": threshold,
        "rank_prefix_count": int(index + 1),
        "unknown_recall_at_prefix": float(unknown_recalled / int(is_unknown.sum())),
        "known_false_rejects_at_prefix": known_false_rejects,
        "unknown_false_accepts_at_prefix": int(is_unknown.sum()) - unknown_recalled,
        "score_tie_count_at_threshold": int(np.sum(scores == threshold)),
    }


def summarize(scores: np.ndarray, true_known: np.ndarray) -> dict:
    known = scores[true_known]
    unknown = scores[~true_known]
    known_q = quantiles(known, (0.90, 0.95, 0.99))
    unknown_q = quantiles(unknown, (0.01, 0.05, 0.10))
    prefix = fpr95_prefix(scores, true_known)
    return {
        "known": {
            "count": int(known.size),
            "mean": float(known.mean()),
            "std": float(known.std()),
            **known_q,
            "fraction_at_or_above_fpr95_threshold": float(
                np.mean(known >= prefix["threshold_score_at_fpr95"])
            ),
        },
        "unknown": {
            "count": int(unknown.size),
            "mean": float(unknown.mean()),
            "std": float(unknown.std()),
            **unknown_q,
            "fraction_below_or_equal_known_q95": float(
                np.mean(unknown <= known_q["q95"])
            ),
        },
        "unknown_minus_known": {
            "mean_gap": float(unknown.mean() - known.mean()),
            "median_gap": float(np.median(unknown) - np.median(known)),
            "q05_unknown_minus_q95_known": float(unknown_q["q05"] - known_q["q95"]),
        },
        "fpr95_prefix": prefix,
    }


def markdown(result: dict) -> str:
    lines = [
        f"# FPR95 tail audit: {result['comparison']}\n",
        "This is a retrospective test-label diagnostic. Labels were not used for training, calibration, checkpoint selection, or method selection.\n",
        "## Tail summary\n",
        "| arm | known q90 | known q95 | known q99 | unknown q01 | unknown q05 | unknown q10 | unknown below known q95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in result["arms"].items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {:.3f}% |".format(
                arm,
                *(f"{item['known'][key]:.6f}" for key in ("q90", "q95", "q99")),
                *(f"{item['unknown'][key]:.6f}" for key in ("q01", "q05", "q10")),
                100.0 * item["unknown"]["fraction_below_or_equal_known_q95"],
            )
        )
    lines += [
        "",
        "## Exact ranked prefix used by FPR95\n",
        "| arm | FPR95 | threshold | unknown recall | known false rejects | unknown false accepts | prefix size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in result["arms"].items():
        prefix = item["fpr95_prefix"]
        lines.append(
            "| {} | {:.3f}% | {:.6f} | {:.3f}% | {} | {} | {} |".format(
                arm,
                100.0 * prefix["fpr95"],
                prefix["threshold_score_at_fpr95"],
                100.0 * prefix["unknown_recall_at_prefix"],
                prefix["known_false_rejects_at_prefix"],
                prefix["unknown_false_accepts_at_prefix"],
                prefix["rank_prefix_count"],
            )
        )
    lines += [
        "",
        "## Interpretation\n",
        "FPR95 is determined by the highest-score ranked prefix needed to recall at least 95% of unknown samples. This audit separates the known high-score tail from the unknown low-score tail; it does not change or select a method.\n",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    arms = {}
    source_files = {}
    for name, directory in args.arm:
        detail_path = Path(directory) / "discovery_detail.json"
        if not detail_path.is_file():
            raise FileNotFoundError(detail_path)
        scores, true_known = read_detail(detail_path)
        arms[name] = summarize(scores, true_known)
        source_files[name] = str(detail_path)

    result = {
        "protocol_note": "Retrospective test-label tail audit only; no labels were used for training, calibration, checkpoint selection, or method selection.",
        "comparison": args.comparison,
        "arms": arms,
        "source_detail_files": source_files,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
