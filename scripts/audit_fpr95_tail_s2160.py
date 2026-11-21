"""Audit the score tails that determine FPR95 on the seed-2160 pair.

This is a retrospective diagnostic. Test labels are read only to explain the
already computed metric and are never used for training, calibration, or model
selection. The audit reports score quantiles and the exact ranked prefix at
which unknown recall first reaches 95 percent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_detail(path: Path) -> tuple[np.ndarray, np.ndarray]:
    detail = json.loads(path.read_text(encoding="utf-8"))
    scores = np.asarray(detail["score"], dtype=np.float64)
    true_known = np.asarray(detail["true_known"], dtype=bool)
    if scores.ndim != 1 or true_known.shape != scores.shape:
        raise ValueError(f"invalid score/label shape in {path}")
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite scores in {path}")
    return scores, true_known


def _quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float]:
    return {f"q{int(probability * 100):02d}": float(np.quantile(values, probability)) for probability in probabilities}


def _fpr95_prefix(scores: np.ndarray, true_known: np.ndarray) -> dict:
    is_unknown = ~true_known
    order = np.argsort(scores)[::-1]
    ranked_unknown = is_unknown[order]
    unknown_total = int(is_unknown.sum())
    known_total = int(true_known.sum())
    cumulative_unknown = np.cumsum(ranked_unknown, dtype=np.int64)
    target = 0.95 * unknown_total
    reached = np.flatnonzero(cumulative_unknown >= target)
    if not len(reached):
        index = len(order) - 1
    else:
        index = int(reached[0])
    prefix = order[: index + 1]
    unknown_recalled = int(is_unknown[prefix].sum())
    known_false_rejects = int(true_known[prefix].sum())
    unknown_false_accepts = unknown_total - unknown_recalled
    threshold_score = float(scores[order[index]])
    return {
        "fpr95": float(known_false_rejects / known_total),
        "rank_index_zero_based": index,
        "rank_prefix_count": int(index + 1),
        "threshold_score_at_fpr95": threshold_score,
        "unknown_recall_at_prefix": float(unknown_recalled / unknown_total),
        "unknown_recalled_at_prefix": unknown_recalled,
        "unknown_total": unknown_total,
        "known_false_rejects_at_prefix": known_false_rejects,
        "known_total": known_total,
        "unknown_false_accepts_at_prefix": unknown_false_accepts,
        "score_tie_count_at_threshold": int(np.sum(scores == threshold_score)),
    }


def _summarize(scores: np.ndarray, true_known: np.ndarray) -> dict:
    known_scores = scores[true_known]
    unknown_scores = scores[~true_known]
    known_q = _quantiles(known_scores, (0.90, 0.95, 0.99))
    unknown_q = _quantiles(unknown_scores, (0.01, 0.05, 0.10))
    fpr = _fpr95_prefix(scores, true_known)
    fpr_threshold = fpr["threshold_score_at_fpr95"]
    return {
        "known": {
            "count": int(known_scores.size),
            "mean": float(known_scores.mean()),
            "std": float(known_scores.std()),
            **known_q,
            "fraction_at_or_above_fpr95_threshold": float(np.mean(known_scores >= fpr_threshold)),
        },
        "unknown": {
            "count": int(unknown_scores.size),
            "mean": float(unknown_scores.mean()),
            "std": float(unknown_scores.std()),
            **unknown_q,
            "fraction_below_or_equal_known_q95": float(
                np.mean(unknown_scores <= known_q["q95"])
            ),
        },
        "unknown_minus_known": {
            "mean_gap": float(unknown_scores.mean() - known_scores.mean()),
            "median_gap": float(np.median(unknown_scores) - np.median(known_scores)),
            "q05_unknown_minus_q95_known": float(unknown_q["q05"] - known_q["q95"]),
        },
        "fpr95_prefix": fpr,
    }


def _fmt(value: float, percent: bool = False) -> str:
    if percent:
        return f"{100.0 * value:.3f}%"
    return f"{value:.6f}"


def _markdown(result: dict) -> str:
    lines = [
        "# FPR95 tail audit: seed 2160",
        "",
        "This is a retrospective test-label diagnostic. Labels were not used for training, calibration, checkpoint selection, or method selection.",
        "",
        "## Tail summary",
        "",
        "| score variant | known q90 | known q95 | known q99 | unknown q01 | unknown q05 | unknown q10 | unknown below known q95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["variants"].items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                _fmt(item["known"]["q90"]),
                _fmt(item["known"]["q95"]),
                _fmt(item["known"]["q99"]),
                _fmt(item["unknown"]["q01"]),
                _fmt(item["unknown"]["q05"]),
                _fmt(item["unknown"]["q10"]),
                _fmt(item["unknown"]["fraction_below_or_equal_known_q95"], True),
            )
        )
    lines += [
        "",
        "## Exact ranked prefix used by FPR95",
        "",
        "| score variant | FPR95 | threshold score | unknown recall at prefix | known false rejects | unknown false accepts | prefix size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["variants"].items():
        fpr = item["fpr95_prefix"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                _fmt(fpr["fpr95"], True),
                _fmt(fpr["threshold_score_at_fpr95"]),
                _fmt(fpr["unknown_recall_at_prefix"], True),
                fpr["known_false_rejects_at_prefix"],
                fpr["unknown_false_accepts_at_prefix"],
                fpr["rank_prefix_count"],
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "FPR95 is determined by the highest-score ranked prefix needed to recall at least 95% of unknown samples. A larger known high-score tail increases the numerator; a larger unknown low-score tail forces the prefix farther down the ranking.",
        "",
        "The boundary variant is the frozen late-decay checkpoint with the previously fixed boundary-uncertainty correction. It is included to test whether the post-hoc correction changes the tail metric; it is not a new trained method.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--late-dir", type=Path, required=True)
    parser.add_argument("--boundary-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "static_control_raw": args.static_dir / "discovery_detail.json",
        "late_decay_raw": args.late_dir / "discovery_detail.json",
        "late_decay_boundary": args.boundary_dir / "discovery_detail.json",
    }
    variants = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        scores, true_known = _read_detail(path)
        variants[name] = _summarize(scores, true_known)

    result = {
        "protocol_note": "Retrospective test-label tail audit only; no labels were used for training, calibration, checkpoint selection, or method selection.",
        "seed": 2160,
        "variants": variants,
        "source_detail_files": {name: str(path) for name, path in paths.items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
