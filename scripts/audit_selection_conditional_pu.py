"""Synthetic audit for the selection-conditional PU prior correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from novel_discovery.losses import selection_conditional_known_prior


def run_case(
    *,
    seed: int,
    known_prior: float,
    known_selection_rate: float,
    unknown_selection_rate: float,
    known_risk: float,
    unknown_risk: float,
    samples: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    is_unknown = rng.random(samples) >= known_prior
    selected_prob = np.where(is_unknown, unknown_selection_rate, known_selection_rate)
    selected = rng.random(samples) < selected_prob
    is_known = ~is_unknown
    mixed_rate = float(selected.mean())
    selected_known_rate = float(selected[is_known].mean())
    selected_prior = selection_conditional_known_prior(
        known_prior, selected_known_rate, mixed_rate
    )
    selected_mixed_risk = float(
        selected_prior * known_risk + (1.0 - float(selected_prior)) * unknown_risk
    )
    recovered = (selected_mixed_risk - float(selected_prior) * known_risk) / (
        1.0 - float(selected_prior)
    )
    analytical_prior = known_prior * known_selection_rate / (
        known_prior * known_selection_rate
        + (1.0 - known_prior) * unknown_selection_rate
    )
    return {
        "known_prior": known_prior,
        "known_selection_rate": known_selection_rate,
        "unknown_selection_rate": unknown_selection_rate,
        "empirical_mixed_selection_rate": mixed_rate,
        "empirical_selected_known_rate": selected_known_rate,
        "analytical_selected_known_prior": analytical_prior,
        "estimated_selected_known_prior": float(selected_prior),
        "true_selected_unknown_risk": unknown_risk,
        "recovered_selected_unknown_risk": recovered,
        "absolute_risk_error": abs(recovered - unknown_risk),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--samples", type=int, default=500_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    cases = [
        run_case(
            seed=args.seed,
            known_prior=0.2,
            known_selection_rate=0.1,
            unknown_selection_rate=0.5,
            known_risk=1.5,
            unknown_risk=4.5,
            samples=args.samples,
        ),
        run_case(
            seed=args.seed + 1,
            known_prior=0.2,
            known_selection_rate=0.8,
            unknown_selection_rate=0.3,
            known_risk=0.25,
            unknown_risk=2.75,
            samples=args.samples,
        ),
    ]
    report = {
        "cases": cases,
        "max_absolute_risk_error": max(c["absolute_risk_error"] for c in cases),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
