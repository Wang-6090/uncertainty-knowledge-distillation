from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

# Allow ``python scripts/diagnose_joint_assignments.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novel_discovery.data import build_data_bundle
from novel_discovery.joint_discovery import (
    NovelPrototypeHead,
    balanced_assignments,
    combine_known_novel_logits,
)
from novel_discovery.models import build_model
from novel_discovery.pipeline import build_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline diagnostics for minibatch joint-discovery assignments."
    )
    parser.add_argument("--run-dir", required=True, help="Student run directory containing config.json and student.pt")
    parser.add_argument(
        "--open-val-ratio", type=float, default=0.0,
        help="Must match the ratio used when the discovery pool was constructed.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


@torch.no_grad()
def collect_logits(model, novel_head, loader, device):
    all_known_logits, all_novel_logits, all_joint_logits = [], [], []
    all_known, all_raw = [], []
    model.eval()
    novel_head.eval()
    for images, _, raw_labels, is_known, _ in loader:
        images = images.to(device)
        outputs = model(images)
        novel_logits = novel_head(outputs["features"])
        joint_logits = combine_known_novel_logits(outputs["logits"], novel_logits)
        all_known_logits.append(outputs["logits"].cpu())
        all_novel_logits.append(novel_logits.cpu())
        all_joint_logits.append(joint_logits.cpu())
        all_known.append(torch.as_tensor(is_known).bool())
        all_raw.append(torch.as_tensor(raw_labels).long())
    return (
        torch.cat(all_known_logits),
        torch.cat(all_novel_logits),
        torch.cat(all_joint_logits),
        torch.cat(all_known),
        torch.cat(all_raw),
    )


def summarize_assignment(assignments, is_known, raw_labels, num_known, num_novel):
    assignments = np.asarray(assignments, dtype=np.int64)
    is_known = np.asarray(is_known, dtype=bool)
    raw_labels = np.asarray(raw_labels, dtype=np.int64)
    novel_assignment = assignments >= num_known
    novel_unknown = (~is_known) & novel_assignment
    novel_counts = np.bincount(
        assignments[novel_unknown] - num_known,
        minlength=num_novel,
    ) if np.any(novel_unknown) else np.zeros(num_novel, dtype=np.int64)
    counts = np.bincount(assignments, minlength=num_known + num_novel)
    probabilities = novel_counts / max(int(novel_counts.sum()), 1)
    positive = probabilities[probabilities > 0]
    normalized_entropy = (
        float(-(positive * np.log(positive)).sum() / np.log(num_novel))
        if positive.size and num_novel > 1 else 0.0
    )
    novel_only_targets = assignments[~is_known].copy()
    novel_only_targets[novel_only_targets < num_known] = -1
    return {
        "samples": int(len(assignments)),
        "known_samples": int(is_known.sum()),
        "novel_samples": int((~is_known).sum()),
        "known_assigned_to_novel_rate": float(novel_assignment[is_known].mean()) if is_known.any() else None,
        "novel_assigned_to_novel_rate": float(novel_assignment[~is_known].mean()) if (~is_known).any() else None,
        "active_novel_prototypes": int(np.count_nonzero(counts[num_known:])),
        "empty_total_prototypes": int(np.count_nonzero(counts == 0)),
        "novel_assignment_normalized_entropy": normalized_entropy,
        "novel_pool_assignment_ari": (
            float(adjusted_rand_score(raw_labels[~is_known], novel_only_targets))
            if (~is_known).any() else None
        ),
    }


def summarize_novel_mass(logits, is_known, num_known):
    probabilities = logits.softmax(dim=-1)
    novel_mass = probabilities[:, num_known:].sum(dim=-1).numpy()
    is_known = np.asarray(is_known, dtype=bool)
    result = {}
    for name, mask in (("known", is_known), ("novel", ~is_known), ("all", np.ones(len(is_known), dtype=bool))):
        values = novel_mass[mask]
        result[name] = {
            "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "q10": float(np.quantile(values, 0.10)) if values.size else None,
            "q90": float(np.quantile(values, 0.90)) if values.size else None,
            "above_0_5": float((values >= 0.5).mean()) if values.size else None,
        }
    return result


def summarize_novel_head(logits, is_known, raw_labels):
    """Diagnose the novel head without competing against known logits."""
    probabilities = logits.softmax(dim=-1)
    assignments = probabilities.argmax(dim=-1).numpy()
    is_known = np.asarray(is_known, dtype=bool)
    raw_labels = np.asarray(raw_labels, dtype=np.int64)
    num_novel = logits.size(-1)
    result = {
        "mean_max_probability": float(probabilities.max(dim=-1).values.mean()),
        "mean_normalized_entropy": float(
            (-(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
             / np.log(max(num_novel, 2))).mean()
        ),
    }
    for name, mask in (("known", is_known), ("novel", ~is_known)):
        values = probabilities[mask]
        assigned = assignments[mask]
        result[name] = {
            "samples": int(mask.sum()),
            "mean_max_probability": float(values.max(dim=-1).values.mean()) if len(values) else None,
            "active_prototypes": int(np.unique(assigned).size) if len(assigned) else 0,
            "assignment_entropy": float(
                -(np.bincount(assigned, minlength=num_novel) / max(len(assigned), 1)
                  * np.log(np.clip(np.bincount(assigned, minlength=num_novel) / max(len(assigned), 1), 1e-8, None))).sum()
                / np.log(max(num_novel, 2))
            ) if len(assigned) else None,
        }
    if (~is_known).any():
        result["novel_pool_argmax_ari"] = float(
            adjusted_rand_score(raw_labels[~is_known], assignments[~is_known])
        )
    else:
        result["novel_pool_argmax_ari"] = None
    return result


def assignments_for_partition(logits, batch_size, permutation, temperature):
    result = torch.empty(logits.size(0), dtype=torch.long)
    for indices in permutation.split(batch_size):
        batch = logits[indices]
        soft = balanced_assignments(batch, temperature=temperature)
        result[indices] = soft.argmax(dim=-1)
    return result


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "student.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    bundle = build_data_bundle(
        config["dataset"],
        config["data_root"],
        int(config["num_known"]),
        int(config["seed"]),
        int(config["image_size"]),
        split_path=config.get("split_path"),
        limit_train=config.get("limit_train") or None,
        limit_val=config.get("limit_val") or None,
        limit_discovery=config.get("limit_discovery") or None,
        discovery_pool_mode=config.get("discovery_pool_mode", "unknown"),
        open_val_ratio=args.open_val_ratio,
        known_split_mode=config.get("known_split_mode", "random"),
    )
    if bundle.discovery_pool is None or not len(bundle.discovery_pool):
        raise ValueError("The configured run has no discovery pool.")

    model = build_model(
        len(bundle.known_classes),
        backbone=config.get("student_backbone") or config.get("backbone", "resnet18"),
        proj_dim=int(config.get("proj_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    novel_count = int(config["joint_num_novel"])
    novel_head = NovelPrototypeHead(
        model.encoder.out_dim,
        novel_count,
        temperature=float(config.get("joint_head_temperature", 0.2)),
    ).to(device)
    novel_state = checkpoint.get("joint_novel_head")
    if novel_state is None:
        raise ValueError("Checkpoint does not contain a joint novel head.")
    novel_head.load_state_dict(novel_state)

    loader = build_loader(
        bundle.discovery_pool,
        int(config.get("discovery_batch_size") or config.get("batch_size", 128)),
        False,
        args.num_workers,
    )
    known_logits, novel_logits, logits, is_known, raw_labels = collect_logits(
        model, novel_head, loader, device
    )
    num_known = len(bundle.known_classes)
    global_assignment = balanced_assignments(
        logits, temperature=float(config.get("joint_assignment_temperature", 1.0))
    ).argmax(dim=-1)
    known_np = is_known.numpy()
    raw_np = raw_labels.numpy()
    global_np = global_assignment.numpy()
    global_summary = summarize_assignment(
        global_np, known_np, raw_np, num_known, novel_count
    )
    novel_head_summary = summarize_novel_head(novel_logits, known_np, raw_np)

    seed = int(config["seed"])
    comparison = []
    for batch_size in args.batch_sizes:
        for repeat in range(args.repeats):
            generator = torch.Generator().manual_seed(seed + repeat)
            permutation = torch.randperm(logits.size(0), generator=generator)
            assignments = assignments_for_partition(
                logits,
                batch_size,
                permutation,
                float(config.get("joint_assignment_temperature", 1.0)),
            )
            assigned_np = assignments.numpy()
            summary = summarize_assignment(
                assigned_np, known_np, raw_np, num_known, novel_count
            )
            summary.update(
                {
                    "batch_size": int(batch_size),
                    "repeat": int(repeat),
                    "agreement_with_global_assignment": float((assigned_np == global_np).mean()),
                    "agreement_with_previous_repeat": None,
                }
            )
            if repeat > 0:
                summary["agreement_with_previous_repeat"] = float(
                    (assigned_np == comparison[-1]["_assignments"]).mean()
                )
            summary["_assignments"] = assigned_np
            comparison.append(summary)

    for item in comparison:
        item.pop("_assignments", None)
    report = {
        "run_dir": str(run_dir),
        "device": str(device),
        "pool_size": int(len(bundle.discovery_pool)),
        "num_known_classes": int(num_known),
        "num_novel_prototypes": int(novel_count),
        "confidence_threshold": float(config.get("joint_confidence_threshold", 1.0)),
        "confidence_threshold_pass_rate": float(
            ((logits.softmax(dim=-1).max(dim=-1).values * logits.size(-1))
             >= float(config.get("joint_confidence_threshold", 1.0))).float().mean()
        ),
        "global_assignment": global_summary,
        "novel_head_without_known_competition": novel_head_summary,
        "joint_softmax_novel_mass": summarize_novel_mass(logits, known_np, num_known),
        "minibatch_assignments": comparison,
        "notes": [
            "Labels are used only for post-hoc diagnostics, never to compute model outputs or assignments.",
            "Features come from one randomly augmented view per pool sample; this is an offline checkpoint diagnostic, not a training replay.",
            "Global Sinkhorn assignments are a comparison reference, not a proposed replacement validated by this report.",
        ],
    }
    output_path = Path(args.output) if args.output else run_dir / "joint_assignment_diagnostics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
