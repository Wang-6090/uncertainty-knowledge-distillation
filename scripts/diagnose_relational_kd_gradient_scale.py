"""Measure feature-KD vs relational-KD backbone gradient scales on known train data.

This is a training-signal diagnostic only. It does not train or update a model,
and it never uses unknown identities, open validation, discovery-pool labels, or
test data. The saved student/teacher checkpoints and a verified local split are
used to compare the current projection feature KD objective against the
candidate batchwise relational distance objective.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.data import OpenSetCIFAR100, build_transforms
from novel_discovery.losses import feature_distillation_loss, relational_distance_distillation_loss
from novel_discovery.models import build_model


SEED = 2109
BATCHES = 16
BATCH_SIZE = 64
FEATURE_KD_ALPHA = 0.1


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"Unsupported checkpoint: {path}")
    return payload


def make_model(config: dict, checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(
        num_classes=len(checkpoint["known_classes"]),
        backbone=config["backbone"],
        proj_dim=int(config["proj_dim"]),
        dropout=float(config["dropout"]),
        pretrained=False,
        cifar_stem=bool(config["cifar_stem"]),
        rejection_feature_dim=int(config.get("rejection_feature_dim", 0)),
        independent_rejection_encoder=bool(config.get("independent_rejection_encoder", False)),
        independent_rejection_encoder_init=config.get("independent_rejection_encoder_init", "shared_backbone"),
        independent_rejection_input_norm=config.get("independent_rejection_input_norm", "none"),
        structure_feature_dim=int(config.get("structure_feature_dim", 0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def grad_for(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    # Preserve parameter offsets even if an objective does not use every
    # encoder parameter; otherwise the two flattened vectors could misalign.
    flattened = [
        (torch.zeros_like(param) if grad is None else grad).detach().float().reshape(-1)
        for param, grad in zip(params, grads)
    ]
    return torch.cat(flattened)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    device = torch.device("cuda")

    split_path = ROOT / "splits/semantic_hard/seed2109/cifar100_60_40_semantic_hard.json"
    student_dir = ROOT / "runs/rejection_epoch_budget_s2109_four_epoch"
    teacher_dir = ROOT / "runs/rejection_epoch_budget_s2109_teacher"
    student_config = read_json(student_dir / "train_student_config.json")
    teacher_config = read_json(teacher_dir / "train_teacher_config.json")
    split = read_json(split_path)
    student_ckpt = load_payload(student_dir / "student.pt")
    teacher_ckpt = load_payload(teacher_dir / "teacher.pt")
    known, novel = split["known_classes"], split["novel_classes"]
    if split.get("protocol") != "semantic_hard" or split.get("metadata", {}).get("seed") != SEED:
        raise ValueError("Split protocol/seed mismatch")
    if len(known) != 60 or len(novel) != 40 or set(known) & set(novel):
        raise ValueError("Invalid 60/40 split")
    for role, config, ckpt in (("student", student_config, student_ckpt), ("teacher", teacher_config, teacher_ckpt)):
        if config.get("seed") != SEED or ckpt.get("known_classes") != known or ckpt.get("novel_classes") != novel:
            raise ValueError(f"{role} config/checkpoint does not match verified split")

    transform = build_transforms(int(student_config["image_size"]), train=True, profile=student_config["transform_profile"])
    dataset = OpenSetCIFAR100(
        str(ROOT / "data"), known, train=True, transform=transform, download=False, include_unknown=False
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                        num_workers=0, drop_last=True)
    student = make_model(student_config, student_ckpt, device)
    teacher = make_model(teacher_config, teacher_ckpt, device)
    params = [p for p in student.encoder.parameters() if p.requires_grad]

    rows = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= BATCHES:
            break
        images = batch[0].to(device)
        # DataLoader only contains the 60 known classes; identity labels are not
        # used for either objective, so no unknown identity signal is available.
        with torch.no_grad():
            teacher_out = teacher(images)
        student_out = student(images)
        feat_loss = feature_distillation_loss(student_out["proj"], teacher_out["proj"])
        rel_loss = relational_distance_distillation_loss(student_out["features"], teacher_out["features"])
        # Both objectives share this student forward graph. Retain it for the
        # first gradient, then release it when computing the second.
        g_feat = grad_for(feat_loss, params, retain_graph=True)
        g_rel = grad_for(rel_loss, params, retain_graph=False)
        feat_norm = float(g_feat.norm().item())
        rel_norm = float(g_rel.norm().item())
        cosine = float(torch.nn.functional.cosine_similarity(g_feat, g_rel, dim=0).item())
        rows.append({
            "batch": batch_index,
            "feature_kd_loss_raw": float(feat_loss.detach().item()),
            "relational_kd_loss_raw": float(rel_loss.detach().item()),
            "feature_kd_backbone_grad_norm_raw": feat_norm,
            "relational_kd_backbone_grad_norm_raw": rel_norm,
            "feature_kd_backbone_grad_norm_weighted_alpha_0_1": FEATURE_KD_ALPHA * feat_norm,
            "relational_alpha_to_match_feature_kd_gradient": FEATURE_KD_ALPHA * feat_norm / max(rel_norm, 1e-12),
            "gradient_cosine": cosine,
        })
        del student_out, teacher_out, feat_loss, rel_loss, g_feat, g_rel

    if len(rows) != BATCHES:
        raise RuntimeError(f"Expected {BATCHES} complete batches, got {len(rows)}")
    fields = [k for k in rows[0] if k != "batch"]
    summary = {}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        summary[field] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    payload = {
        "purpose": "known-train multi-batch gradient-scale diagnostic; no optimizer steps",
        "seed": SEED,
        "protocol": "semantic_hard",
        "known_classes": len(known),
        "unknown_identity_labels_used": False,
        "data_accessed": "CIFAR-100 train images restricted to known classes only",
        "checkpoint_epochs": {"student": student_config.get("best_epoch"), "teacher": teacher_config.get("best_epoch")},
        "batches": BATCHES,
        "batch_size": BATCH_SIZE,
        "feature_kd_alpha_baseline": FEATURE_KD_ALPHA,
        "compared_objectives": {
            "feature_kd": "existing cosine loss between 128-D student/teacher projections",
            "relational_kd": "normalized within-batch pairwise distances between 512-D classification features",
        },
        "summary": summary,
        "per_batch": rows,
    }
    output = ROOT / "analysis/relational_kd_gradient_scale_s2109_20261114.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
