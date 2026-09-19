from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch

# Keep downloaded torchvision weights inside the project by default.
os.environ.setdefault("TORCH_HOME", str(Path.cwd() / ".torch_cache"))

from novel_discovery.data import TwoViewDataset, build_data_bundle
from novel_discovery.metrics import compute_auroc
from novel_discovery.models import build_model
from novel_discovery.pipeline import (
    build_loader,
    calibrate_threshold,
    collect_diagonal_gaussian_stats,
    collect_prototypes,
    fit_score_normalization,
    evaluate_classification,
    extract_outputs,
    compute_open_score,
    apply_temperature,
    calibration_diagnostics,
    fit_temperature,
    run_discovery,
    train_one_epoch_student,
    train_one_epoch_teacher,
    uncertainty_error_diagnostics,
)
from novel_discovery.utils import ensure_dir, save_json, set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagefolder", "toy", "fake"])
        p.add_argument("--data-root", default="./data")
        p.add_argument("--split-path", default="./splits.json")
        p.add_argument("--num-known", type=int, default=60)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--image-size", type=int, default=224)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--backbone", default="resnet18")
        p.add_argument("--teacher-backbone", default=None)
        p.add_argument("--student-backbone", default=None)
        p.add_argument("--pretrained", action="store_true")
        p.add_argument("--proj-dim", type=int, default=128)
        p.add_argument("--dropout", type=float, default=0.2)
        p.add_argument("--download", action="store_true")
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        p.add_argument("--work-dir", default="./runs")
        p.add_argument("--limit-train", type=int, default=0)
        p.add_argument("--limit-val", type=int, default=0)
        p.add_argument("--limit-test", type=int, default=0)
        p.add_argument("--limit-discovery", type=int, default=0)
        p.add_argument("--discovery-pool-mode", choices=["unknown", "mixed"], default="unknown")

    p = sub.add_parser("train_teacher")
    add_common(p)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha-unc", type=float, default=0.1)
    p.add_argument("--alpha-proto", type=float, default=0.0)
    p.add_argument("--alpha-proxy", type=float, default=0.0)
    p.add_argument("--proxy-temperature", type=float, default=0.1)
    p.add_argument("--alpha-pseudo", type=float, default=0.0)
    p.add_argument("--alpha-energy", type=float, default=0.0)
    p.add_argument("--energy-margin", type=float, default=1.0)
    p.add_argument("--energy-temperature", type=float, default=1.0)
    # Keep the validated baseline as the default; ``strong`` remains an
    # explicit experimental option and is documented by its own run.
    p.add_argument("--pseudo-mode", choices=["legacy", "strong"], default="legacy")
    p.add_argument("--pseudo-feature-noise", type=float, default=0.05)
    p.add_argument("--uncertainty-target-mode", choices=["confidence", "classification_error", "margin"], default="confidence")
    p.add_argument("--teacher-ckpt", default="./runs/teacher.pt")

    p = sub.add_parser("train_student")
    add_common(p)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha-unc", type=float, default=0.1)
    p.add_argument("--alpha-kd", type=float, default=1.0)
    p.add_argument("--alpha-feat-kd", type=float, default=0.0)
    p.add_argument("--kd-mode", choices=["standard", "uncertainty"], default="uncertainty")
    p.add_argument("--alpha-supcon", type=float, default=0.1)
    p.add_argument("--alpha-proto", type=float, default=0.0)
    p.add_argument("--alpha-proxy", type=float, default=0.0)
    p.add_argument("--proxy-temperature", type=float, default=0.1)
    p.add_argument("--alpha-pseudo", type=float, default=0.0)
    p.add_argument("--alpha-energy", type=float, default=0.0)
    p.add_argument("--energy-margin", type=float, default=1.0)
    p.add_argument("--energy-temperature", type=float, default=1.0)
    p.add_argument("--pseudo-mode", choices=["legacy", "strong"], default="legacy")
    p.add_argument("--pseudo-feature-noise", type=float, default=0.05)
    p.add_argument("--uncertainty-target-mode", choices=["confidence", "classification_error", "margin"], default="confidence")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--uncertainty-weight-mode", choices=["raw", "mean_normalized"], default="raw")
    p.add_argument("--teacher-ckpt", default="./runs/teacher.pt")
    p.add_argument("--student-ckpt", default="./runs/student.pt")
    p.add_argument(
        "--discovery-pool",
        action="store_true",
        help="Use unlabeled novel-class training images for two-view consistency learning.",
    )
    p.add_argument("--alpha-discovery", type=float, default=0.0)
    p.add_argument("--alpha-discovery-unknown", type=float, default=0.0)
    p.add_argument(
        "--alpha-discovery-energy",
        type=float,
        default=0.0,
        help="Apply energy-margin separation between known samples and a pure unknown discovery pool.",
    )
    p.add_argument(
        "--alpha-discovery-selective-unknown",
        type=float,
        default=0.0,
        help="Apply unknown loss only to high-risk samples selected from an unlabeled discovery pool.",
    )
    p.add_argument(
        "--alpha-discovery-selective-energy",
        type=float,
        default=0.0,
        help="Apply energy separation only to high-risk samples selected from an unlabeled discovery pool.",
    )
    p.add_argument("--discovery-select-ratio", type=float, default=0.25)
    p.add_argument(
        "--discovery-select-mode",
        choices=["entropy", "max_softmax", "energy", "entropy_uncertainty", "consensus"],
        default="entropy_uncertainty",
    )
    p.add_argument(
        "--discovery-selective-warmup-epochs",
        type=int,
        default=0,
        help="Keep selective discovery unknown/energy losses disabled for the first N epochs.",
    )
    p.add_argument(
        "--discovery-selective-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp selective discovery unknown/energy loss weights after warmup.",
    )
    p.add_argument(
        "--discovery-selection-model",
        choices=["student", "ema"],
        default="student",
        help="Model used to select high-risk discovery samples; ema follows Mean Teacher-style smoothing.",
    )
    p.add_argument(
        "--discovery-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay for discovery-selection model when --discovery-selection-model ema is used.",
    )
    p.add_argument(
        "--discovery-neighbor-filter",
        action="store_true",
        help="Filter selected discovery candidates by kNN agreement in projection space.",
    )
    p.add_argument(
        "--discovery-neighbor-k",
        type=int,
        default=5,
        help="Number of nearest neighbors used by --discovery-neighbor-filter.",
    )
    p.add_argument(
        "--discovery-neighbor-min-votes",
        type=int,
        default=2,
        help="Minimum selected neighbors required to keep a discovery candidate.",
    )
    p.add_argument("--discovery-batch-size", type=int, default=0)
    p.add_argument("--discovery-loss", choices=["consistency", "nt_xent"], default="nt_xent")
    p.add_argument("--discovery-temperature", type=float, default=0.2)

    p = sub.add_parser("discover")
    add_common(p)
    p.add_argument("--student-ckpt", default="./runs/student.pt")
    p.add_argument("--num-novel", type=int, default=40)
    p.add_argument("--cluster-k", choices=["oracle", "auto"], default="auto")
    p.add_argument(
        "--cluster-method",
        choices=["kmeans", "agglomerative", "spectral"],
        default="kmeans",
        help="Clustering algorithm for the rejected/unknown candidate pool.",
    )
    p.add_argument(
        "--cluster-feature",
        choices=["projection", "projection_pca", "feature", "feature_pca"],
        default="projection",
        help="Representation used for clustering; *_pca applies PCA and whitening.",
    )
    p.add_argument(
        "--cluster-selection",
        choices=["silhouette", "composite", "stability"],
        default="silhouette",
        help="K selection criterion: silhouette, composite internal metrics, or stability.",
    )
    p.add_argument("--cluster-pca-dim", type=int, default=32)
    p.add_argument(
        "--cluster-normalize",
        action="store_true",
        help="L2-normalize clustering features (default remains raw features for compatibility).",
    )
    p.add_argument("--cluster-no-whiten", action="store_true")
    p.add_argument("--cluster-n-init", type=int, default=10)
    p.add_argument("--cluster-stability-repeats", type=int, default=5)
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    p.add_argument("--mc-samples", type=int, default=8)
    p.add_argument("--temperature-calibration", action="store_true")
    p.add_argument("--calibration-bins", type=int, default=15)
    p.add_argument(
        "--score-mode",
        default="entropy_proto",
        choices=[
            "auto",
            "full",
            "max_softmax",
            "odin_msp",
            "energy",
            "entropy_only",
            "proto_only",
            "entropy_proto",
            "normalized_entropy_proto",
            "entropy_epistemic",
            "entropy_aleatoric",
            "expected_entropy",
            "epistemic",
            "mahalanobis",
            "mahalanobis_diag",
            "mahalanobis_shared",
            "entropy_mahalanobis",
            "entropy_mahalanobis_diag",
            "entropy_mahalanobis_shared",
            "normalized_entropy_mahalanobis",
            "normalized_entropy_mahalanobis_diag",
            "normalized_entropy_mahalanobis_shared",
        ],
    )
    p.add_argument("--open-val-ratio", type=float, default=0.0)
    p.add_argument("--auto-calibrate-score", action="store_true")
    p.add_argument(
        "--odin-epsilon",
        type=float,
        default=0.0,
        help="Enable ODIN-style input perturbation for odin_msp score; value is in normalized image units.",
    )
    p.add_argument(
        "--odin-temperature",
        type=float,
        default=1000.0,
        help="Temperature used by ODIN-style MSP scoring.",
    )

    p = sub.add_parser("inspect_data")
    add_common(p)
    p.add_argument("--sample-count", type=int, default=5)
    return parser.parse_args(argv)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available in this Python environment, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def save_checkpoint(model, path, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def scheduled_weight(epoch: int, warmup_epochs: int = 0, ramp_epochs: int = 0) -> float:
    """Return a loss multiplier using epoch-indexed warmup and linear ramp."""
    warmup_epochs = max(int(warmup_epochs), 0)
    ramp_epochs = max(int(ramp_epochs), 0)
    epoch_number = int(epoch) + 1
    if epoch_number <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, (epoch_number - warmup_epochs) / ramp_epochs))


def _checkpoint_parts(path_arg: str):
    path = Path(path_arg)
    return tuple(part for part in path.parts if part not in {"."})


def _is_default_checkpoint(path_arg: str, default_name: str) -> bool:
    parts = _checkpoint_parts(path_arg)
    return parts in {(default_name,), ("runs", default_name)}


def resolve_output_checkpoint(path_arg: str, work_dir: str, default_name: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if _is_default_checkpoint(path_arg, default_name):
        return Path(work_dir) / path.name
    return path


def resolve_input_checkpoint(path_arg: str, work_dir: str, default_name: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute() or not _is_default_checkpoint(path_arg, default_name):
        if path.exists():
            return path
        available = sorted(str(p) for p in Path(work_dir).glob("*.pt"))
        raise FileNotFoundError(
            f"Could not find checkpoint at {path}. Available .pt files in work dir: {available}"
        )

    candidates = [Path(work_dir) / default_name, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = sorted(str(p) for p in Path(work_dir).glob("*.pt"))
    raise FileNotFoundError(
        f"Could not find checkpoint. Checked: {', '.join(str(c) for c in candidates)}. Available .pt files in work dir: {available}"
    )


def _select_score_mode(
    outputs_known,
    outputs_open,
    prototypes,
    score_modes,
    normalization=None,
    gaussian_stats=None,
):
    unknown_mask = np.asarray(outputs_open["is_known"], dtype=bool) == 0
    results = []
    for score_mode in score_modes:
        try:
            known_scores, _ = compute_open_score(
                outputs_known,
                prototypes=prototypes,
                score_mode=score_mode,
                normalization=normalization,
                gaussian_stats=gaussian_stats,
            )
            open_scores, _ = compute_open_score(
                outputs_open,
                prototypes=prototypes,
                score_mode=score_mode,
                normalization=normalization,
                gaussian_stats=gaussian_stats,
            )
        except ValueError as exc:
            results.append(
                {
                    "score_mode": score_mode,
                    "auroc": float("nan"),
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            continue
        unknown_scores = open_scores[unknown_mask]
        labels = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
        scores = np.concatenate([known_scores, unknown_scores])
        auroc = compute_auroc(labels, scores)
        results.append(
            {
                "score_mode": score_mode,
                "auroc": auroc,
                "status": "ok",
            }
        )

    valid = [item for item in results if item["status"] == "ok" and np.isfinite(item["auroc"])]
    if not valid:
        raise ValueError("No usable score mode was available for automatic calibration.")
    valid.sort(key=lambda x: x["auroc"], reverse=True)
    return valid[0], results


def _score_needs_shared_mahalanobis(score_mode: str, auto_calibrate: bool = False) -> bool:
    if auto_calibrate or score_mode == "auto":
        return True
    return score_mode.endswith("_shared") or score_mode in {
        "mahalanobis",
        "entropy_mahalanobis",
        "normalized_entropy_mahalanobis",
    }


def fit_teacher(args):
    set_seed(args.seed)
    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=args.download,
        split_path=args.split_path,
        limit_train=args.limit_train or None,
        limit_val=args.limit_val or None,
        limit_test=args.limit_test or None,
        limit_discovery=args.limit_discovery or None,
        discovery_pool_mode=args.discovery_pool_mode,
        open_val_ratio=getattr(args, "open_val_ratio", 0.0),
    )
    device = resolve_device(args.device)
    print(f"device: {device}")
    train_loader = build_loader(bundle.train, args.batch_size, True, args.num_workers)
    val_loader = build_loader(bundle.val, args.batch_size, False, args.num_workers)
    model = build_model(
        len(bundle.known_classes),
        backbone=args.teacher_backbone or args.backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(args.epochs):
        stats = train_one_epoch_teacher(
            model,
            train_loader,
            optim,
            device,
            alpha_unc=args.alpha_unc,
            alpha_proto=args.alpha_proto,
            alpha_proxy=args.alpha_proxy,
            alpha_pseudo=args.alpha_pseudo,
            alpha_energy=args.alpha_energy,
            pseudo_mode=args.pseudo_mode,
            pseudo_feature_noise=args.pseudo_feature_noise,
            uncertainty_target_mode=args.uncertainty_target_mode,
            energy_margin=args.energy_margin,
            energy_temperature=args.energy_temperature,
            proxy_temperature=args.proxy_temperature,
        )
        val_stats = evaluate_classification(model, val_loader, device)
        print(f"[teacher][{epoch+1}/{args.epochs}] {stats} {val_stats}")
        if val_stats["known_acc"] > best_acc:
            best_acc = val_stats["known_acc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
    if best_state is not None:
        model.load_state_dict(best_state)
    run_dir = ensure_dir(args.work_dir)
    save_json(run_dir / "config.json", {**vars(args), "best_epoch": best_epoch, "best_val_known_acc": best_acc})
    proto_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
    prototypes = collect_prototypes(model, proto_loader, device, len(bundle.known_classes)).cpu()
    gaussian_stats = collect_diagonal_gaussian_stats(
        model, proto_loader, device, len(bundle.known_classes)
    )
    gaussian_stats = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in gaussian_stats.items()}
    ckpt_path = resolve_output_checkpoint(args.teacher_ckpt, args.work_dir, "teacher.pt")
    save_checkpoint(
        model,
        ckpt_path,
        extra={
            "known_classes": list(bundle.known_classes),
            "prototypes": prototypes,
            "gaussian_stats": gaussian_stats,
        },
    )
    torch.save({"known_classes": list(bundle.known_classes), "novel_classes": list(bundle.novel_classes)}, run_dir / "split.pt")
    print(f"saved to {ckpt_path} (best_epoch={best_epoch}, best_val_known_acc={best_acc:.4f})")


def fit_student(args):
    set_seed(args.seed)
    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=args.download,
        split_path=args.split_path,
        limit_train=args.limit_train or None,
        limit_val=args.limit_val or None,
        limit_test=args.limit_test or None,
        limit_discovery=args.limit_discovery or None,
        discovery_pool_mode=args.discovery_pool_mode,
        open_val_ratio=getattr(args, "open_val_ratio", 0.0),
    )
    device = resolve_device(args.device)
    print(f"device: {device}")
    train_loader = build_loader(bundle.train, args.batch_size, True, args.num_workers)
    val_loader = build_loader(bundle.val, args.batch_size, False, args.num_workers)
    discovery_loader = None
    uses_discovery_regularizer = (
        args.alpha_discovery_unknown > 0.0
        or args.alpha_discovery_energy > 0.0
        or args.alpha_discovery_selective_unknown > 0.0
        or args.alpha_discovery_selective_energy > 0.0
    )
    if uses_discovery_regularizer and not args.discovery_pool:
        raise ValueError("discovery unknown/energy regularizers require --discovery-pool.")
    if args.discovery_pool and bundle.discovery_pool is not None and len(bundle.discovery_pool) > 0:
        if args.discovery_pool_mode == "mixed" and (
            args.alpha_discovery_unknown > 0.0 or args.alpha_discovery_energy > 0.0
        ):
            raise ValueError(
                "alpha-discovery-unknown/energy requires --discovery-pool-mode unknown."
            )
        discovery_loader = build_loader(
            TwoViewDataset(bundle.discovery_pool),
            args.discovery_batch_size or args.batch_size,
            True,
            args.num_workers,
        )
    teacher_backbone = args.teacher_backbone or args.backbone
    student_backbone = args.student_backbone or args.backbone
    teacher = build_model(
        len(bundle.known_classes),
        backbone=teacher_backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    student = build_model(
        len(bundle.known_classes),
        backbone=student_backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    discovery_selection_model = None
    if args.discovery_selection_model == "ema":
        discovery_selection_model = copy.deepcopy(student).to(device)
        discovery_selection_model.eval()
        for parameter in discovery_selection_model.parameters():
            parameter.requires_grad_(False)
    teacher_ckpt = resolve_input_checkpoint(args.teacher_ckpt, args.work_dir, "teacher.pt")
    load_checkpoint(teacher, teacher_ckpt, device)
    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(args.epochs):
        selective_weight = scheduled_weight(
            epoch,
            warmup_epochs=args.discovery_selective_warmup_epochs,
            ramp_epochs=args.discovery_selective_ramp_epochs,
        )
        stats = train_one_epoch_student(
            student,
            teacher,
            train_loader,
            optim,
            device,
            alpha_unc=args.alpha_unc,
            alpha_kd=args.alpha_kd,
            alpha_feat_kd=args.alpha_feat_kd,
            alpha_supcon=args.alpha_supcon,
            alpha_proto=args.alpha_proto,
            alpha_proxy=args.alpha_proxy,
            alpha_pseudo=args.alpha_pseudo,
            alpha_energy=args.alpha_energy,
            pseudo_mode=args.pseudo_mode,
            pseudo_feature_noise=args.pseudo_feature_noise,
            uncertainty_target_mode=args.uncertainty_target_mode,
            temperature=args.temperature,
            kd_mode=args.kd_mode,
            uncertainty_weight_mode=args.uncertainty_weight_mode,
            discovery_loader=discovery_loader,
            alpha_discovery=args.alpha_discovery,
            alpha_discovery_unknown=args.alpha_discovery_unknown,
            alpha_discovery_energy=args.alpha_discovery_energy,
            alpha_discovery_selective_unknown=args.alpha_discovery_selective_unknown * selective_weight,
            alpha_discovery_selective_energy=args.alpha_discovery_selective_energy * selective_weight,
            discovery_select_ratio=args.discovery_select_ratio,
            discovery_select_mode=args.discovery_select_mode,
            discovery_loss_mode=args.discovery_loss,
            discovery_temperature=args.discovery_temperature,
            energy_margin=args.energy_margin,
            energy_temperature=args.energy_temperature,
            proxy_temperature=args.proxy_temperature,
            discovery_selection_model=discovery_selection_model,
            discovery_ema_decay=args.discovery_ema_decay,
            discovery_neighbor_filter=args.discovery_neighbor_filter,
            discovery_neighbor_k=args.discovery_neighbor_k,
            discovery_neighbor_min_votes=args.discovery_neighbor_min_votes,
        )
        stats["discovery_selective_weight"] = selective_weight
        val_stats = evaluate_classification(student, val_loader, device)
        print(f"[student][{epoch+1}/{args.epochs}] {stats} {val_stats}")
        if val_stats["known_acc"] > best_acc:
            best_acc = val_stats["known_acc"]
            best_state = copy.deepcopy(student.state_dict())
            best_epoch = epoch + 1
    if best_state is not None:
        student.load_state_dict(best_state)
    run_dir = ensure_dir(args.work_dir)
    save_json(run_dir / "config.json", {**vars(args), "best_epoch": best_epoch, "best_val_known_acc": best_acc})
    proto_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
    prototypes = collect_prototypes(student, proto_loader, device, len(bundle.known_classes)).cpu()
    gaussian_stats = collect_diagonal_gaussian_stats(
        student, proto_loader, device, len(bundle.known_classes)
    )
    gaussian_stats = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in gaussian_stats.items()}
    student_ckpt = resolve_output_checkpoint(args.student_ckpt, args.work_dir, "student.pt")
    save_checkpoint(
        student,
        student_ckpt,
        extra={
            "known_classes": list(bundle.known_classes),
            "prototypes": prototypes,
            "gaussian_stats": gaussian_stats,
        },
    )
    print(f"saved to {student_ckpt} (best_epoch={best_epoch}, best_val_known_acc={best_acc:.4f})")


def discover(args):
    set_seed(args.seed)
    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=args.download,
        split_path=args.split_path,
        limit_train=args.limit_train or None,
        limit_val=args.limit_val or None,
        limit_test=args.limit_test or None,
        limit_discovery=args.limit_discovery or None,
        discovery_pool_mode=args.discovery_pool_mode,
        open_val_ratio=args.open_val_ratio,
    )
    device = resolve_device(args.device)
    print(f"device: {device}")
    test_loader = build_loader(bundle.test, args.batch_size, False, args.num_workers)
    val_loader = build_loader(bundle.val, args.batch_size, False, args.num_workers)
    model = build_model(
        len(bundle.known_classes),
        backbone=args.student_backbone or args.backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    ckpt_path = resolve_input_checkpoint(args.student_ckpt, args.work_dir, "student.pt")
    ckpt = load_checkpoint(model, ckpt_path, device)
    outputs_test = extract_outputs(
        model,
        test_loader,
        device,
        mc_samples=args.mc_samples,
        odin_epsilon=args.odin_epsilon,
        odin_temperature=args.odin_temperature,
    )
    outputs_val = extract_outputs(
        model,
        val_loader,
        device,
        mc_samples=args.mc_samples,
        odin_epsilon=args.odin_epsilon,
        odin_temperature=args.odin_temperature,
    )
    open_val_loader = build_loader(bundle.open_val, args.batch_size, False, args.num_workers) if bundle.open_val is not None else None
    outputs_open_val = (
        extract_outputs(
            model,
            open_val_loader,
            device,
            mc_samples=args.mc_samples,
            odin_epsilon=args.odin_epsilon,
            odin_temperature=args.odin_temperature,
        )
        if open_val_loader is not None
        else None
    )
    temperature = fit_temperature(outputs_val) if args.temperature_calibration else 1.0
    if args.temperature_calibration:
        outputs_test = apply_temperature(outputs_test, temperature)
        outputs_val = apply_temperature(outputs_val, temperature)
        if outputs_open_val is not None:
            outputs_open_val = apply_temperature(outputs_open_val, temperature)
    proto = ckpt.get("prototypes")
    if proto is not None and torch.is_tensor(proto):
        proto = proto.detach().cpu().numpy()
    gaussian_stats = ckpt.get("gaussian_stats")
    if gaussian_stats is not None:
        gaussian_stats = {
            key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            for key, value in gaussian_stats.items()
        }
    if (
        gaussian_stats is None or "precision" not in gaussian_stats
    ) and _score_needs_shared_mahalanobis(args.score_mode, args.auto_calibrate_score):
        stats_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
        gaussian_stats = collect_diagonal_gaussian_stats(
            model, stats_loader, device, len(bundle.known_classes)
        )
    selected_score_mode = args.score_mode
    needs_normalization = selected_score_mode.startswith("normalized_")
    score_normalization = (
        fit_score_normalization(outputs_val, proto, gaussian_stats)
        if needs_normalization or args.score_mode == "auto" or args.auto_calibrate_score
        else {}
    )
    calibration_report = {
        "temperature": float(temperature),
        "temperature_enabled": bool(args.temperature_calibration),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "odin": {
            "epsilon": float(args.odin_epsilon),
            "temperature": float(args.odin_temperature),
            "enabled": bool(args.odin_epsilon > 0.0),
        },
        "reliability": calibration_diagnostics(
            outputs_val["probs"], outputs_val["labels"], args.calibration_bins
        ),
        "uncertainty_error": uncertainty_error_diagnostics(outputs_val),
    }
    threshold = None
    if (args.score_mode == "auto" or args.auto_calibrate_score) and outputs_open_val is not None:
        candidate_modes = [
            "full",
            "entropy_proto",
            "entropy_only",
            "proto_only",
            "max_softmax",
            "odin_msp",
            "energy",
            "normalized_entropy_proto",
            "entropy_epistemic",
            "expected_entropy",
            "mahalanobis",
            "mahalanobis_diag",
            "mahalanobis_shared",
            "entropy_mahalanobis",
            "entropy_mahalanobis_shared",
            "normalized_entropy_mahalanobis",
            "normalized_entropy_mahalanobis_shared",
        ]
        selected, all_candidates = _select_score_mode(
            outputs_val,
            outputs_open_val,
            proto,
            candidate_modes,
            normalization=score_normalization,
            gaussian_stats=gaussian_stats,
        )
        selected_score_mode = selected["score_mode"]
        scores_val, _ = compute_open_score(
            outputs_val,
            prototypes=proto,
            score_mode=selected_score_mode,
            normalization=score_normalization,
            gaussian_stats=gaussian_stats,
        )
        threshold = calibrate_threshold(scores_val, percentile=args.threshold_percentile)
        calibration_report.update(
            {
                "selected": selected,
                "candidates": all_candidates,
                "open_val_size": int(len(outputs_open_val["labels"])),
                "open_val_unknown_size": int(np.sum(np.asarray(outputs_open_val["is_known"], dtype=bool) == 0)),
                "threshold_policy": {
                    "type": "known_val_percentile",
                    "percentile": args.threshold_percentile,
                },
                "known_val_threshold": float(threshold),
            }
        )
    else:
        if selected_score_mode == "auto":
            selected_score_mode = "full"
        scores_val, _ = compute_open_score(
            outputs_val,
            prototypes=proto,
            score_mode=selected_score_mode,
            normalization=score_normalization,
            gaussian_stats=gaussian_stats,
        )
        threshold = calibrate_threshold(scores_val, percentile=args.threshold_percentile)
        calibration_report.update(
            {
                "threshold_policy": {
                    "type": "known_val_percentile",
                    "percentile": args.threshold_percentile,
                },
                "known_val_threshold": float(threshold),
            }
        )

    result, scores_test, pred_known, detail = run_discovery(
        outputs_test,
        threshold,
        args.num_novel,
        prototypes=proto,
        score_mode=selected_score_mode,
        normalization=score_normalization,
        gaussian_stats=gaussian_stats,
        cluster_k=args.cluster_k,
        cluster_method=args.cluster_method,
        cluster_feature=args.cluster_feature,
        cluster_selection=args.cluster_selection,
        cluster_pca_dim=args.cluster_pca_dim,
        cluster_normalize=args.cluster_normalize,
        cluster_whiten=not args.cluster_no_whiten,
        cluster_n_init=args.cluster_n_init,
        cluster_stability_repeats=args.cluster_stability_repeats,
    )
    run_dir = ensure_dir(args.work_dir)
    save_json(run_dir / "config.json", vars(args))
    torch.save(
        {
            "known_classes": list(bundle.known_classes),
            "novel_classes": list(bundle.novel_classes),
        },
        run_dir / "split.pt",
    )
    save_json(run_dir / "discovery_report.json", result)
    save_json(run_dir / "calibration_report.json", calibration_report)
    save_json(run_dir / "discovery_detail.json", {
        "score_mode": detail["score_mode"],
        "score": detail["score"].tolist(),
        "entropy": detail["entropy"].tolist(),
        "epistemic": detail["epistemic"].tolist(),
        "aleatoric": detail["aleatoric"].tolist(),
        "expected_entropy": detail["expected_entropy"].tolist(),
        "head_uncertainty": detail["head_uncertainty"].tolist(),
        "proto_dist": detail["proto_dist"].tolist(),
        "mahalanobis": detail["mahalanobis"].tolist(),
        "odin_msp": detail["odin_msp"].tolist(),
        "pred_known": detail["pred_known"].astype(int).tolist(),
        "true_known": detail["true_known"].astype(int).tolist(),
        "pred_class": detail["pred_class"].tolist(),
        "true_label": detail["true_label"].tolist(),
        "raw_labels": detail["raw_labels"].tolist(),
        "pred_cluster": None if detail["pred_cluster"] is None else detail["pred_cluster"].tolist(),
        "cluster_k": detail.get("cluster_k"),
        "cluster_method": detail.get("cluster_method"),
        "cluster_feature": detail.get("cluster_feature"),
        "cluster_selection": detail.get("cluster_selection"),
        "cluster_diagnostics": detail.get("cluster_diagnostics", []),
    })
    np.save(run_dir / "open_scores.npy", scores_test)
    print(result)
    if "selected" in calibration_report:
        selected = calibration_report["selected"]
        print(
            "calibration:",
            {
                "selected_score_mode": selected["score_mode"],
                "auroc": selected["auroc"],
                "known_val_threshold": calibration_report["known_val_threshold"],
                "threshold_policy": calibration_report["threshold_policy"]["type"],
            },
        )
    save_json(run_dir / "score_normalization.json", score_normalization)
    print(f"threshold={threshold:.6f}, temperature={temperature:.4f}")


def inspect_data(args):
    set_seed(args.seed)
    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=args.download,
        split_path=args.split_path,
        limit_train=args.limit_train or None,
        limit_val=args.limit_val or None,
        limit_test=args.limit_test or None,
        limit_discovery=args.limit_discovery or None,
        discovery_pool_mode=args.discovery_pool_mode,
    )
    print("dataset:", args.dataset)
    print("known classes:", len(bundle.known_classes))
    print("novel classes:", len(bundle.novel_classes))
    print("train size:", len(bundle.train))
    print("val size:", len(bundle.val))
    print("test size:", len(bundle.test))
    print("discovery pool size:", len(bundle.discovery_pool) if bundle.discovery_pool is not None else 0)
    print("first known classes:", list(bundle.known_classes)[: min(args.sample_count, len(bundle.known_classes))])
    print("first novel classes:", list(bundle.novel_classes)[: min(args.sample_count, len(bundle.novel_classes))])


def main():
    args = parse_args()
    if args.command == "train_teacher":
        fit_teacher(args)
    elif args.command == "train_student":
        fit_student(args)
    elif args.command == "discover":
        discover(args)
    elif args.command == "inspect_data":
        inspect_data(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
