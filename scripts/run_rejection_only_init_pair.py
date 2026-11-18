"""Run a rejection-only initialization comparison from one fixed classifier.

The protocol loads the classification part of one existing student checkpoint,
freezes that path, and trains only the independent rejection path.  The two
arms differ only in rejection-encoder initialization:

The default pair is deliberately same-architecture:

* ``random_backbone``: an independent copy of the classification backbone
  with random initialization;
* ``shared_backbone``: an independent copy initialized from the student
  classification backbone.

``scratch_cnn`` remains available as an architecture reference, but a
``scratch_cnn``/``shared_backbone`` comparison changes both architecture and
initialization and must not be interpreted as a pure initialization result.

This script deliberately uses a known-class cross-entropy objective first.
It isolates the initialization factor before adding NNPU or other objectives.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from novel_discovery.models import freeze_batchnorm_stats, build_model  # noqa: E402
from novel_discovery.pipeline import build_loader, evaluate_classification  # noqa: E402
from novel_discovery.rejection_audit import (  # noqa: E402
    batchnorm_snapshot,
    build_rejection_only_optimizer,
    compare_parameter_snapshots,
    compare_state_dicts_exact,
    gradient_audit,
    parameter_snapshot,
    select_state_dict_prefixes,
    state_dict_structure,
    sync_shared_backbone_rejection_encoder,
)
from novel_discovery.utils import save_json, set_seed  # noqa: E402
from train import (  # noqa: E402
    build_bundle_from_args,
    load_checkpoint,
    parse_args,
    resolve_device,
    save_checkpoint,
    save_command_config,
)


def _load_source_args(config_path: Path, split_path: Path | None):
    values = json.loads(config_path.read_text(encoding="utf-8"))
    args = parse_args(["train_student"])
    for key, value in values.items():
        if hasattr(args, key):
            setattr(args, key, value)
    if split_path is not None:
        args.split_path = str(split_path)
    args.command = "train_student"
    return args


def _path_snapshot(snapshot: dict[str, torch.Tensor], prefixes: tuple[str, ...]):
    return {
        name: value
        for name, value in snapshot.items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }


def _tensor_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.detach().cpu() - second.detach().cpu()).abs().max().item())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight_initialization_pair(
    *, source_args, checkpoint: Path, bundle, val_loader, device, init_modes, seed: int
):
    """Verify the causal comparison before any rejection training starts."""
    arms = {}
    classification_prefixes = (
        "encoder",
        "classifier",
        "projector",
        "uncertainty_head",
        "structure_projector",
    )
    rejection_prefixes = (
        "rejection_encoder",
        "rejection_input_norm",
        "rejection_projector",
        "rejection_classifier",
        "rejection_uncertainty_head",
    )
    for init_mode in init_modes:
        model, optimizer, optimizer_report, copy_report = _build_arm(
            source_args, checkpoint, bundle, device, init_mode, seed
        )
        state_dict = model.state_dict()
        classification_state = select_state_dict_prefixes(state_dict, classification_prefixes)
        rejection_state = select_state_dict_prefixes(state_dict, rejection_prefixes)
        classification_probe = _classification_probe(model, val_loader, device)
        arms[init_mode] = {
            "classification_state": classification_state,
            "rejection_state": rejection_state,
            "classification_probe": classification_probe,
            "classification_structure": state_dict_structure(classification_state),
            "rejection_structure": state_dict_structure(rejection_state),
            "optimizer_report": optimizer_report,
            "shared_copy": copy_report,
            "classification_trainable": any(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if any(name == prefix or name.startswith(prefix + ".") for prefix in classification_prefixes)
            ),
        }

    reference_name = init_modes[0]
    reference = arms[reference_name]
    reports = {
        "classification_state": {},
        "classification_probe": {},
        "rejection_structure": {},
    }
    failures = []
    for init_mode in init_modes[1:]:
        current = arms[init_mode]
        state_report = compare_state_dicts_exact(
            reference["classification_state"], current["classification_state"]
        )
        probe_report = {
            "logits_max_abs": _tensor_difference(
                reference["classification_probe"]["logits"],
                current["classification_probe"]["logits"],
            ),
            "features_max_abs": _tensor_difference(
                reference["classification_probe"]["features"],
                current["classification_probe"]["features"],
            ),
        }
        structure_report = {
            "same": reference["rejection_structure"] == current["rejection_structure"],
            "reference": reference["rejection_structure"],
            "current": current["rejection_structure"],
        }
        reports["classification_state"][init_mode] = state_report
        reports["classification_probe"][init_mode] = probe_report
        reports["rejection_structure"][init_mode] = structure_report
        if not state_report["exact"]:
            failures.append(f"classification state differs between {reference_name} and {init_mode}")
        if probe_report["logits_max_abs"] != 0.0 or probe_report["features_max_abs"] != 0.0:
            failures.append(f"classification probe differs between {reference_name} and {init_mode}")
        if not structure_report["same"]:
            failures.append(f"rejection encoder structure differs between {reference_name} and {init_mode}")

    for init_mode, arm in arms.items():
        if arm["classification_trainable"]:
            failures.append(f"classification parameters remain trainable in {init_mode}")
        if arm["optimizer_report"]["optimizer"]["classification_parameter_names"]:
            failures.append(f"optimizer contains classification parameters in {init_mode}")

    report = {
        "passed": not failures,
        "factor": "rejection_encoder_initialization",
        "reference_arm": reference_name,
        "arms": list(init_modes),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "classification_checkpoint": str(checkpoint),
        "fixed_classification_state": True,
        "fixed_classification_outputs": True,
        "same_rejection_structure": True,
        "classification_parameters_frozen": True,
        "only_intended_difference": list(init_modes) == ["random_backbone", "shared_backbone"],
        "comparisons": reports,
        "failures": failures,
    }
    if failures:
        raise RuntimeError(
            "rejection initialization preflight failed; no training was started: "
            + "; ".join(failures)
        )
    return report


@torch.no_grad()
def _classification_probe(model, loader, device):
    model.eval()
    for batch in loader:
        images = batch[0].to(device)
        outputs = model(images)
        return {
            "logits": outputs["logits"].detach().cpu(),
            "features": outputs["features"].detach().cpu(),
        }
    raise RuntimeError("cannot create a classification probe from an empty loader")


def _build_arm(source_args, checkpoint, bundle, device, init_mode: str, seed: int):
    # Re-seeding before each construction makes the rejection heads identical
    # across arms.  The independent encoder itself is the intended treatment.
    set_seed(seed)
    model = build_model(
        len(bundle.known_classes),
        backbone=source_args.student_backbone or source_args.backbone,
        proj_dim=source_args.proj_dim,
        dropout=source_args.dropout,
        pretrained=source_args.pretrained,
        cifar_stem=source_args.cifar_stem,
        rejection_feature_dim=source_args.rejection_feature_dim,
        independent_rejection_encoder=True,
        independent_rejection_encoder_init=init_mode,
        independent_rejection_input_norm=source_args.independent_rejection_input_norm,
        structure_feature_dim=source_args.structure_feature_dim,
    ).to(device)
    load_checkpoint(
        model,
        checkpoint,
        device,
        expected_known_classes=bundle.known_classes,
        expected_novel_classes=bundle.novel_classes,
        load_mode="classification_only",
    )
    copy_report = None
    if init_mode == "shared_backbone":
        copy_report = sync_shared_backbone_rejection_encoder(model)
    optimizer, optimizer_report = build_rejection_only_optimizer(
        model,
        lr=float(source_args.lr),
        weight_decay=float(source_args.weight_decay),
    )
    # Keep every BatchNorm running statistic fixed.  Affine parameters remain
    # trainable if they belong to the rejection path and are in the optimizer.
    model.train()
    freeze_batchnorm_stats(model)
    return model, optimizer, optimizer_report, copy_report


def _run_arm(
    *,
    source_args,
    checkpoint: Path,
    bundle,
    train_loader,
    val_loader,
    device,
    init_mode: str,
    seed: int,
    epochs: int,
    limit_train_batches: int,
):
    model, optimizer, optimizer_report, copy_report = _build_arm(
        source_args, checkpoint, bundle, device, init_mode, seed
    )
    model.train()
    freeze_batchnorm_stats(model)
    before_params = parameter_snapshot(model)
    before_bn = batchnorm_snapshot(model)
    initial_probe = _classification_probe(model, val_loader, device)

    history = []
    last_gradient_report = None
    for epoch in range(1, epochs + 1):
        model.train()
        freeze_batchnorm_stats(model)
        total_loss = 0.0
        total_count = 0
        for batch_index, batch in enumerate(train_loader):
            if limit_train_batches > 0 and batch_index >= limit_train_batches:
                break
            images, labels = batch[0].to(device), batch[1].to(device)
            valid = labels >= 0
            if not valid.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images[valid])
            if outputs["rejection_logits"] is None:
                raise RuntimeError("rejection-only protocol requires rejection logits")
            loss = F.cross_entropy(outputs["rejection_logits"], labels[valid])
            loss.backward()
            if last_gradient_report is None:
                last_gradient_report = gradient_audit(model)
            optimizer.step()
            count = int(valid.sum().item())
            total_loss += float(loss.detach().item()) * count
            total_count += count
        val_stats = evaluate_classification(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "rejection_ce": total_loss / max(total_count, 1),
                "train_samples": total_count,
                "val_known_acc": float(val_stats["known_acc"]),
            }
        )

    after_params = parameter_snapshot(model)
    after_bn = batchnorm_snapshot(model)
    final_probe = _classification_probe(model, val_loader, device)
    classification_delta = compare_parameter_snapshots(
        _path_snapshot(before_params, ("encoder", "classifier", "projector", "uncertainty_head", "structure_projector")),
        _path_snapshot(after_params, ("encoder", "classifier", "projector", "uncertainty_head", "structure_projector")),
    )
    rejection_delta = compare_parameter_snapshots(
        _path_snapshot(
            before_params,
            (
                "rejection_encoder",
                "rejection_input_norm",
                "rejection_projector",
                "rejection_classifier",
                "rejection_uncertainty_head",
            ),
        ),
        _path_snapshot(
            after_params,
            (
                "rejection_encoder",
                "rejection_input_norm",
                "rejection_projector",
                "rejection_classifier",
                "rejection_uncertainty_head",
            ),
        ),
    )
    bn_delta = compare_parameter_snapshots(before_bn, after_bn)
    return {
        "init_mode": init_mode,
        "optimizer": optimizer_report,
        "shared_copy": copy_report,
        "gradient_audit_first_batch": last_gradient_report,
        "classification_parameter_delta": classification_delta,
        "rejection_parameter_delta": rejection_delta,
        "batchnorm_state_delta": bn_delta,
        "classification_probe": {
            "initial_vs_final_logits_max_abs": _tensor_difference(
                initial_probe["logits"], final_probe["logits"]
            ),
            "initial_vs_final_features_max_abs": _tensor_difference(
                initial_probe["features"], final_probe["features"]
            ),
        },
        "history": history,
        "final_known_val": history[-1]["val_known_acc"] if history else None,
        "state_dict": model.state_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2081)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument(
        "--init-modes",
        nargs="+",
        choices=("random_backbone", "shared_backbone", "scratch_cnn"),
        default=("random_backbone", "shared_backbone"),
        help=(
            "Rejection encoder treatments. The default pair keeps the "
            "architecture fixed and changes only initialization."
        ),
    )
    options = parser.parse_args()

    source_config = options.source_config if options.source_config.is_absolute() else ROOT / options.source_config
    source_checkpoint = options.source_checkpoint if options.source_checkpoint.is_absolute() else ROOT / options.source_checkpoint
    split_path = options.split_path
    if split_path is not None and not split_path.is_absolute():
        split_path = ROOT / split_path
    output_root = options.output_root if options.output_root.is_absolute() else ROOT / options.output_root
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if split_path is not None and not split_path.is_file():
        raise FileNotFoundError(split_path)
    if options.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if len(set(options.init_modes)) != len(options.init_modes):
        raise ValueError("--init-modes must not contain duplicates")
    if len(options.init_modes) < 2:
        raise ValueError("at least two --init-modes are required")
    if set(options.init_modes) != {"random_backbone", "shared_backbone"}:
        raise ValueError(
            "this entry point only permits the causal initialization pair "
            "random_backbone/shared_backbone; use a separate exploratory script "
            "for architecture comparisons"
        )

    source_args = _load_source_args(source_config, split_path)
    source_args.seed = int(options.seed)
    source_args.model_seed = None
    if options.limit_train > 0:
        source_args.limit_train = int(options.limit_train)
    if options.limit_val > 0:
        source_args.limit_val = int(options.limit_val)
    source_args.num_workers = 0
    source_args.device = "auto"
    source_args.work_dir = str(output_root)
    source_args.independent_rejection_encoder = True
    source_args.rejection_feature_dim = int(source_args.rejection_feature_dim)
    if source_args.rejection_feature_dim <= 0:
        raise ValueError("source config must enable --rejection-feature-dim")

    set_seed(options.seed)
    bundle = build_bundle_from_args(source_args)
    device = resolve_device(source_args.device)
    train_loader = build_loader(bundle.train, source_args.batch_size, True, source_args.num_workers)
    val_loader = build_loader(bundle.val, source_args.batch_size, False, source_args.num_workers)
    if len(bundle.train) == 0 or len(bundle.val) == 0:
        raise ValueError("training and validation datasets must be non-empty")

    preflight = _preflight_initialization_pair(
        source_args=source_args,
        checkpoint=source_checkpoint,
        bundle=bundle,
        val_loader=val_loader,
        device=device,
        init_modes=tuple(options.init_modes),
        seed=options.seed,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    save_json(output_root / "preflight.json", preflight)

    results = {}
    arm_states = {}
    for init_mode in options.init_modes:
        arm_dir = output_root / init_mode
        arm_dir.mkdir(parents=True, exist_ok=False)
        result = _run_arm(
            source_args=source_args,
            checkpoint=source_checkpoint,
            bundle=bundle,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            init_mode=init_mode,
            seed=options.seed,
            epochs=options.epochs,
            limit_train_batches=options.limit_train_batches,
        )
        arm_states[init_mode] = result.pop("state_dict")
        save_checkpoint(
            type("CheckpointModel", (), {"state_dict": lambda self, state=arm_states[init_mode]: state})(),
            arm_dir / "rejection_only.pt",
            extra={
                "known_classes": list(bundle.known_classes),
                "novel_classes": list(bundle.novel_classes),
                "protocol": "fixed_classification_rejection_only_ce",
                "init_mode": init_mode,
            },
        )
        save_json(arm_dir / "audit.json", result)
        save_json(arm_dir / "history.json", {"history": result["history"]})
        results[init_mode] = result

    # The fixed classifier must produce identical outputs before training in
    # both arms.  The arm-level probe also confirms each arm preserved them.
    summary = {
        "protocol": {
            "factor": (
                "same_architecture_initialization"
                if set(options.init_modes) == {"random_backbone", "shared_backbone"}
                else "rejection_encoder_treatment"
            ),
            "arms": list(options.init_modes),
            "same_architecture_initialization_pair": (
                set(options.init_modes) == {"random_backbone", "shared_backbone"}
            ),
            "classification_checkpoint": str(source_checkpoint),
            "classification_frozen": True,
            "objective": "known_class_rejection_cross_entropy_only",
            "batchnorm_running_statistics_frozen": True,
            "seed": int(options.seed),
            "epochs": int(options.epochs),
            "preflight_passed": bool(preflight["passed"]),
            "only_intended_difference": bool(preflight["only_intended_difference"]),
        },
        "dataset": {
            "train_samples": len(bundle.train),
            "val_samples": len(bundle.val),
            "known_classes": len(bundle.known_classes),
            "novel_classes": len(bundle.novel_classes),
        },
        "arms": results,
        "preflight": preflight,
    }
    save_json(output_root / "summary.json", summary)
    save_command_config(output_root, "rejection_only_init_pair", vars(source_args))
    print(json.dumps({"output_root": str(output_root), "summary": str(output_root / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
