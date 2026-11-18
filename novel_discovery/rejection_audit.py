"""Audits and utilities for controlled rejection-encoder comparisons.

The initialization experiment must reuse one trained student classification
model and vary only the independent rejection encoder.  This module keeps the
parameter-freezing, optimizer, state-snapshot, and gradient checks in one
place so an experiment can fail loudly when the intended protocol is not
actually being followed.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
from typing import Any

import torch
from torch import nn


CLASSIFICATION_MODULE_NAMES = (
    "encoder",
    "classifier",
    "projector",
    "uncertainty_head",
    "structure_projector",
)

REJECTION_MODULE_NAMES = (
    "rejection_encoder",
    "rejection_input_norm",
    "rejection_projector",
    "rejection_classifier",
    "rejection_uncertainty_head",
)


def _belongs(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def select_state_dict_prefixes(
    state_dict: dict[str, torch.Tensor], prefixes: Iterable[str]
) -> dict[str, torch.Tensor]:
    """Return a deterministic subset of a state dict by module prefix."""
    prefixes = tuple(prefixes)
    return {
        name: value
        for name, value in state_dict.items()
        if _belongs(name, prefixes)
    }


def state_dict_structure(state_dict: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Describe state-dict keys and tensor shapes without exposing values."""
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in sorted(state_dict.items())
    ]


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash names, shapes, dtypes, and tensor bytes in a stable order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def compare_state_dicts_exact(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Compare two state-dict subsets for exact key, shape, and value equality."""
    first_keys = set(first)
    second_keys = set(second)
    common = sorted(first_keys & second_keys)
    shape_mismatches = []
    value_mismatches = []
    for name in common:
        if first[name].shape != second[name].shape or first[name].dtype != second[name].dtype:
            shape_mismatches.append(name)
        elif not torch.equal(first[name].detach().cpu(), second[name].detach().cpu()):
            value_mismatches.append(name)
    return {
        "exact": not (
            first_keys - second_keys
            or second_keys - first_keys
            or shape_mismatches
            or value_mismatches
        ),
        "missing_names": sorted(first_keys - second_keys),
        "added_names": sorted(second_keys - first_keys),
        "shape_mismatch_names": shape_mismatches,
        "value_mismatch_names": value_mismatches,
        "first_sha256": state_dict_sha256(first),
        "second_sha256": state_dict_sha256(second),
    }


def named_module_parameters(model: nn.Module, module_names: Iterable[str]):
    """Yield parameters whose names belong to one of ``module_names``."""
    prefixes = tuple(module_names)
    for name, parameter in model.named_parameters():
        if _belongs(name, prefixes):
            yield name, parameter


def freeze_classification_modules(model: nn.Module) -> dict[str, list[str]]:
    """Freeze classification modules and enable the rejection modules.

    The function does not change train/eval mode.  BatchNorm mode is a separate
    protocol decision and is audited independently by ``batchnorm_snapshot``.
    """
    classification_names: list[str] = []
    rejection_names: list[str] = []
    classification_prefixes = CLASSIFICATION_MODULE_NAMES
    rejection_prefixes = REJECTION_MODULE_NAMES
    for name, parameter in model.named_parameters():
        if _belongs(name, classification_prefixes):
            parameter.requires_grad_(False)
            classification_names.append(name)
        elif _belongs(name, rejection_prefixes):
            parameter.requires_grad_(True)
            rejection_names.append(name)
    if not classification_names:
        raise ValueError("model has no recognized classification parameters")
    if not rejection_names:
        raise ValueError("model has no recognized rejection parameters")
    return {
        "classification_parameters": classification_names,
        "rejection_parameters": rejection_names,
    }


def sync_shared_backbone_rejection_encoder(model: nn.Module) -> dict[str, Any]:
    """Copy the student encoder into the independent rejection encoder.

    ``copy.deepcopy`` is used by the model constructor, but this explicit copy
    is useful immediately before an audit or a treatment initialization.  It
    verifies that both encoders have identical named state tensors and that the
    copy is independent in memory.
    """
    encoder = getattr(model, "encoder", None)
    rejection_encoder = getattr(model, "rejection_encoder", None)
    if encoder is None or rejection_encoder is None:
        raise ValueError("shared-backbone initialization requires both encoders")
    source = encoder.state_dict()
    target = rejection_encoder.state_dict()
    if list(source.keys()) != list(target.keys()):
        raise ValueError(
            "classification and rejection encoders do not have matching state keys"
        )
    rejection_encoder.load_state_dict(source)
    same_values = all(torch.equal(source[name], rejection_encoder.state_dict()[name]) for name in source)
    aliased = any(
        source[name].data_ptr() == rejection_encoder.state_dict()[name].data_ptr()
        for name in source
        if torch.is_tensor(source[name])
    )
    if not same_values or aliased:
        raise AssertionError("shared-backbone copy is not an independent equal copy")
    return {
        "state_keys": list(source.keys()),
        "equal_after_copy": bool(same_values),
        "parameter_storage_aliased": bool(aliased),
    }


def build_rejection_only_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.0,
    optimizer_cls= torch.optim.AdamW,
):
    """Freeze classification parameters and build an optimizer for rejection only."""
    freeze_report = freeze_classification_modules(model)
    rejection_parameters = [
        parameter
        for _, parameter in named_module_parameters(model, REJECTION_MODULE_NAMES)
        if parameter.requires_grad
    ]
    classification_ids = {
        id(parameter)
        for _, parameter in named_module_parameters(model, CLASSIFICATION_MODULE_NAMES)
    }
    if any(id(parameter) in classification_ids for parameter in rejection_parameters):
        raise AssertionError("classification and rejection parameter sets overlap")
    if not rejection_parameters:
        raise ValueError("rejection-only optimizer received no trainable parameters")
    optimizer = optimizer_cls(rejection_parameters, lr=lr, weight_decay=weight_decay)
    optimizer_report = audit_optimizer_parameters(optimizer, model)
    if optimizer_report["unexpected_parameter_names"]:
        raise AssertionError(
            "rejection-only optimizer contains non-rejection parameters: "
            + ", ".join(optimizer_report["unexpected_parameter_names"])
        )
    return optimizer, {"freeze": freeze_report, "optimizer": optimizer_report}


def audit_optimizer_parameters(optimizer: torch.optim.Optimizer, model: nn.Module) -> dict[str, Any]:
    """Report exactly which named model parameters occur in an optimizer."""
    parameter_to_name = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_names = [parameter_to_name.get(identifier, f"<external:{identifier}>") for identifier in optimizer_ids]
    allowed = set(REJECTION_MODULE_NAMES)
    unexpected = sorted(
        name for name in optimizer_names if not name.startswith("<external:") and not _belongs(name, allowed)
    )
    classification = sorted(
        name for name in optimizer_names if not name.startswith("<external:") and _belongs(name, CLASSIFICATION_MODULE_NAMES)
    )
    return {
        "optimizer_parameter_count": len(optimizer_ids),
        "optimizer_parameter_names": sorted(optimizer_names),
        "unexpected_parameter_names": unexpected,
        "classification_parameter_names": classification,
        "all_parameters_belong_to_model": all(not name.startswith("<external:") for name in optimizer_names),
    }


def parameter_snapshot(model: nn.Module, *, include_buffers: bool = False) -> dict[str, torch.Tensor]:
    """Clone parameters, optionally including buffers, for before/after audits."""
    snapshot = {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}
    if include_buffers:
        snapshot.update({name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()})
    return snapshot


def compare_parameter_snapshots(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], *, atol: float = 0.0
) -> dict[str, Any]:
    """Summarize changed, missing, and added tensors in two snapshots."""
    before_keys = set(before)
    after_keys = set(after)
    common = sorted(before_keys & after_keys)
    changed = []
    max_abs_delta = 0.0
    for name in common:
        delta = (after[name] - before[name]).abs()
        value = float(delta.max().item()) if delta.numel() else 0.0
        max_abs_delta = max(max_abs_delta, value)
        if value > atol:
            changed.append(name)
    return {
        "changed_names": changed,
        "changed_count": len(changed),
        "missing_names": sorted(before_keys - after_keys),
        "added_names": sorted(after_keys - before_keys),
        "max_abs_delta": max_abs_delta,
    }


def batchnorm_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture BatchNorm running state for explicit protocol comparison."""
    return {
        name: buffer.detach().cpu().clone()
        for name, buffer in model.named_buffers()
        if name.endswith("running_mean")
        or name.endswith("running_var")
        or name.endswith("num_batches_tracked")
    }


def gradient_audit(model: nn.Module) -> dict[str, Any]:
    """Summarize gradient presence and magnitude by classification/rejection path."""
    rows = []
    for name, parameter in model.named_parameters():
        grad_norm = None if parameter.grad is None else float(parameter.grad.detach().norm().item())
        rows.append(
            {
                "name": name,
                "requires_grad": bool(parameter.requires_grad),
                "has_gradient": parameter.grad is not None,
                "grad_norm": grad_norm,
                "path": (
                    "classification"
                    if _belongs(name, CLASSIFICATION_MODULE_NAMES)
                    else "rejection"
                    if _belongs(name, REJECTION_MODULE_NAMES)
                    else "other"
                ),
            }
        )
    return {
        "parameters": rows,
        "classification_trainable": any(
            row["requires_grad"] for row in rows if row["path"] == "classification"
        ),
        "rejection_has_gradient": any(
            row["has_gradient"] and (row["grad_norm"] or 0.0) > 0.0
            for row in rows
            if row["path"] == "rejection"
        ),
    }
