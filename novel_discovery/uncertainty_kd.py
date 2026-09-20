from __future__ import annotations

import torch
from torch.nn import functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def uncertainty_weights(
    uncertainty: torch.Tensor,
    mode: str = "raw",
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert teacher uncertainty into detached per-sample KD weights."""
    weight = torch.exp(-uncertainty.detach())
    if mode == "raw":
        pass
    elif mode == "mean_normalized":
        if weight.numel() > 0:
            weight = weight / weight.mean().clamp_min(eps)
    else:
        raise ValueError(f"Unsupported uncertainty weight mode: {mode}")
    if clamp_min is not None or clamp_max is not None:
        min_value = -float("inf") if clamp_min is None else float(clamp_min)
        max_value = float("inf") if clamp_max is None else float(clamp_max)
        if min_value > max_value:
            raise ValueError("clamp_min must be less than or equal to clamp_max.")
        weight = weight.clamp(min=min_value, max=max_value)
    return weight


def per_sample_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Return Hinton-style temperature KL loss for each sample."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape.")
    if student_logits.numel() == 0:
        return student_logits.new_empty(0)
    temperature = max(float(temperature), 1e-6)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * (
        temperature**2
    )


def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
    temperature: float = 2.0,
    uncertainty_weighted: bool = True,
    uncertainty_weight_mode: str = "raw",
    uncertainty_weight_min: float | None = None,
    uncertainty_weight_max: float | None = None,
) -> torch.Tensor:
    """Average standard or uncertainty-weighted KL distillation loss."""
    per_sample = per_sample_kl_distillation_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
    )
    if per_sample.numel() == 0:
        return _zero_loss(student_logits)
    if teacher_uncertainty is not None and uncertainty_weighted:
        if teacher_uncertainty.numel() != per_sample.numel():
            raise ValueError("teacher_uncertainty must match the batch size.")
        weight = uncertainty_weights(
            teacher_uncertainty,
            mode=uncertainty_weight_mode,
            clamp_min=uncertainty_weight_min,
            clamp_max=uncertainty_weight_max,
        )
        per_sample = per_sample * weight
    return per_sample.mean()


def feature_distillation_loss(
    student_projection: torch.Tensor,
    teacher_projection: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
    uncertainty_weight_mode: str = "raw",
    uncertainty_weight_min: float | None = None,
    uncertainty_weight_max: float | None = None,
) -> torch.Tensor:
    """Distill normalized representations with optional uncertainty weights."""
    if student_projection.shape != teacher_projection.shape:
        raise ValueError("student_projection and teacher_projection must have the same shape.")
    if student_projection.numel() == 0:
        return _zero_loss(student_projection)
    per_sample = 1.0 - F.cosine_similarity(
        student_projection,
        teacher_projection.detach(),
        dim=-1,
    )
    if teacher_uncertainty is not None:
        if teacher_uncertainty.numel() != per_sample.numel():
            raise ValueError("teacher_uncertainty must match the batch size.")
        weight = uncertainty_weights(
            teacher_uncertainty,
            mode=uncertainty_weight_mode,
            clamp_min=uncertainty_weight_min,
            clamp_max=uncertainty_weight_max,
        )
        per_sample = per_sample * weight
    return per_sample.mean()


def uncertainty_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mode: str = "confidence",
) -> torch.Tensor:
    """Build detached uncertainty calibration targets on known samples."""
    if logits.size(0) != labels.numel():
        raise ValueError("labels must match the logits batch size.")
    if logits.numel() == 0:
        return logits.new_empty(0)
    probs = logits.softmax(dim=-1)
    true_conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    if mode == "confidence":
        return (1.0 - true_conf).detach()
    if mode == "classification_error":
        return (logits.argmax(dim=-1) != labels).float().detach()
    if mode == "margin":
        true_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = logits.detach().clone()
        other_logits.scatter_(1, labels.unsqueeze(1), float("-inf"))
        strongest_other = other_logits.max(dim=-1).values
        return torch.sigmoid(strongest_other - true_logit.detach())
    raise ValueError(f"Unsupported uncertainty target mode: {mode}")


def uncertainty_calibration_loss(
    uncertainty: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mode: str = "confidence",
) -> torch.Tensor:
    """Calibrate the auxiliary uncertainty head against detached targets."""
    if uncertainty.numel() != labels.numel():
        raise ValueError("uncertainty must match the batch size.")
    if uncertainty.numel() == 0:
        return _zero_loss(uncertainty)
    target = uncertainty_targets(logits, labels, mode=target_mode)
    if target_mode == "classification_error":
        return F.binary_cross_entropy(uncertainty.clamp(1e-6, 1.0 - 1e-6), target)
    return F.mse_loss(uncertainty, target)
