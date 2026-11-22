from __future__ import annotations

import torch
from torch.nn import functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def uncertainty_weights(
    uncertainty: torch.Tensor,
    mode: str = "raw",
    uncertainty_weight_min: float | None = None,
    uncertainty_weight_max: float | None = None,
) -> torch.Tensor:
    """Convert detached teacher uncertainty into per-sample KD weights."""
    weights = torch.exp(-uncertainty.detach())
    if mode == "mean_normalized":
        weights = weights / weights.mean().clamp_min(1e-6)
    elif mode != "raw":
        raise ValueError(f"Unsupported uncertainty weight mode: {mode}")
    if uncertainty_weight_min is not None or uncertainty_weight_max is not None:
        min_value = (
            -float("inf")
            if uncertainty_weight_min is None
            else float(uncertainty_weight_min)
        )
        max_value = (
            float("inf")
            if uncertainty_weight_max is None
            else float(uncertainty_weight_max)
        )
        if min_value > max_value:
            raise ValueError("uncertainty_weight_min must be <= uncertainty_weight_max")
        weights = weights.clamp(min=min_value, max=max_value)
    return weights


def per_sample_kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Return temperature-scaled KL loss for each sample."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape")
    if student_logits.numel() == 0:
        return student_logits.new_empty((student_logits.size(0),))
    temperature = max(float(temperature), 1e-6)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * (
        temperature**2
    )


def teacher_quality_weights(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    floor: float = 0.25,
    power: float = 1.0,
    mode: str = "mean_normalized",
) -> torch.Tensor:
    """Build detached KD weights from labelled teacher quality.

    The training loader contains known-class labels, so the teacher's
    probability assigned to the ground-truth class is an admissible quality
    signal.  It is deliberately separate from the teacher uncertainty head:
    a high uncertainty value is not assumed to be a calibrated estimate of
    teacher error.  The floor prevents a few hard known samples from removing
    the KD signal entirely; mean normalization keeps the expected KD scale
    comparable with the standard objective.
    """
    if teacher_logits.ndim != 2:
        raise ValueError("teacher_logits must have shape [batch_size, classes]")
    if labels.shape != (teacher_logits.size(0),):
        raise ValueError("labels must have shape [batch_size]")
    if teacher_logits.size(0) == 0:
        return teacher_logits.new_empty((0,))
    floor = float(floor)
    power = float(power)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("quality floor must be in [0, 1]")
    if power <= 0.0:
        raise ValueError("quality power must be positive")
    if bool(((labels < 0) | (labels >= teacher_logits.size(1))).any()):
        raise ValueError("labels must index the logits class dimension")
    quality = F.softmax(teacher_logits.detach(), dim=-1).gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)
    weights = quality.clamp_min(floor).pow(power)
    if mode == "mean_normalized":
        weights = weights / weights.mean().clamp_min(1e-6)
    elif mode != "raw":
        raise ValueError(f"Unsupported quality weight mode: {mode}")
    return weights.detach()


def quality_aware_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    quality_floor: float = 0.25,
    quality_power: float = 1.0,
    quality_weight_mode: str = "mean_normalized",
) -> torch.Tensor:
    """Weight full logits KD by detached, label-verified teacher quality.

    This objective is an opt-in alternative to uncertainty weighting.  The
    teacher's uncertainty head is not used, and the ground-truth labels only
    affect the weight, while the standard CE loss remains the source of hard
    target supervision.
    """
    per_sample = per_sample_kl_distillation_loss(
        student_logits, teacher_logits, temperature=temperature
    )
    if per_sample.numel() == 0:
        return _zero_loss(student_logits)
    weights = teacher_quality_weights(
        teacher_logits,
        labels,
        floor=quality_floor,
        power=quality_power,
        mode=quality_weight_mode,
    )
    return (per_sample * weights).mean()


def selective_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distill only samples the teacher classifies correctly.

    The known training label is used only to decide whether the detached
    teacher soft target is admissible.  Returning the selected fraction keeps
    the hard gate auditable and avoids conflating it with uncertainty weights.
    A batch with no correct teacher prediction contributes a finite zero loss.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have matching [batch, classes] shapes")
    if labels.shape != (student_logits.size(0),):
        raise ValueError("labels must have shape [batch_size]")
    if student_logits.size(0) == 0:
        return _zero_loss(student_logits), student_logits.new_tensor(0.0)
    if bool(((labels < 0) | (labels >= student_logits.size(1))).any()):
        raise ValueError("labels must index the logits class dimension")
    teacher_prediction = teacher_logits.detach().argmax(dim=1)
    selected = teacher_prediction.eq(labels)
    active_fraction = selected.float().mean()
    if not bool(selected.any()):
        return _zero_loss(student_logits), active_fraction
    per_sample = per_sample_kl_distillation_loss(
        student_logits[selected], teacher_logits[selected], temperature=temperature
    )
    return per_sample.mean(), active_fraction


def uncertainty_decoupled_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
    temperature: float = 2.0,
    uncertainty_weight_mode: str = "mean_normalized",
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decouple target-class and non-target KD; gate only non-target transfer.

    The target-vs-rest term remains unweighted. Detached teacher uncertainty
    redistributes only the conditional non-target KL, while the caller's CE
    loss remains the label-supervised target.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have matching [batch, classes] shapes")
    batch_size, class_count = student_logits.shape
    if labels.shape != (batch_size,):
        raise ValueError("labels must have shape [batch_size]")
    if teacher_uncertainty is not None and teacher_uncertainty.shape != (batch_size,):
        raise ValueError("teacher_uncertainty must have shape [batch_size]")
    if batch_size == 0:
        return _zero_loss(student_logits)
    if class_count < 2:
        raise ValueError("decoupled KD requires at least two classes")
    if bool(((labels < 0) | (labels >= class_count)).any()):
        raise ValueError("labels must index the logits class dimension")

    temperature = max(float(temperature), 1e-6)
    student = student_logits.float() / temperature
    teacher = teacher_logits.detach().float() / temperature
    row = torch.arange(batch_size, device=student.device)
    target_mask = F.one_hot(labels, class_count).bool()

    student_target = student[row, labels]
    teacher_target = teacher[row, labels]
    student_target_rest = torch.stack(
        (
            student_target,
            torch.logsumexp(student.masked_fill(target_mask, float("-inf")), dim=1),
        ),
        dim=1,
    )
    teacher_target_rest = torch.stack(
        (
            teacher_target,
            torch.logsumexp(teacher.masked_fill(target_mask, float("-inf")), dim=1),
        ),
        dim=1,
    )
    target_kl = F.kl_div(
        F.log_softmax(student_target_rest, dim=1),
        F.softmax(teacher_target_rest, dim=1),
        reduction="none",
    ).sum(dim=1)

    negative = torch.finfo(student.dtype).min
    student_non_target = student.masked_fill(target_mask, negative)
    teacher_non_target = teacher.masked_fill(target_mask, negative)
    non_target_kl = F.kl_div(
        F.log_softmax(student_non_target, dim=1),
        F.softmax(teacher_non_target, dim=1),
        reduction="none",
    ).sum(dim=1)
    weights = (
        uncertainty_weights(teacher_uncertainty, mode=uncertainty_weight_mode)
        if teacher_uncertainty is not None
        else torch.ones_like(non_target_kl)
    )
    target_term = target_kl.mean() * (temperature**2)
    non_target_term = non_target_kl.mean() * (temperature**2)
    weighted_non_target_term = (weights * non_target_kl).mean() * (temperature**2)
    if return_components:
        return target_term, non_target_term, weighted_non_target_term
    return target_term + weighted_non_target_term


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
    """Compute standard or uncertainty-weighted KL distillation."""
    per_sample = per_sample_kl_distillation_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
    )
    if per_sample.numel() == 0:
        return _zero_loss(student_logits)
    if teacher_uncertainty is not None and uncertainty_weighted:
        if teacher_uncertainty.shape != (student_logits.size(0),):
            raise ValueError("teacher_uncertainty must have shape [batch_size]")
        per_sample = per_sample * uncertainty_weights(
            teacher_uncertainty,
            mode=uncertainty_weight_mode,
            uncertainty_weight_min=uncertainty_weight_min,
            uncertainty_weight_max=uncertainty_weight_max,
        )
    return per_sample.mean()


def feature_distillation_loss(
    student_projection: torch.Tensor,
    teacher_projection: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
    uncertainty_weight_mode: str = "raw",
    uncertainty_weight_min: float | None = None,
    uncertainty_weight_max: float | None = None,
) -> torch.Tensor:
    """Align student and teacher projections with optional KD weights."""
    if student_projection.shape != teacher_projection.shape:
        raise ValueError(
            "student_projection and teacher_projection must have the same shape"
        )
    if student_projection.numel() == 0:
        return _zero_loss(student_projection)
    per_sample = 1.0 - F.cosine_similarity(
        student_projection,
        teacher_projection.detach(),
        dim=-1,
    )
    if teacher_uncertainty is not None:
        if teacher_uncertainty.shape != (student_projection.size(0),):
            raise ValueError("teacher_uncertainty must have shape [batch_size]")
        per_sample = per_sample * uncertainty_weights(
            teacher_uncertainty,
            mode=uncertainty_weight_mode,
            uncertainty_weight_min=uncertainty_weight_min,
            uncertainty_weight_max=uncertainty_weight_max,
        )
    return per_sample.mean()


def uncertainty_targets(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mode: str = "confidence",
) -> torch.Tensor:
    """Build detached uncertainty targets from known-class predictions."""
    if logits.size(0) != labels.numel():
        raise ValueError("labels must match the logits batch size")
    if logits.numel() == 0:
        return logits.new_empty((logits.size(0),))
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
    """Calibrate the uncertainty head without backpropagating through targets."""
    if uncertainty.shape != (logits.size(0),):
        raise ValueError("uncertainty must have shape [batch_size]")
    target = uncertainty_targets(logits, labels, mode=target_mode)
    if target.numel() == 0:
        return _zero_loss(uncertainty)
    if target_mode == "classification_error":
        return F.binary_cross_entropy(
            uncertainty.clamp(1e-6, 1.0 - 1e-6),
            target,
        )
    return F.mse_loss(uncertainty, target)
