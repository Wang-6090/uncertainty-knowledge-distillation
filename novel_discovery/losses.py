from __future__ import annotations

import torch
from torch.nn import functional as F

from .uncertainty_kd import (
    feature_distillation_loss as _feature_distillation_loss,
    kl_distillation_loss,
    uncertainty_calibration_loss,
    uncertainty_weights as _uncertainty_weights,
)


def uncertainty_weights(
    uncertainty: torch.Tensor,
    mode: str = "raw",
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> torch.Tensor:
    return _uncertainty_weights(
        uncertainty,
        mode=mode,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def uncertainty_alignment_loss(
    uncertainty: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mode: str = "confidence",
) -> torch.Tensor:
    return uncertainty_calibration_loss(
        uncertainty,
        logits,
        labels,
        target_mode=target_mode,
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
    temperature: float = 2.0,
    uncertainty_weighted: bool = True,
    uncertainty_weight_mode: str = "raw",
    uncertainty_weight_min: float | None = None,
    uncertainty_weight_max: float | None = None,
) -> torch.Tensor:
    return kl_distillation_loss(
        student_logits,
        teacher_logits,
        teacher_uncertainty=teacher_uncertainty,
        temperature=temperature,
        uncertainty_weighted=uncertainty_weighted,
        uncertainty_weight_mode=uncertainty_weight_mode,
        uncertainty_weight_min=uncertainty_weight_min,
        uncertainty_weight_max=uncertainty_weight_max,
    )


def feature_distillation_loss(
    student_projection: torch.Tensor,
    teacher_projection: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
    uncertainty_weight_mode: str = "raw",
    uncertainty_weight_min: float | None = None,
    uncertainty_weight_max: float | None = None,
) -> torch.Tensor:
    return _feature_distillation_loss(
        student_projection,
        teacher_projection,
        teacher_uncertainty=teacher_uncertainty,
        uncertainty_weight_mode=uncertainty_weight_mode,
        uncertainty_weight_min=uncertainty_weight_min,
        uncertainty_weight_max=uncertainty_weight_max,
    )


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if features.size(0) <= 1:
        return features.new_tensor(0.0)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    eye = torch.eye(mask.size(0), device=mask.device)
    mask = mask - eye
    valid = labels.squeeze(1) >= 0
    if valid.sum() <= 1:
        return features.new_tensor(0.0)
    features = F.normalize(features, dim=-1)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = 1.0 - eye
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    pos_counts = mask.sum(dim=1)
    valid = valid & (pos_counts > 0)
    if valid.sum() == 0:
        return features.new_tensor(0.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_counts.clamp_min(1.0)
    loss = -mean_log_prob_pos[valid].mean()
    return loss


def prototype_alignment_loss(features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    valid = labels >= 0
    if valid.sum() == 0:
        return features.new_tensor(0.0)
    feats = F.normalize(features[valid], dim=-1)
    proto = F.normalize(prototypes[labels[valid]], dim=-1)
    return 1.0 - (feats * proto).sum(dim=-1).mean()


def pseudo_unknown_loss(logits: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    probs = logits.softmax(dim=-1)
    confidence = probs.max(dim=-1).values
    log_num_classes = logits.new_tensor(float(logits.size(-1))).log()
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    entropy_gap = (1.0 - entropy / log_num_classes).pow(2)
    unc_target = torch.ones_like(uncertainty)
    loss_unc = F.mse_loss(uncertainty, unc_target)
    return confidence.mean() + 0.5 * entropy_gap.mean() + loss_unc


def discovery_unknown_loss(
    logits: torch.Tensor,
    uncertainty: torch.Tensor,
) -> torch.Tensor:
    """Encourage unlabeled discovery samples to leave the known classifier.

    This is only appropriate when the discovery pool is known by protocol to
    contain novel samples. For mixed pools, use view consistency/contrast only.
    """
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    probs = logits.softmax(dim=-1)
    confidence = probs.max(dim=-1).values
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    max_entropy = logits.new_tensor(float(logits.size(-1))).log()
    normalized_entropy = entropy / max_entropy.clamp_min(1e-8)
    return (
        confidence.mean()
        + (1.0 - normalized_entropy).pow(2).mean()
        + F.mse_loss(uncertainty, torch.ones_like(uncertainty))
    )


def discovery_consistency_loss(
    first_projection: torch.Tensor,
    second_projection: torch.Tensor,
) -> torch.Tensor:
    """Align two augmented views in normalized projection space."""
    if first_projection.numel() == 0:
        return first_projection.new_tensor(0.0)
    return 1.0 - F.cosine_similarity(
        first_projection, second_projection, dim=-1
    ).mean()


def discovery_contrastive_loss(
    first_projection: torch.Tensor,
    second_projection: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """SimCLR-style NT-Xent loss for unlabeled discovery samples."""
    if first_projection.size(0) <= 1:
        return discovery_consistency_loss(first_projection, second_projection)

    first_projection = F.normalize(first_projection, dim=-1)
    second_projection = F.normalize(second_projection, dim=-1)
    features = torch.cat([first_projection, second_projection], dim=0)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits.masked_fill(
        torch.eye(logits.size(0), dtype=torch.bool, device=logits.device),
        float("-inf"),
    )
    batch_size = first_projection.size(0)
    targets = torch.arange(2 * batch_size, device=logits.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)


def discovery_view_loss(
    first_projection: torch.Tensor,
    second_projection: torch.Tensor,
    mode: str = "nt_xent",
    temperature: float = 0.2,
) -> torch.Tensor:
    if mode == "consistency":
        return discovery_consistency_loss(first_projection, second_projection)
    if mode == "nt_xent":
        return discovery_contrastive_loss(first_projection, second_projection, temperature)
    raise ValueError(f"Unsupported discovery loss mode: {mode}")
