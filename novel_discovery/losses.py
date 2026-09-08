from __future__ import annotations

import torch
from torch.nn import functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def uncertainty_alignment_loss(
    uncertainty: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mode: str = "confidence",
) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    true_conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    if target_mode == "confidence":
        target = (1.0 - true_conf).detach()
        return F.mse_loss(uncertainty, target)
    if target_mode == "classification_error":
        # Learn uncertainty as an error probability on known samples. The
        # target is detached because it is a supervision signal, not a path
        # through which the classifier should optimize its own predictions.
        pred = logits.argmax(dim=-1)
        target = (pred != labels).float().detach()
        return F.binary_cross_entropy(uncertainty.clamp(1e-6, 1.0 - 1e-6), target)
    if target_mode == "margin":
        # Continuous uncertainty target from the decision margin. A sample
        # becomes uncertain when the strongest competing class approaches or
        # exceeds the true-class logit. Detach the target so this auxiliary
        # task does not distort the classifier through its own target.
        true_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = logits.clone()
        other_logits.scatter_(1, labels.unsqueeze(1), float("-inf"))
        strongest_other = other_logits.max(dim=-1).values
        target = torch.sigmoid((strongest_other - true_logit).detach())
        return F.mse_loss(uncertainty, target)
    raise ValueError(f"Unsupported uncertainty target mode: {target_mode}")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
    temperature: float = 2.0,
    uncertainty_weighted: bool = True,
) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1).detach()
    per_sample = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * (temperature**2)
    if teacher_uncertainty is not None and uncertainty_weighted:
        weight = torch.exp(-teacher_uncertainty.detach())
        per_sample = per_sample * weight
    return per_sample.mean()


def feature_distillation_loss(
    student_projection: torch.Tensor,
    teacher_projection: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distill relationally useful normalized representations.

    The projection head has a fixed dimension, so this remains valid when the
    teacher and student use different encoder backbones.
    """
    per_sample = 1.0 - F.cosine_similarity(
        student_projection, teacher_projection.detach(), dim=-1
    )
    if teacher_uncertainty is not None:
        per_sample = per_sample * torch.exp(-teacher_uncertainty.detach())
    return per_sample.mean()


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
    pos_counts = mask.sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_counts
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
