from __future__ import annotations

import torch
from torch.nn import functional as F


def uncertainty_weights(
    uncertainty: torch.Tensor,
    mode: str = "raw",
) -> torch.Tensor:
    """Convert teacher uncertainty to per-sample distillation weights.

    ``raw`` preserves the original behavior. ``mean_normalized`` keeps the
    batch-average KD strength close to one, so ablations compare where the KD
    signal is allocated instead of also changing the total KD strength.
    """
    weights = torch.exp(-uncertainty.detach())
    if mode == "raw":
        return weights
    if mode == "mean_normalized":
        return weights / weights.mean().clamp_min(1e-6)
    raise ValueError(f"Unsupported uncertainty weight mode: {mode}")


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
    uncertainty_weight_mode: str = "raw",
) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1).detach()
    per_sample = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * (temperature**2)
    if teacher_uncertainty is not None and uncertainty_weighted:
        weight = uncertainty_weights(teacher_uncertainty, mode=uncertainty_weight_mode)
        per_sample = per_sample * weight
    return per_sample.mean()


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute the energy score; lower values indicate more familiar inputs."""
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def energy_margin_loss(
    known_logits: torch.Tensor,
    outlier_logits: torch.Tensor,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Separate known and pseudo-outlier energies without absolute thresholds."""
    if known_logits.numel() == 0 or outlier_logits.numel() == 0:
        return known_logits.new_tensor(0.0)
    return per_sample_energy_margin_loss(
        known_logits,
        outlier_logits,
        margin=margin,
        temperature=temperature,
    ).mean()


def per_sample_energy_margin_loss(
    known_logits: torch.Tensor,
    outlier_logits: torch.Tensor,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return one Energy-margin loss value per paired known/outlier sample."""
    if known_logits.numel() == 0 or outlier_logits.numel() == 0:
        return known_logits.new_empty(0)
    known_energy = energy_score(known_logits, temperature=temperature)
    outlier_energy = energy_score(outlier_logits, temperature=temperature)
    pair_count = min(known_energy.size(0), outlier_energy.size(0))
    known_energy = known_energy[:pair_count]
    outlier_energy = outlier_energy[:pair_count]
    return F.relu(known_energy - outlier_energy + margin)


def weighted_energy_margin_loss(
    known_logits: torch.Tensor,
    outlier_logits: torch.Tensor,
    outlier_weight: torch.Tensor | None = None,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply a normalized per-outlier weight to Energy-margin training.

    A zero total weight returns a differentiable zero tensor. This lets mixed
    discovery batches safely contain no selected candidates.
    """
    per_sample = per_sample_energy_margin_loss(
        known_logits,
        outlier_logits,
        margin=margin,
        temperature=temperature,
    )
    if per_sample.numel() == 0:
        return known_logits.new_tensor(0.0)
    if outlier_weight is None:
        return per_sample.mean()
    weight = outlier_weight.reshape(-1).to(per_sample).clamp_min(0.0)
    weight = weight[: per_sample.numel()]
    if weight.numel() != per_sample.numel():
        raise ValueError("outlier_weight must match the outlier batch size.")
    denominator = weight.sum()
    if denominator <= 0:
        return per_sample.sum() * 0.0
    return (per_sample * weight).sum() / denominator


def feature_distillation_loss(
    student_projection: torch.Tensor,
    teacher_projection: torch.Tensor,
    teacher_uncertainty: torch.Tensor | None = None,
    uncertainty_weight_mode: str = "raw",
) -> torch.Tensor:
    """Distill relationally useful normalized representations.

    The projection head has a fixed dimension, so this remains valid when the
    teacher and student use different encoder backbones.
    """
    per_sample = 1.0 - F.cosine_similarity(
        student_projection, teacher_projection.detach(), dim=-1
    )
    if teacher_uncertainty is not None:
        weight = uncertainty_weights(teacher_uncertainty, mode=uncertainty_weight_mode)
        per_sample = per_sample * weight
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


def proxy_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    proxies: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Use class proxies as contrastive targets when batch positives are sparse.

    This is a lightweight Proxy-NCA / normalized-softmax style loss: each
    feature is pulled toward its class proxy and pushed away from other class
    proxies, without requiring another same-class sample in the mini-batch.
    """
    valid = labels >= 0
    if valid.sum() == 0:
        return features.new_tensor(0.0)
    feats = F.normalize(features[valid], dim=-1)
    proxy = F.normalize(proxies, dim=-1)
    logits = feats @ proxy.T / max(float(temperature), 1e-6)
    return F.cross_entropy(logits, labels[valid])


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
    sample_weight: torch.Tensor | None = None,
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
    per_sample = confidence + (1.0 - normalized_entropy).pow(2) + (uncertainty - 1.0).pow(2)
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.reshape(-1).to(per_sample).clamp_min(0.0)
    if weight.numel() != per_sample.numel():
        raise ValueError("sample_weight must match the logits batch size.")
    denominator = weight.sum()
    if denominator <= 0:
        return per_sample.sum() * 0.0
    return (per_sample * weight).sum() / denominator


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
