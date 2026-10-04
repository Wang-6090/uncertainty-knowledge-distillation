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
    uncertainty_weight_min: float | None = None,
    uncertainty_weight_max: float | None = None,
) -> torch.Tensor:
    return _uncertainty_weights(
        uncertainty,
        mode=mode,
        uncertainty_weight_min=uncertainty_weight_min,
        uncertainty_weight_max=uncertainty_weight_max,
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


def prototype_repulsion_loss(
    prototypes: torch.Tensor,
    similarity_margin: float = 0.0,
) -> torch.Tensor:
    """Penalize classifier prototypes that are too close to one another.

    This is an optional class-space regularizer. It does not push every pair
    to be maximally opposite; it only penalizes off-diagonal cosine
    similarities above ``similarity_margin`` and therefore stays compatible
    with many-class classifiers.
    """
    if prototypes.ndim != 2 or prototypes.size(0) <= 1:
        return prototypes.new_tensor(0.0)
    if not -1.0 <= float(similarity_margin) <= 1.0:
        raise ValueError("similarity_margin must be in [-1, 1]")
    normalized = F.normalize(prototypes, dim=-1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    violations = F.relu(similarity[mask] - float(similarity_margin))
    return violations.mean() if violations.numel() else prototypes.new_tensor(0.0)


def unknown_feature_margin_loss(
    features: torch.Tensor,
    known_prototypes: torch.Tensor,
    similarity_margin: float = 0.2,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push unlabeled unknown features away from the nearest known prototype.

    This is intended for a protocol-defined pure-unknown pool or a filtered
    candidate subset, not arbitrary mixed unlabeled batches.
    """
    if features.numel() == 0 or known_prototypes.numel() == 0:
        return features.new_tensor(0.0)
    feature = F.normalize(features, dim=-1)
    prototypes = F.normalize(known_prototypes, dim=-1)
    nearest_similarity = (feature @ prototypes.T).max(dim=-1).values
    per_sample = F.relu(nearest_similarity - float(similarity_margin))
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.reshape(-1).to(per_sample).clamp_min(0.0)
    if weight.numel() != per_sample.numel():
        raise ValueError("sample_weight must match the feature batch size.")
    denominator = weight.sum()
    if denominator <= 0:
        return per_sample.sum() * 0.0
    return (per_sample * weight).sum() / denominator


def unknown_feature_separation_loss(
    known_features: torch.Tensor,
    unknown_features: torch.Tensor,
    similarity_margin: float = 0.0,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Push unknown features away from the *observed known feature batch*.

    The existing prototype-margin loss uses classifier weights as proxies.  That
    proxy can be misaligned with the feature statistics used by Mahalanobis
    scoring.  This loss instead compares discovery features with current known
    embeddings, using a smooth maximum over all known samples.  Known features
    are detached so this auxiliary term cannot move the known representation
    merely to make the separation task easier.
    """
    if known_features.numel() == 0 or unknown_features.numel() == 0:
        return unknown_features.new_tensor(0.0)
    known = F.normalize(known_features.detach(), dim=-1)
    unknown = F.normalize(unknown_features, dim=-1)
    similarities = unknown @ known.T
    tau = max(float(temperature), 1e-6)
    # Remove the batch-size-dependent log(N) offset from logsumexp so this is
    # a true smooth approximation of max cosine similarity.
    smooth_max = tau * (
        torch.logsumexp(similarities / tau, dim=-1)
        - torch.log(similarities.new_tensor(float(similarities.size(1))))
    )
    violations = F.relu(smooth_max - float(similarity_margin))
    return violations.mean()


def unknown_feature_boundary_loss(
    known_features: torch.Tensor,
    known_labels: torch.Tensor,
    unknown_features: torch.Tensor,
    known_prototypes: torch.Tensor,
    similarity_margin: float = 0.2,
    prototype_weight: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Repel pure-unknown features from a hybrid known support boundary.

    The classifier weights provide a global class prototype, while the
    labelled features in the current batch provide a local class centroid.
    Their detached combination is used as a stable, class-conditional support
    approximation.  This is an auxiliary open-set representation objective;
    it is intentionally restricted to a protocol-defined pure-unknown pool.
    It does not claim to reproduce a complete ARPL, VOS, or Mahalanobis model.
    """
    if (
        known_features.numel() == 0
        or unknown_features.numel() == 0
        or known_prototypes.numel() == 0
    ):
        return unknown_features.new_tensor(0.0)
    if known_features.ndim != 2 or unknown_features.ndim != 2:
        raise ValueError("known_features and unknown_features must be 2-D")
    if known_labels.ndim != 1 or known_labels.size(0) != known_features.size(0):
        raise ValueError("known_labels must contain one label per known feature")
    if known_prototypes.ndim != 2 or known_prototypes.size(1) != known_features.size(1):
        raise ValueError("known_prototypes must match the feature dimension")

    weight = min(max(float(prototype_weight), 0.0), 1.0)
    tau = max(float(temperature), 1e-6)
    known = F.normalize(known_features.detach(), dim=-1)
    unknown = F.normalize(unknown_features, dim=-1)
    prototypes = F.normalize(known_prototypes.detach(), dim=-1)

    # Estimate only the class centroids observed in this batch.  The missing
    # classes are still covered by the classifier prototypes.
    centroids = []
    for cls in torch.unique(known_labels.detach()):
        if int(cls) < 0:
            continue
        class_features = known[known_labels == cls]
        if class_features.numel() == 0:
            continue
        centroids.append(F.normalize(class_features.mean(dim=0, keepdim=True), dim=-1).squeeze(0))
    if centroids:
        local_centroids = torch.stack(centroids, dim=0)
        local_similarity = unknown @ local_centroids.T
        # Smooth max avoids a single unstable pair dominating the loss.
        local_max = tau * (
            torch.logsumexp(local_similarity / tau, dim=-1)
            - torch.log(local_similarity.new_tensor(float(local_similarity.size(1))))
        )
    else:
        local_max = torch.zeros(unknown.size(0), device=unknown.device, dtype=unknown.dtype)
    prototype_max = (unknown @ prototypes.T).max(dim=-1).values
    hybrid_max = weight * prototype_max + (1.0 - weight) * local_max
    return F.relu(hybrid_max - float(similarity_margin)).mean()


def objectosphere_loss(
    known_features: torch.Tensor,
    unknown_features: torch.Tensor,
    known_radius: float = 10.0,
    unknown_weight: float = 1.0,
) -> torch.Tensor:
    """Separate known and pure-unknown samples by feature norm.

    Known features retain a minimum radius while auxiliary unknown features
    are pulled toward the origin, following the Objectosphere objective. The
    terms are normalized by the target radius to reduce backbone-scale
    sensitivity.
    """
    if known_features.numel() == 0 or unknown_features.numel() == 0:
        reference = known_features if known_features.numel() else unknown_features
        return reference.new_tensor(0.0)
    radius = float(known_radius)
    if radius <= 0.0:
        raise ValueError("known_radius must be positive")
    if float(unknown_weight) < 0.0:
        raise ValueError("unknown_weight must be non-negative")
    known_norm = known_features.norm(dim=-1)
    unknown_norm = unknown_features.norm(dim=-1)
    scale = radius * radius
    known_penalty = F.relu(radius - known_norm).pow(2).mean() / scale
    unknown_penalty = unknown_norm.pow(2).mean() / scale
    return known_penalty + float(unknown_weight) * unknown_penalty


def knn_support_boundary_loss(
    unknown_features: torch.Tensor,
    support_features_by_class: list[torch.Tensor],
    support_radii: torch.Tensor,
    k: int = 5,
    margin: float = 0.02,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push pure-unknown features outside known class-conditional kNN support.

    Each class support radius is estimated from known training features only.
    The bank/radii are fixed targets; gradients flow only through unknown
    features. This is a project-specific training adaptation, not a full OOD
    nearest-neighbor detector reproduction.
    """
    if unknown_features.numel() == 0 or not support_features_by_class:
        return unknown_features.new_tensor(0.0)
    if support_radii.ndim != 1 or support_radii.numel() != len(support_features_by_class):
        raise ValueError("support_radii must contain one radius per class bank")
    if int(k) <= 0 or float(margin) < 0.0:
        raise ValueError("k must be positive and margin non-negative")

    unknown = F.normalize(unknown_features, dim=-1)
    radii = support_radii.detach().to(device=unknown.device, dtype=unknown.dtype)
    class_violations = []
    for class_index, bank in enumerate(support_features_by_class):
        if bank.numel() == 0:
            continue
        normalized_bank = F.normalize(bank.detach().to(unknown), dim=-1)
        similarities = unknown @ normalized_bank.T
        neighbor_count = min(int(k), normalized_bank.size(0))
        nearest_similarity = similarities.topk(neighbor_count, dim=-1).values.mean(dim=-1)
        distance = 1.0 - nearest_similarity
        class_violations.append(
            F.relu(radii[class_index] + float(margin) - distance)
        )
    if not class_violations:
        return unknown_features.new_tensor(0.0)
    # A sample is penalized when it remains inside at least one known class
    # support region. Optional weights let reliable candidates contribute more
    # without changing the default hard-gated behavior.
    violations = torch.stack(class_violations, dim=1).max(dim=1).values
    if sample_weights is None:
        return violations.mean()
    weights = sample_weights.to(device=violations.device, dtype=violations.dtype).flatten()
    if weights.numel() != violations.numel():
        raise ValueError("sample_weights must contain one value per unknown feature")
    weights = weights.clamp_min(0.0)
    normalizer = weights.sum()
    if normalizer <= 0:
        return violations.new_tensor(0.0)
    return (violations * weights).sum() / normalizer


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


def proxy_anchor_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    proxies: torch.Tensor,
    alpha: float = 32.0,
    margin: float = 0.1,
) -> torch.Tensor:
    """Proxy Anchor loss for proxy-based metric learning.

    The formulation follows Kim et al., *Proxy Anchor Loss for Deep Metric
    Learning* (CVPR 2020). It aggregates positive and negative examples per
    proxy, which is useful when a batch contains few examples from many
    classes. This is an optional representation-learning loss, not a direct
    unknown detector.
    """
    valid = labels >= 0
    if valid.sum() == 0:
        return features.new_tensor(0.0)
    alpha = float(alpha)
    margin = float(margin)
    if alpha <= 0.0 or margin < 0.0:
        raise ValueError("proxy-anchor alpha must be positive and margin non-negative")
    feats = F.normalize(features[valid], dim=-1)
    proxy = F.normalize(proxies, dim=-1)
    similarity = feats @ proxy.T
    labels = labels[valid].long()
    positive = F.one_hot(labels, num_classes=proxy.size(0)).to(dtype=torch.bool)
    present = positive.any(dim=0)
    if not present.any():
        return features.new_tensor(0.0)
    similarity = similarity[:, present]
    positive = positive[:, present]
    negative = ~positive

    positive_logits = -alpha * (similarity - margin)
    positive_logits = positive_logits.masked_fill(~positive, float("-inf"))
    positive_term = F.softplus(torch.logsumexp(positive_logits, dim=0)).mean()

    negative_logits = alpha * (similarity + margin)
    negative_logits = negative_logits.masked_fill(~negative, float("-inf"))
    valid_negative = negative.any(dim=0)
    if valid_negative.any():
        negative_term = F.softplus(
            torch.logsumexp(negative_logits[:, valid_negative], dim=0)
        ).mean()
    else:
        negative_term = similarity.new_tensor(0.0)
    return positive_term + negative_term


def reciprocal_point_loss(
    known_features: torch.Tensor,
    unknown_features: torch.Tensor,
    known_prototypes: torch.Tensor,
    reciprocal_points: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """ARPL-inspired reciprocal-point objective.

    Known features are repelled from learned reciprocal points. Discovery
    features are attracted to their nearest reciprocal point and repelled from
    known class prototypes. The objective is deliberately exposed as an
    auxiliary loss: it is an ARPL-inspired adaptation for this project's
    pure-unknown discovery pool, not a claim of reproducing full ARPL.
    """
    if known_features.numel() == 0 or unknown_features.numel() == 0:
        return known_features.new_tensor(0.0)
    if reciprocal_points.numel() == 0 or known_prototypes.numel() == 0:
        return known_features.new_tensor(0.0)
    margin = float(margin)
    if margin < 0.0 or margin >= 1.0:
        raise ValueError("reciprocal margin must be in [0, 1)")
    known = F.normalize(known_features, dim=-1)
    unknown = F.normalize(unknown_features, dim=-1)
    prototypes = F.normalize(known_prototypes, dim=-1)
    points = F.normalize(reciprocal_points, dim=-1)

    known_point_similarity = (known @ points.T).max(dim=-1).values
    repel_known = F.relu(known_point_similarity - margin).mean()
    unknown_point_similarity = (unknown @ points.T).max(dim=-1).values
    attract_unknown = (1.0 - unknown_point_similarity).mean()
    unknown_known_similarity = (unknown @ prototypes.T).max(dim=-1).values
    repel_unknown_known = F.relu(unknown_known_similarity - margin).mean()

    if points.size(0) > 1:
        pairwise = points @ points.T
        mask = ~torch.eye(points.size(0), dtype=torch.bool, device=points.device)
        diversify = F.relu(pairwise[mask] - margin).mean()
    else:
        diversify = points.new_tensor(0.0)
    return repel_known + attract_unknown + repel_unknown_known + 0.1 * diversify


def known_pseudo_label_consistency_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match two augmented views to detached known-class pseudo labels.

    This is intended for a mixed unlabeled pool: only samples selected by an
    external confidence/agreement gate should reach this loss.  The labels are
    supplied by a detached teacher/EMA model, so this term cannot reinforce a
    student's own changing logits through the target path.
    """
    if first_logits.shape != second_logits.shape:
        raise ValueError("first_logits and second_logits must have the same shape")
    if pseudo_labels.ndim != 1 or pseudo_labels.size(0) != first_logits.size(0):
        raise ValueError("pseudo_labels must contain one label per sample")
    if first_logits.size(0) == 0:
        return first_logits.new_tensor(0.0)
    if mask is None:
        mask = torch.ones(first_logits.size(0), dtype=torch.bool, device=first_logits.device)
    else:
        mask = mask.to(device=first_logits.device, dtype=torch.bool)
    if not mask.any():
        return first_logits.new_tensor(0.0)
    labels = pseudo_labels.detach().to(device=first_logits.device, dtype=torch.long)
    first = F.cross_entropy(first_logits[mask], labels[mask], reduction="mean")
    second = F.cross_entropy(second_logits[mask], labels[mask], reduction="mean")
    return 0.5 * (first + second)


def angular_margin_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor,
    margin: float = 0.2,
    scale: float = 16.0,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """ArcFace-style auxiliary loss for tighter known-class features.

    This follows the additive angular-margin idea of Deng et al. (CVPR
    2019), but is kept as an auxiliary loss so the ordinary inference logits
    and historical default training remain unchanged.
    """
    valid = labels >= 0
    if sample_mask is not None:
        if sample_mask.ndim != 1 or sample_mask.numel() != labels.numel():
            raise ValueError("sample_mask must contain one boolean value per label")
        valid = valid & sample_mask.to(device=labels.device, dtype=torch.bool)
    if valid.sum() == 0:
        return features.new_tensor(0.0)
    margin = float(margin)
    scale = float(scale)
    if margin < 0.0 or margin >= 1.0:
        raise ValueError("angular margin must be in [0, 1)")
    if scale <= 0.0:
        raise ValueError("angular scale must be positive")
    feats = F.normalize(features[valid], dim=-1)
    weights = F.normalize(classifier_weight, dim=-1)
    cosine = (feats @ weights.T).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    target = labels[valid].long()
    sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
    margin_tensor = cosine.new_tensor(margin)
    phi = cosine * torch.cos(margin_tensor) - sine * torch.sin(margin_tensor)
    logits = cosine.clone()
    rows = torch.arange(target.numel(), device=target.device)
    logits[rows, target] = phi[rows, target]
    return F.cross_entropy(logits * scale, target)


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


def outlier_exposure_uniform_loss(
    logits: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match a pure outlier batch to a uniform known-class distribution.

    This is the classification-logit part of Outlier Exposure: novel samples
    should not receive a concentrated probability on any known class.  It is
    intentionally separate from ``discovery_unknown_loss`` so experiments can
    distinguish uniform-logit exposure from uncertainty-head supervision.
    """
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -log_probs.mean(dim=-1)
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.reshape(-1).to(per_sample).clamp_min(0.0)
    if weight.numel() != per_sample.numel():
        raise ValueError("sample_weight must match the logits batch size.")
    denominator = weight.sum()
    if denominator <= 0:
        return per_sample.sum() * 0.0
    return (per_sample * weight).sum() / denominator


def uncertainty_separation_loss(
    known_uncertainty: torch.Tensor,
    unknown_uncertainty: torch.Tensor,
) -> torch.Tensor:
    """Train the uncertainty head as a known-vs-unknown binary score."""
    if known_uncertainty.numel() == 0 or unknown_uncertainty.numel() == 0:
        return known_uncertainty.new_tensor(0.0)
    values = torch.cat([known_uncertainty, unknown_uncertainty], dim=0).clamp(1e-6, 1.0 - 1e-6)
    targets = torch.cat(
        [torch.zeros_like(known_uncertainty), torch.ones_like(unknown_uncertainty)], dim=0
    )
    return F.binary_cross_entropy(values, targets)


def nnpu_known_uncertainty_loss(
    known_uncertainty: torch.Tensor,
    unlabeled_uncertainty: torch.Tensor,
    known_prior: float,
) -> torch.Tensor:
    """Non-negative PU risk for known-vs-unknown scoring from a mixed pool.

    The uncertainty head estimates novelty, so ``1 - uncertainty`` is treated
    as the positive (known) probability. Labeled known samples are P; the
    mixed discovery pool is U and is never assigned per-example unknown labels.
    ``known_prior`` is the assumed known proportion in U.
    """
    prior = float(known_prior)
    if not 0.0 < prior < 1.0:
        raise ValueError("known_prior must be in (0, 1)")
    if known_uncertainty.numel() == 0 or unlabeled_uncertainty.numel() == 0:
        reference = known_uncertainty if known_uncertainty.numel() else unlabeled_uncertainty
        return reference.new_tensor(0.0)

    known_u = known_uncertainty.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    unlabeled_u = unlabeled_uncertainty.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    # g is the knownness logit: positive values indicate known samples.
    known_logit = torch.log1p(-known_u) - torch.log(known_u)
    unlabeled_logit = torch.log1p(-unlabeled_u) - torch.log(unlabeled_u)

    positive_risk = prior * F.softplus(-known_logit).mean()
    negative_risk = (
        F.softplus(unlabeled_logit).mean()
        - prior * F.softplus(known_logit).mean()
    )
    if negative_risk.detach().item() < 0.0:
        # Kiryo et al.'s nnPU correction uses gradient ascent in the region
        # where the empirical negative-risk estimate becomes negative. A
        # plain clamp would zero this gradient and is not the same update.
        return positive_risk - negative_risk
    return positive_risk + negative_risk


def uncertainty_ranking_loss(
    known_uncertainty: torch.Tensor,
    unknown_uncertainty: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Rank unknown uncertainty above known uncertainty with a pairwise margin."""
    if known_uncertainty.numel() == 0 or unknown_uncertainty.numel() == 0:
        return known_uncertainty.new_tensor(0.0)
    known = known_uncertainty.reshape(-1)
    unknown = unknown_uncertainty.reshape(-1)
    pairwise = F.relu(float(margin) + known[:, None] - unknown[None, :])
    return pairwise.mean()


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
