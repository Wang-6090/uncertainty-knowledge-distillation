from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.stats import weibull_min
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from .losses import (
    classification_loss,
    discovery_unknown_loss,
    discovery_view_loss,
    distillation_loss,
    energy_margin_loss,
    angular_margin_loss,
    feature_distillation_loss,
    pseudo_unknown_loss,
    proxy_contrastive_loss,
    proxy_anchor_loss,
    reciprocal_point_loss,
    known_pseudo_label_consistency_loss,
    prototype_alignment_loss,
    prototype_repulsion_loss,
    supervised_contrastive_loss,
    uncertainty_alignment_loss,
    weighted_energy_margin_loss,
    unknown_feature_margin_loss,
    unknown_feature_separation_loss,
    unknown_feature_boundary_loss,
    knn_support_boundary_loss,
    objectosphere_loss,
    uncertainty_separation_loss,
    uncertainty_ranking_loss,
    nnpu_known_uncertainty_loss,
    outlier_exposure_uniform_loss,
)
from .joint_discovery import (
    combine_known_novel_logits,
    joint_discovery_loss,
    known_residual_weights,
    neighbor_novel_support_weights,
    novel_mass_weights,
    bounded_novel_mass_weights,
)
from .metrics import (
    clustering_report,
    compute_aupr,
    compute_auroc,
    compute_fpr95,
    compute_oscr,
    open_set_confusion,
)
from .utils import AverageMeter
from .models import freeze_batchnorm_stats


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def weighted_discovery_view_loss(
    first_projection: torch.Tensor,
    second_projection: torch.Tensor,
    alpha: float,
    mode: str = "nt_xent",
    temperature: float = 0.2,
) -> torch.Tensor:
    """Apply two-view loss only when its own explicit coefficient is active."""
    if float(alpha) <= 0.0:
        return first_projection.new_tensor(0.0)
    return discovery_view_loss(
        first_projection,
        second_projection,
        mode=mode,
        temperature=temperature,
    )


def make_pseudo_unknown(
    images: torch.Tensor,
    noise_scale: float = 0.15,
    mode: str = "strong",
) -> torch.Tensor:
    """Create a pseudo-unknown batch from known images.

    ``legacy`` reproduces the original mix-plus-noise generator. ``strong``
    additionally applies random erasing and a local smoothing perturbation,
    producing samples farther from individual known-class appearances.
    """
    if images.size(0) > 1:
        permutation = torch.randperm(images.size(0), device=images.device)
        mixed = 0.5 * images + 0.5 * images[permutation]
    else:
        mixed = images
    if mode == "legacy":
        return mixed + noise_scale * torch.randn_like(mixed)
    if mode != "strong":
        raise ValueError(f"Unsupported pseudo mode: {mode}")

    # Randomly erase a rectangle in each image. The mask is sampled per image
    # so the pseudo batch does not collapse to one shared corruption pattern.
    batch, _, height, width = mixed.shape
    erase_h = max(1, int(height * 0.25))
    erase_w = max(1, int(width * 0.25))
    top = torch.randint(0, max(height - erase_h + 1, 1), (batch, 1, 1), device=images.device)
    left = torch.randint(0, max(width - erase_w + 1, 1), (batch, 1, 1), device=images.device)
    yy = torch.arange(height, device=images.device).view(1, height, 1)
    xx = torch.arange(width, device=images.device).view(1, 1, width)
    erase_mask = (yy >= top) & (yy < top + erase_h) & (xx >= left) & (xx < left + erase_w)
    mixed = mixed.masked_fill(erase_mask.unsqueeze(1), 0.0)

    # Local averaging changes texture while preserving the tensor shape and
    # keeps the operation cheap on both CPU and GPU.
    smoothed = torch.nn.functional.avg_pool2d(mixed, kernel_size=3, stride=1, padding=1)
    mixed = 0.75 * mixed + 0.25 * smoothed
    return mixed + (noise_scale * 1.5) * torch.randn_like(mixed)


def pseudo_forward_from_features(model, features: torch.Tensor, stochastic: bool = True):
    """Classify perturbed pseudo-unknown features with the model heads."""
    features = torch.nn.functional.dropout(
        features,
        p=model.dropout_p,
        training=stochastic,
    )
    logits = model.classifier(features)
    uncertainty = torch.sigmoid(model.uncertainty_head(features)).squeeze(-1)
    return logits, uncertainty


def synthesize_virtual_outliers(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor,
    tail_scale: float = 1.5,
    noise_scale: float = 0.25,
) -> torch.Tensor:
    """Generate VOS-inspired low-density feature candidates.

    This lightweight approximation estimates class centers from the current
    known batch and extrapolates a sample toward a low-density tail direction.
    The classifier weight is only a fallback for a degenerate class center. A
    small random orthogonal perturbation prevents collapse to one direction.
    It is not a reproduction of the full VOS algorithm.
    """
    if features.numel() == 0:
        return features.new_empty(features.shape)
    if labels.numel() != features.size(0):
        raise ValueError("labels must have one entry per feature")
    if classifier_weight.ndim != 2 or classifier_weight.size(1) != features.size(1):
        raise ValueError("classifier_weight must have shape [num_classes, feature_dim]")
    if labels.min().item() < 0 or labels.max().item() >= classifier_weight.size(0):
        raise ValueError("labels contain an invalid classifier index")

    feature_norm = features.norm(dim=-1, keepdim=True).detach().clamp_min(1e-4)
    class_means = features.new_zeros(classifier_weight.size(0), features.size(1))
    class_means.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=classifier_weight.size(0)).to(features)
    class_means = class_means / counts.clamp_min(1.0).unsqueeze(-1)
    batch_proxies = class_means[labels]
    fallback_proxies = classifier_weight[labels]
    use_fallback = batch_proxies.norm(dim=-1, keepdim=True) < 1e-6
    proxies = torch.where(use_fallback, fallback_proxies, batch_proxies)
    proxies = F.normalize(proxies, dim=-1)
    normalized_features = F.normalize(features, dim=-1)
    centered = features - class_means[labels].detach()
    residual = F.normalize(centered, dim=-1)
    residual_norm = centered.norm(dim=-1, keepdim=True)
    residual = torch.where(residual_norm > 1e-6, residual, normalized_features)
    projection = (residual * proxies).sum(dim=-1, keepdim=True)
    residual = residual - projection * proxies
    noise = torch.randn_like(normalized_features)
    noise = noise - (noise * proxies).sum(dim=-1, keepdim=True) * proxies
    candidate_direction = residual + float(noise_scale) * noise
    candidate_norm = candidate_direction.norm(dim=-1, keepdim=True)
    fallback_direction = F.normalize(noise, dim=-1)
    direction = torch.where(
        candidate_norm > 1e-6,
        F.normalize(candidate_direction, dim=-1),
        fallback_direction,
    )
    base = class_means[labels].detach()
    base = torch.where(use_fallback, fallback_proxies, base)
    base_norm = base.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    return base + float(tail_scale) * base_norm * direction


def synthesize_gaussian_virtual_outliers(
    labels: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    tail_scale: float = 2.0,
) -> torch.Tensor:
    """Sample class-conditional low-likelihood feature tails.

    ``means`` and ``variances`` are estimated from known training features and
    treated as fixed statistics for the current epoch. A normalized random
    direction is scaled by the class-wise diagonal standard deviation and a
    tail multiplier, which avoids the very large norm produced by sampling an
    unnormalized 512-dimensional Gaussian vector. This is a lightweight
    VOS-style approximation, not a full VOS implementation.
    """
    if means.ndim != 2 or variances.shape != means.shape:
        raise ValueError("means and variances must have shape [num_classes, feature_dim]")
    if labels.numel() == 0:
        return means.new_empty((0, means.size(1)))
    if labels.min().item() < 0 or labels.max().item() >= means.size(0):
        raise ValueError("labels contain an invalid Gaussian class index")
    class_mean = means[labels]
    class_std = variances[labels].clamp_min(1e-6).sqrt()
    direction = F.normalize(torch.randn_like(class_mean), dim=-1)
    return class_mean + float(tail_scale) * direction * class_std


def _rank_normalize(score: torch.Tensor) -> torch.Tensor:
    """Map a 1D score to [0, 1] ranks while preserving high-risk ordering."""
    if score.numel() <= 1:
        return torch.ones_like(score)
    order = torch.argsort(score, descending=False)
    ranks = torch.empty_like(score, dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, steps=score.numel(), device=score.device)
    return ranks


def select_discovery_candidates(
    logits: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    ratio: float = 0.25,
    mode: str = "entropy",
    features: torch.Tensor | None = None,
    prototypes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select high-risk unlabeled discovery samples without using labels."""
    batch_size = logits.size(0)
    if batch_size == 0 or ratio <= 0.0:
        return torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
    with torch.no_grad():
        probs = logits.softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        max_softmax_risk = 1.0 - probs.max(dim=-1).values
        energy_risk = -torch.logsumexp(logits, dim=-1)
        prototype_risk = None
        if mode in {"prototype_distance", "distance_consensus"}:
            if features is None or prototypes is None:
                raise ValueError(
                    f"{mode} candidate selection requires features and prototypes"
                )
            normalized_features = F.normalize(features.detach(), dim=-1)
            normalized_prototypes = F.normalize(prototypes.detach(), dim=-1)
            prototype_risk = 1.0 - (
                normalized_features @ normalized_prototypes.T
            ).max(dim=-1).values
        if mode == "prototype_distance":
            score = prototype_risk
        elif mode == "distance_consensus":
            components = [
                _rank_normalize(prototype_risk),
                _rank_normalize(entropy),
                _rank_normalize(max_softmax_risk),
            ]
            if uncertainty is not None:
                components.append(_rank_normalize(uncertainty))
            score = torch.stack(components, dim=0).mean(dim=0)
        elif mode == "entropy":
            score = entropy
        elif mode == "max_softmax":
            score = max_softmax_risk
        elif mode == "energy":
            score = energy_risk
        elif mode == "entropy_uncertainty":
            score = entropy if uncertainty is None else entropy + uncertainty
        elif mode == "consensus":
            count = max(1, min(batch_size, int(np.ceil(batch_size * ratio))))
            scores = [entropy, max_softmax_risk, energy_risk]
            if uncertainty is not None:
                scores.append(uncertainty)
            top_masks = []
            for item in scores:
                item_indices = torch.topk(item, k=count, largest=True).indices
                item_mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
                item_mask[item_indices] = True
                top_masks.append(item_mask)
            votes = torch.stack(top_masks, dim=0).sum(dim=0)
            required_votes = len(scores) if len(scores) <= 3 else len(scores) - 1
            mask = votes >= required_votes
            max_selected = count
            min_selected = max(1, min(batch_size, int(np.ceil(count * 0.25))))
            composite = _rank_normalize(entropy) + _rank_normalize(max_softmax_risk) + _rank_normalize(energy_risk)
            if uncertainty is not None:
                composite = composite + _rank_normalize(uncertainty)
            selected = int(mask.sum().item())
            if selected > max_selected:
                selected_indices = torch.topk(composite.masked_fill(~mask, float("-inf")), k=max_selected, largest=True).indices
                mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
                mask[selected_indices] = True
            elif selected < min_selected:
                fill_score = votes.float() * (len(scores) + 1.0) + _rank_normalize(composite)
                fill_indices = torch.topk(fill_score, k=min_selected, largest=True).indices
                mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
                mask[fill_indices] = True
            return mask
        else:
            raise ValueError(f"Unsupported discovery selection mode: {mode}")
        count = max(1, min(batch_size, int(np.ceil(batch_size * ratio))))
        indices = torch.topk(score, k=count, largest=True).indices
        mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
        mask[indices] = True
    return mask


@torch.no_grad()
def select_joint_candidates(
    known_logits: torch.Tensor,
    novel_logits: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    known_prototypes: torch.Tensor | None = None,
    ratio: float = 0.25,
    mode: str = "consensus",
    known_temperature: float = 1.0,
) -> torch.Tensor:
    """Select joint-discovery candidates without using discovery labels.

    ``novel_mass`` uses the posterior mass of the novel subspace rather than
    the entropy of the known classifier.  This avoids treating every hard
    known example as a pseudo-novel sample.
    """
    if mode in {"prototype_distance", "distance_consensus"}:
        if features is None or known_prototypes is None:
            raise ValueError(
                "prototype_distance candidate selection requires features and known_prototypes"
            )
        if features.size(0) == 0:
            return torch.zeros(0, dtype=torch.bool, device=known_logits.device)
        normalized_features = F.normalize(features.detach(), dim=-1)
        normalized_prototypes = F.normalize(known_prototypes.detach(), dim=-1)
        prototype_risk = 1.0 - (normalized_features @ normalized_prototypes.T).max(dim=-1).values
        if mode == "distance_consensus":
            probs = known_logits.softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
            max_softmax_risk = 1.0 - probs.max(dim=-1).values
            components = [
                _rank_normalize(prototype_risk),
                _rank_normalize(entropy),
                _rank_normalize(max_softmax_risk),
            ]
            if uncertainty is not None:
                components.append(_rank_normalize(uncertainty))
            risk = torch.stack(components, dim=0).mean(dim=0)
        else:
            risk = prototype_risk
        count = max(1, min(known_logits.size(0), int(np.ceil(known_logits.size(0) * ratio))))
        indices = torch.topk(risk, k=count, largest=True).indices
        mask = torch.zeros(known_logits.size(0), dtype=torch.bool, device=known_logits.device)
        mask[indices] = True
        return mask
    if mode != "novel_mass":
        return select_discovery_candidates(
            known_logits,
            uncertainty,
            ratio=ratio,
            mode=mode,
            features=features,
            prototypes=known_prototypes,
        )
    if known_logits.size(0) == 0:
        return torch.zeros(0, dtype=torch.bool, device=known_logits.device)
    unified = combine_known_novel_logits(
        known_logits, novel_logits, known_temperature=known_temperature
    )
    novel_mass = unified.softmax(dim=-1)[:, known_logits.size(-1):].sum(dim=-1)
    count = max(1, min(known_logits.size(0), int(np.ceil(known_logits.size(0) * ratio))))
    indices = torch.topk(novel_mass, k=count, largest=True).indices
    mask = torch.zeros(known_logits.size(0), dtype=torch.bool, device=known_logits.device)
    mask[indices] = True
    return mask


def filter_discovery_candidates_by_neighbors(
    mask: torch.Tensor,
    features: torch.Tensor,
    k: int = 5,
    min_votes: int = 2,
) -> tuple[torch.Tensor, float]:
    """Keep selected discovery samples whose nearest neighbors are also selected.

    The filter follows the SCAN/AutoNovel-style intuition that reliable novel
    candidates should not only look risky by themselves, but should also live
    near other risky samples in representation space.
    """
    agreement = compute_neighbor_agreement(mask, features, k=k)
    batch_size = mask.numel()
    if batch_size <= 1 or k <= 0:
        return mask, agreement[mask].mean().item() if mask.any() else 0.0
    with torch.no_grad():
        effective_k = min(int(k), batch_size - 1)
        required_votes = min(max(int(min_votes), 1), effective_k)
        filtered = mask & (agreement * effective_k >= required_votes)
        selected_agreement = agreement[mask].mean().item() if mask.any() else 0.0
    return filtered, selected_agreement


@torch.no_grad()
def compute_neighbor_agreement(
    mask: torch.Tensor,
    features: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Return the fraction of selected kNN neighbors for every sample."""
    batch_size = mask.numel()
    agreement = torch.zeros(batch_size, dtype=torch.float32, device=mask.device)
    if batch_size <= 1 or k <= 0:
        return agreement
    effective_k = min(int(k), batch_size - 1)
    normalized = F.normalize(features.detach().float(), dim=-1)
    similarity = torch.matmul(normalized, normalized.T)
    similarity.fill_diagonal_(float("-inf"))
    neighbor_indices = torch.topk(similarity, k=effective_k, dim=1, largest=True).indices
    return mask[neighbor_indices].float().mean(dim=1)


@torch.no_grad()
def compute_discovery_candidate_weights(
    logits: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    student_logits: torch.Tensor | None = None,
    student_uncertainty: torch.Tensor | None = None,
    prototypes: torch.Tensor | None = None,
    ratio: float = 0.25,
    mode: str = "consensus",
    neighbor_k: int = 5,
    neighbor_temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute soft candidate weights without using discovery labels.

    The hard candidate mask remains the gate. Selected samples receive a
    continuous weight based on risk strength, local neighbor agreement, and
    optional EMA/student agreement. This follows FixMatch-style confidence
    weighting while using SCAN/AutoNovel-style local structure.
    """
    mask = select_discovery_candidates(
        logits,
        uncertainty,
        ratio=ratio,
        mode=mode,
        features=features,
        prototypes=prototypes,
    )
    weights = torch.zeros_like(logits[:, 0], dtype=torch.float32)
    if not mask.any():
        return mask, weights, 0.0
    probs = logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    max_softmax_risk = 1.0 - probs.max(dim=-1).values
    energy_risk = -torch.logsumexp(logits, dim=-1)
    if mode in {"prototype_distance", "distance_consensus"}:
        if features is None or prototypes is None:
            raise ValueError(
                f"{mode} candidate weighting requires features and prototypes"
            )
        normalized_features = F.normalize(features.detach(), dim=-1)
        normalized_prototypes = F.normalize(prototypes.detach(), dim=-1)
        prototype_risk = 1.0 - (
            normalized_features @ normalized_prototypes.T
        ).max(dim=-1).values
    else:
        prototype_risk = None
    if mode == "prototype_distance":
        risk = prototype_risk
    elif mode == "distance_consensus":
        components = [
            _rank_normalize(prototype_risk),
            _rank_normalize(entropy),
            _rank_normalize(max_softmax_risk),
        ]
        if uncertainty is not None:
            components.append(_rank_normalize(uncertainty))
        risk = torch.stack(components, dim=0).mean(dim=0)
    elif mode == "entropy":
        risk = entropy
    elif mode == "max_softmax":
        risk = max_softmax_risk
    elif mode == "energy":
        risk = energy_risk
    elif mode == "entropy_uncertainty":
        risk = entropy if uncertainty is None else entropy + uncertainty
    elif mode == "consensus":
        components = [entropy, max_softmax_risk, energy_risk]
        if uncertainty is not None:
            components.append(uncertainty)
        risk = torch.stack([_rank_normalize(item) for item in components], dim=0).mean(dim=0)
    else:
        raise ValueError(f"Unsupported discovery selection mode: {mode}")
    risk = _rank_normalize(risk)
    candidate_weight = 0.5 + 0.5 * risk
    neighbor_agreement = torch.zeros_like(candidate_weight)
    if features is not None:
        neighbor_agreement = compute_neighbor_agreement(mask, features, k=neighbor_k)
        temperature = max(float(neighbor_temperature), 1e-6)
        # A smooth gate keeps isolated candidates trainable, but downweights them.
        candidate_weight = candidate_weight * torch.sigmoid(
            (neighbor_agreement - 0.5) / temperature
        )
    if student_logits is not None:
        student_mask = select_discovery_candidates(
            student_logits,
            student_uncertainty,
            ratio=ratio,
            mode=mode,
            features=features,
            prototypes=prototypes,
        )
        agreement = (student_mask == mask).float()
        candidate_weight = candidate_weight * (0.5 + 0.5 * agreement)
    weights[mask] = candidate_weight[mask].clamp(0.0, 1.0)
    selected_agreement = neighbor_agreement[mask].mean().item() if mask.any() else 0.0
    return mask, weights, selected_agreement


@torch.no_grad()
def compute_joint_candidate_weights(
    logits: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    novel_logits: torch.Tensor | None = None,
    ratio: float = 0.25,
    mode: str = "entropy_uncertainty",
    floor: float = 0.05,
    known_temperature: float = 1.0,
) -> torch.Tensor:
    """Return continuous risk weights for joint discovery without labels."""
    if logits.size(0) == 0:
        return logits.new_empty((0,))
    if mode == "novel_mass":
        if novel_logits is None:
            raise ValueError("novel_mass candidate weighting requires novel_logits")
        unified = combine_known_novel_logits(
            logits, novel_logits, known_temperature=known_temperature
        )
        risk = _rank_normalize(
            unified.softmax(dim=-1)[:, logits.size(-1):].sum(dim=-1)
        )
        ratio = min(max(float(ratio), 1e-3), 1.0)
        cutoff = torch.quantile(risk.detach(), 1.0 - ratio)
        high_risk = ((risk - cutoff) / (1.0 - cutoff).clamp_min(1e-6)).clamp(0.0, 1.0)
        floor = min(max(float(floor), 0.0), 1.0)
        return floor + (1.0 - floor) * high_risk
    probs = logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    max_softmax_risk = 1.0 - probs.max(dim=-1).values
    energy_risk = -torch.logsumexp(logits, dim=-1)
    if mode == "entropy":
        risk = entropy
    elif mode == "max_softmax":
        risk = max_softmax_risk
    elif mode == "energy":
        risk = energy_risk
    elif mode == "entropy_uncertainty":
        risk = entropy if uncertainty is None else entropy + uncertainty
    elif mode == "consensus":
        components = [entropy, max_softmax_risk, energy_risk]
        if uncertainty is not None:
            components.append(uncertainty)
        risk = torch.stack([_rank_normalize(item) for item in components], dim=0).mean(dim=0)
    else:
        raise ValueError(f"Unsupported discovery selection mode: {mode}")
    risk = _rank_normalize(risk)
    ratio = min(max(float(ratio), 1e-3), 1.0)
    cutoff = torch.quantile(risk.detach(), 1.0 - ratio)
    high_risk = ((risk - cutoff) / (1.0 - cutoff).clamp_min(1e-6)).clamp(0.0, 1.0)
    floor = min(max(float(floor), 0.0), 1.0)
    return floor + (1.0 - floor) * high_risk


def update_ema_model(ema_model, model, decay: float = 0.99):
    """Update a non-gradient EMA copy used for stable discovery selection."""
    decay = min(max(float(decay), 0.0), 0.999999)
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        for name, parameter in model.named_parameters():
            ema_params[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
        ema_buffers = dict(ema_model.named_buffers())
        for name, buffer in model.named_buffers():
            # BatchNorm statistics are copied from the current student instead of
            # being exponentially averaged, matching common EMA implementations.
            ema_buffers[name].copy_(buffer.detach())
    ema_model.eval()


def train_one_epoch_teacher(
    model,
    loader,
    optimizer,
    device,
    alpha_unc: float = 0.1,
    alpha_proto: float = 0.0,
    alpha_proto_repulsion: float = 0.0,
    proto_repulsion_margin: float = 0.0,
    alpha_proxy: float = 0.0,
    alpha_pseudo: float = 0.0,
    alpha_energy: float = 0.0,
    pseudo_mode: str = "strong",
    pseudo_feature_noise: float = 0.05,
    uncertainty_target_mode: str = "confidence",
    energy_margin: float = 1.0,
    energy_temperature: float = 1.0,
    proxy_temperature: float = 0.1,
    alpha_angular: float = 0.0,
    angular_margin: float = 0.2,
    angular_scale: float = 16.0,
    angular_correct_only: bool = False,
    alpha_proxy_anchor: float = 0.0,
    proxy_anchor_alpha: float = 32.0,
    proxy_anchor_margin: float = 0.1,
    reciprocal_points=None,
    alpha_reciprocal: float = 0.0,
    reciprocal_margin: float = 0.2,
    freeze_bn_stats: bool = False,
):
    model.train()
    if freeze_bn_stats:
        freeze_batchnorm_stats(model)
    ce_meter = AverageMeter()
    unc_meter = AverageMeter()
    proto_meter = AverageMeter()
    proto_repulsion_meter = AverageMeter()
    proxy_meter = AverageMeter()
    pseudo_meter = AverageMeter()
    energy_meter = AverageMeter()
    angular_meter = AverageMeter()
    angular_active_meter = AverageMeter()
    proxy_anchor_meter = AverageMeter()
    reciprocal_meter = AverageMeter()
    for batch in tqdm(loader, desc="teacher-train", leave=False):
        images, labels, raw_labels, is_known, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss_ce = classification_loss(out["logits"], labels)
        loss_unc = uncertainty_alignment_loss(
            out["uncertainty"], out["logits"], labels, target_mode=uncertainty_target_mode
        )
        loss_proto = prototype_alignment_loss(out["features"], labels, model.classifier.weight)
        loss_proto_repulsion = prototype_repulsion_loss(
            model.classifier.weight, similarity_margin=proto_repulsion_margin
        )
        loss_proxy = proxy_contrastive_loss(
            out["features"], labels, model.classifier.weight, temperature=proxy_temperature
        )
        pseudo_images = make_pseudo_unknown(images, mode=pseudo_mode)
        pseudo_out = model(pseudo_images)
        pseudo_features = pseudo_out["features"] + pseudo_feature_noise * torch.randn_like(pseudo_out["features"])
        pseudo_logits, pseudo_uncertainty = pseudo_forward_from_features(model, pseudo_features)
        loss_pseudo = pseudo_unknown_loss(pseudo_logits, pseudo_uncertainty)
        loss_energy = energy_margin_loss(
            out["logits"], pseudo_logits, margin=energy_margin, temperature=energy_temperature
        )
        angular_mask = (
            out["logits"].argmax(dim=-1).eq(labels)
            if angular_correct_only
            else None
        )
        loss_angular = angular_margin_loss(
            out["features"], labels, model.classifier.weight,
            margin=angular_margin, scale=angular_scale,
            sample_mask=angular_mask,
        )
        angular_active_meter.update(
            angular_mask.float().mean().item() if angular_mask is not None else 1.0,
            images.size(0),
        )
        loss_proxy_anchor = proxy_anchor_loss(
            out["features"], labels, model.classifier.weight,
            alpha=proxy_anchor_alpha, margin=proxy_anchor_margin,
        )
        loss_reciprocal = out["logits"].new_tensor(0.0)
        loss = (
            loss_ce
            + alpha_unc * loss_unc
            + alpha_proto * loss_proto
            + alpha_proto_repulsion * loss_proto_repulsion
            + alpha_proxy * loss_proxy
            + alpha_pseudo * loss_pseudo
            + alpha_energy * loss_energy
            + alpha_angular * loss_angular
            + alpha_proxy_anchor * loss_proxy_anchor
            + alpha_reciprocal * loss_reciprocal
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_meter.update(loss_ce.item(), images.size(0))
        unc_meter.update(loss_unc.item(), images.size(0))
        proto_meter.update(loss_proto.item(), images.size(0))
        proto_repulsion_meter.update(loss_proto_repulsion.item(), images.size(0))
        proxy_meter.update(loss_proxy.item(), images.size(0))
        pseudo_meter.update(loss_pseudo.item(), images.size(0))
        energy_meter.update(loss_energy.item(), images.size(0))
        angular_meter.update(loss_angular.item(), images.size(0))
        proxy_anchor_meter.update(loss_proxy_anchor.item(), images.size(0))
        reciprocal_meter.update(loss_reciprocal.item(), images.size(0))
    return {
        "ce": ce_meter.avg,
        "unc": unc_meter.avg,
        "proto": proto_meter.avg,
        "proto_repulsion": proto_repulsion_meter.avg,
        "proxy": proxy_meter.avg,
        "pseudo": pseudo_meter.avg,
        "energy": energy_meter.avg,
        "angular": angular_meter.avg,
        "angular_active_ratio": angular_active_meter.avg,
        "proxy_anchor": proxy_anchor_meter.avg,
        "reciprocal": reciprocal_meter.avg,
    }


@torch.no_grad()
def teacher_kd_uncertainty(
    teacher,
    images: torch.Tensor,
    source: str = "head",
    mc_samples: int = 4,
):
    """Return teacher outputs and the uncertainty used by KD.

    The historical ``head`` source is the learned uncertainty head.  The MC
    sources use stochastic dropout passes and normalize entropy by log(K),
    making the result comparable to the head's [0, 1] output.  This helper is
    deliberately separate so the uncertainty source can be unit-tested and
    audited independently from the large student-training loop.
    """
    if source not in {"head", "mc_epistemic", "mc_predictive_entropy"}:
        raise ValueError(
            "source must be 'head', 'mc_epistemic', or 'mc_predictive_entropy'"
        )
    if source != "head" and int(mc_samples) < 2:
        raise ValueError("mc_samples must be at least 2 for MC uncertainty")
    outputs = teacher(images)
    logits = outputs["logits"]
    uncertainty = outputs["uncertainty"]
    if source == "head":
        return outputs, logits, uncertainty
    mc = teacher.mc_predict(images, mc_samples=int(mc_samples))
    logits = mc["mean_logits"]
    normalizer = max(float(outputs["logits"].size(-1)), 2.0)
    if source == "mc_epistemic":
        uncertainty = mc["epistemic"]
    else:
        uncertainty = mc["predictive_entropy"]
    uncertainty = (
        uncertainty / outputs["logits"].new_tensor(normalizer).log()
    ).clamp(0.0, 1.0)
    return outputs, logits, uncertainty


def train_one_epoch_student(
    student,
    teacher,
    loader,
    optimizer,
    device,
    alpha_unc: float = 0.1,
    alpha_kd: float = 1.0,
    alpha_feat_kd: float = 0.0,
    alpha_supcon: float = 0.1,
    alpha_raw_supcon: float = 0.0,
    alpha_proto: float = 0.0,
    alpha_proto_repulsion: float = 0.0,
    proto_repulsion_margin: float = 0.0,
    alpha_proxy: float = 0.0,
    alpha_pseudo: float = 0.0,
    alpha_energy: float = 0.0,
    alpha_vos: float = 0.0,
    vos_mode: str = "gaussian",
    vos_gaussian_stats: Dict[str, torch.Tensor] | None = None,
    vos_tail_scale: float = 1.5,
    vos_noise_scale: float = 0.25,
    vos_uniform_weight: float = 0.25,
    pseudo_mode: str = "strong",
    pseudo_feature_noise: float = 0.05,
    uncertainty_target_mode: str = "confidence",
    temperature: float = 2.0,
    kd_mode: str = "uncertainty",
    kd_uncertainty_source: str = "head",
    kd_mc_samples: int = 4,
    uncertainty_weight_mode: str = "raw",
    uncertainty_weight_min: float | None = None,
    uncertainty_weight_max: float | None = None,
    discovery_loader=None,
    alpha_discovery: float = 0.0,
    alpha_discovery_unknown: float = 0.0,
    alpha_discovery_energy: float = 0.0,
    alpha_discovery_uniform: float = 0.0,
    alpha_discovery_feature_margin: float = 0.0,
    discovery_feature_margin: float = 0.2,
    alpha_discovery_objectosphere: float = 0.0,
    objectosphere_known_radius: float = 10.0,
    objectosphere_unknown_weight: float = 1.0,
    alpha_discovery_feature_separation: float = 0.0,
    discovery_feature_separation_margin: float = 0.0,
    discovery_feature_separation_temperature: float = 0.1,
    discovery_feature_candidate_gating: bool = False,
    discovery_cross_view_gating: bool = False,
    discovery_pool_mode: str = "unknown",
    alpha_discovery_boundary: float = 0.0,
    discovery_boundary_margin: float = 0.2,
    discovery_boundary_prototype_weight: float = 0.5,
    discovery_boundary_temperature: float = 0.1,
    alpha_discovery_knn_boundary: float = 0.0,
    discovery_knn_support=None,
    discovery_knn_k: int = 5,
    discovery_knn_margin: float = 0.02,
    discovery_knn_soft_weighting: bool = False,
    alpha_discovery_uncertainty_separation: float = 0.0,
    discovery_uncertainty_loss: str = "bce",
    discovery_uncertainty_margin: float = 0.1,
    alpha_discovery_uncertainty_pu: float = 0.0,
    discovery_uncertainty_known_prior: float = 0.2,
    alpha_discovery_selective_unknown: float = 0.0,
    alpha_discovery_selective_energy: float = 0.0,
    alpha_angular: float = 0.0,
    angular_margin: float = 0.2,
    angular_scale: float = 16.0,
    angular_correct_only: bool = False,
    alpha_proxy_anchor: float = 0.0,
    proxy_anchor_alpha: float = 32.0,
    proxy_anchor_margin: float = 0.1,
    reciprocal_points=None,
    alpha_reciprocal: float = 0.0,
    reciprocal_margin: float = 0.2,
    discovery_select_ratio: float = 0.25,
    discovery_select_mode: str = "entropy_uncertainty",
    discovery_loss_mode: str = "nt_xent",
    discovery_temperature: float = 0.2,
    energy_margin: float = 1.0,
    energy_temperature: float = 1.0,
    proxy_temperature: float = 0.1,
    discovery_selection_model=None,
    discovery_selection_model_updates_ema: bool = True,
    discovery_ema_decay: float = 0.99,
    discovery_neighbor_filter: bool = False,
    discovery_neighbor_k: int = 5,
    discovery_neighbor_min_votes: int = 2,
    discovery_soft_weighting: bool = False,
    discovery_neighbor_temperature: float = 0.5,
    novel_head=None,
    alpha_joint_discovery: float = 0.0,
    joint_confidence_threshold: float = 0.6,
    joint_assignment_temperature: float = 1.0,
    alpha_joint_consistency: float = 1.0,
    alpha_joint_balance: float = 0.1,
    alpha_joint_information: float = 0.1,
    alpha_joint_neighbor: float = 0.1,
    alpha_joint_pseudo: float = 0.0,
    alpha_joint_gate: float = 0.0,
    joint_gate_margin: float = 0.1,
    joint_neighbor_k: int = 5,
    alpha_joint_proto_repulsion: float = 0.0,
    joint_proto_repulsion_margin: float = 0.0,
    joint_space: str = "novel",
    alpha_joint_known_ce: float = 0.1,
    joint_known_temperature: float = 1.0,
    joint_mixed_residual: bool = False,
    joint_residual_temperature: float = 1.0,
    joint_residual_floor: float = 0.0,
    joint_novel_mass: bool = False,
    joint_novel_weight_floor: float = 0.0,
    joint_novel_neighbor_support: bool = False,
    joint_novel_neighbor_k: int = 5,
    joint_novel_ema_weights: bool = False,
    joint_weighted_sinkhorn: bool = True,
    alpha_joint_memory_neighbor: float = 0.0,
    joint_memory_size: int = 4096,
    joint_memory_k: int = 5,
    joint_memory_temperature: float = 0.2,
    joint_memory_warmup_size: int = 256,
    joint_memory_bank: dict | None = None,
    alpha_joint_novel_margin: float = 0.0,
    joint_novel_margin: float = 0.2,
    joint_candidate_gating: bool = False,
    joint_candidate_ratio: float = 0.25,
    joint_candidate_mode: str = "consensus",
    joint_candidate_neighbor_filter: bool = False,
    joint_candidate_neighbor_k: int = 5,
    joint_candidate_min_votes: int = 2,
    joint_candidate_soft_weighting: bool = False,
    joint_candidate_weight_floor: float = 0.05,
    joint_mixed_known_consistency: bool = False,
    alpha_joint_known_consistency: float = 0.0,
    joint_known_confidence_threshold: float = 0.8,
    joint_known_target: str = "ema",
    outlier_loader=None,
    alpha_outlier_uniform: float = 0.0,
    alpha_outlier_energy: float = 0.0,
    alpha_outlier_uncertainty: float = 0.0,
    alpha_outlier_feature_margin: float = 0.0,
    outlier_feature_margin: float = 0.2,
    freeze_bn_stats: bool = False,
):
    if kd_uncertainty_source not in {"head", "mc_epistemic", "mc_predictive_entropy"}:
        raise ValueError(
            "kd_uncertainty_source must be 'head', 'mc_epistemic', "
            "or 'mc_predictive_entropy'"
        )
    if kd_uncertainty_source != "head" and int(kd_mc_samples) < 2:
        raise ValueError("kd_mc_samples must be at least 2 for MC uncertainty")
    student.train()
    if freeze_bn_stats:
        freeze_batchnorm_stats(student)
    teacher.eval()
    ce_meter = AverageMeter()
    kd_meter = AverageMeter()
    feat_kd_meter = AverageMeter()
    unc_meter = AverageMeter()
    sc_meter = AverageMeter()
    raw_sc_meter = AverageMeter()
    proto_meter = AverageMeter()
    proto_repulsion_meter = AverageMeter()
    proxy_meter = AverageMeter()
    pseudo_meter = AverageMeter()
    energy_meter = AverageMeter()
    vos_meter = AverageMeter()
    discovery_meter = AverageMeter()
    discovery_unknown_meter = AverageMeter()
    discovery_energy_meter = AverageMeter()
    discovery_uniform_meter = AverageMeter()
    discovery_feature_margin_meter = AverageMeter()
    discovery_objectosphere_meter = AverageMeter()
    discovery_feature_separation_meter = AverageMeter()
    discovery_boundary_meter = AverageMeter()
    discovery_knn_boundary_meter = AverageMeter()
    discovery_uncertainty_separation_meter = AverageMeter()
    discovery_uncertainty_pu_meter = AverageMeter()
    discovery_selective_unknown_meter = AverageMeter()
    discovery_selective_energy_meter = AverageMeter()
    angular_meter = AverageMeter()
    angular_active_meter = AverageMeter()
    proxy_anchor_meter = AverageMeter()
    reciprocal_meter = AverageMeter()
    discovery_selected_meter = AverageMeter()
    discovery_raw_selected_meter = AverageMeter()
    discovery_neighbor_agreement_meter = AverageMeter()
    discovery_weight_mean_meter = AverageMeter()
    feature_candidate_ratio_meter = AverageMeter()
    feature_candidate_weight_mean_meter = AverageMeter()
    joint_meter = AverageMeter()
    joint_consistency_meter = AverageMeter()
    joint_balance_meter = AverageMeter()
    joint_information_meter = AverageMeter()
    joint_neighbor_meter = AverageMeter()
    joint_memory_neighbor_meter = AverageMeter()
    joint_pseudo_meter = AverageMeter()
    joint_gate_meter = AverageMeter()
    joint_proto_repulsion_meter = AverageMeter()
    joint_known_ce_meter = AverageMeter()
    joint_candidate_meter = AverageMeter()
    joint_known_consistency_meter = AverageMeter()
    joint_residual_weight_meter = AverageMeter()
    joint_novel_mass_weight_meter = AverageMeter()
    joint_novel_weight_min_meter = AverageMeter()
    joint_novel_margin_meter = AverageMeter()
    joint_novel_neighbor_support_meter = AverageMeter()
    outlier_uniform_meter = AverageMeter()
    outlier_energy_meter = AverageMeter()
    outlier_uncertainty_meter = AverageMeter()
    outlier_feature_margin_meter = AverageMeter()
    discovery_iter = iter(discovery_loader) if discovery_loader is not None else None
    if joint_memory_bank is None:
        joint_memory_bank = {}
    joint_memory_features = joint_memory_bank.get("features")
    joint_memory_logits = joint_memory_bank.get("logits")
    outlier_iter = iter(outlier_loader) if outlier_loader is not None else None
    for batch in tqdm(loader, desc="student-train", leave=False):
        images, labels, raw_labels, is_known, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            t_out, teacher_kd_logits, teacher_kd_uncertainty_value = teacher_kd_uncertainty(
                teacher,
                images,
                source=kd_uncertainty_source,
                mc_samples=kd_mc_samples,
            )
        s_out = student(images)
        loss_ce = classification_loss(s_out["logits"], labels)
        loss_unc = uncertainty_alignment_loss(
            s_out["uncertainty"], s_out["logits"], labels, target_mode=uncertainty_target_mode
        )
        loss_kd = distillation_loss(
            s_out["logits"],
            teacher_kd_logits,
            teacher_uncertainty=teacher_kd_uncertainty_value,
            temperature=temperature,
            uncertainty_weighted=(kd_mode == "uncertainty"),
            uncertainty_weight_mode=uncertainty_weight_mode,
            uncertainty_weight_min=uncertainty_weight_min,
            uncertainty_weight_max=uncertainty_weight_max,
        )
        feature_teacher_uncertainty = (
            teacher_kd_uncertainty_value if kd_mode == "uncertainty" else None
        )
        loss_feat_kd = feature_distillation_loss(
            s_out["proj"],
            t_out["proj"],
            teacher_uncertainty=feature_teacher_uncertainty,
            uncertainty_weight_mode=uncertainty_weight_mode,
            uncertainty_weight_min=uncertainty_weight_min,
            uncertainty_weight_max=uncertainty_weight_max,
        )
        loss_supcon = supervised_contrastive_loss(s_out["proj"], labels)
        loss_raw_supcon = supervised_contrastive_loss(s_out["features"], labels)
        loss_proto = prototype_alignment_loss(s_out["features"], labels, student.classifier.weight)
        loss_proto_repulsion = prototype_repulsion_loss(
            student.classifier.weight, similarity_margin=proto_repulsion_margin
        )
        loss_proxy = proxy_contrastive_loss(
            s_out["features"], labels, student.classifier.weight, temperature=proxy_temperature
        )
        pseudo_images = make_pseudo_unknown(images, mode=pseudo_mode)
        pseudo_out = student(pseudo_images)
        pseudo_features = pseudo_out["features"] + pseudo_feature_noise * torch.randn_like(pseudo_out["features"])
        pseudo_logits, pseudo_uncertainty = pseudo_forward_from_features(student, pseudo_features)
        loss_pseudo = pseudo_unknown_loss(pseudo_logits, pseudo_uncertainty)
        loss_energy = energy_margin_loss(
            s_out["logits"], pseudo_logits, margin=energy_margin, temperature=energy_temperature
        )
        loss_vos = s_out["logits"].new_tensor(0.0)
        if alpha_vos > 0.0:
            if vos_mode == "gaussian" and vos_gaussian_stats is not None:
                virtual_features = synthesize_gaussian_virtual_outliers(
                    labels,
                    vos_gaussian_stats["means"],
                    vos_gaussian_stats["variances"],
                    tail_scale=vos_tail_scale,
                )
            elif vos_mode == "batch_center":
                virtual_features = synthesize_virtual_outliers(
                    s_out["features"],
                    labels,
                    student.classifier.weight,
                    tail_scale=vos_tail_scale,
                    noise_scale=vos_noise_scale,
                )
            else:
                raise ValueError(
                    "Gaussian VOS requires vos_gaussian_stats; use batch_center "
                    "for the online fallback."
                )
            virtual_logits, _ = pseudo_forward_from_features(
                student, virtual_features, stochastic=True
            )
            loss_vos = energy_margin_loss(
                s_out["logits"],
                virtual_logits,
                margin=energy_margin,
                temperature=energy_temperature,
            )
            if vos_uniform_weight > 0.0:
                loss_vos = loss_vos + float(vos_uniform_weight) * outlier_exposure_uniform_loss(
                    virtual_logits
                )
        angular_mask = (
            s_out["logits"].argmax(dim=-1).eq(labels)
            if angular_correct_only
            else None
        )
        loss_angular = angular_margin_loss(
            s_out["features"], labels, student.classifier.weight,
            margin=angular_margin, scale=angular_scale,
            sample_mask=angular_mask,
        )
        angular_active_meter.update(
            angular_mask.float().mean().item() if angular_mask is not None else 1.0,
            images.size(0),
        )
        loss_proxy_anchor = proxy_anchor_loss(
            s_out["features"], labels, student.classifier.weight,
            alpha=proxy_anchor_alpha, margin=proxy_anchor_margin,
        )
        loss_reciprocal = s_out["logits"].new_tensor(0.0)
        loss_discovery = s_out["logits"].new_tensor(0.0)
        loss_discovery_unknown = s_out["logits"].new_tensor(0.0)
        loss_discovery_energy = s_out["logits"].new_tensor(0.0)
        loss_discovery_uniform = s_out["logits"].new_tensor(0.0)
        loss_discovery_feature_margin = s_out["logits"].new_tensor(0.0)
        loss_discovery_objectosphere = s_out["logits"].new_tensor(0.0)
        loss_discovery_feature_separation = s_out["logits"].new_tensor(0.0)
        loss_discovery_boundary = s_out["logits"].new_tensor(0.0)
        loss_discovery_knn_boundary = s_out["logits"].new_tensor(0.0)
        loss_discovery_uncertainty_separation = s_out["logits"].new_tensor(0.0)
        loss_discovery_uncertainty_pu = s_out["logits"].new_tensor(0.0)
        loss_discovery_selective_unknown = s_out["logits"].new_tensor(0.0)
        loss_discovery_selective_energy = s_out["logits"].new_tensor(0.0)
        loss_joint_discovery = s_out["logits"].new_tensor(0.0)
        joint_consistency = s_out["logits"].new_tensor(0.0)
        joint_balance = s_out["logits"].new_tensor(0.0)
        joint_information = s_out["logits"].new_tensor(0.0)
        joint_neighbor = s_out["logits"].new_tensor(0.0)
        joint_memory_neighbor = s_out["logits"].new_tensor(0.0)
        joint_pseudo = s_out["logits"].new_tensor(0.0)
        joint_gate = s_out["logits"].new_tensor(0.0)
        joint_proto_repulsion = s_out["logits"].new_tensor(0.0)
        loss_joint_known_ce = s_out["logits"].new_tensor(0.0)
        loss_joint_known_consistency = s_out["logits"].new_tensor(0.0)
        loss_joint_novel_margin = s_out["logits"].new_tensor(0.0)
        joint_candidate_ratio_value = 0.0
        joint_residual_weight_value = 0.0
        joint_novel_mass_weight_value = 0.0
        joint_novel_margin_value = 0.0
        joint_novel_neighbor_support_value = 0.0
        joint_sample_weights = None
        first_selection_out = None
        second_selection_out = None
        main_novel_logits = None
        loss_outlier_uniform = s_out["logits"].new_tensor(0.0)
        loss_outlier_energy = s_out["logits"].new_tensor(0.0)
        loss_outlier_uncertainty = s_out["logits"].new_tensor(0.0)
        loss_outlier_feature_margin = s_out["logits"].new_tensor(0.0)
        if outlier_iter is not None and (
            alpha_outlier_uniform > 0.0
            or alpha_outlier_energy > 0.0
            or alpha_outlier_uncertainty > 0.0
            or alpha_outlier_feature_margin > 0.0
        ):
            try:
                outlier_images, _ = next(outlier_iter)
            except StopIteration:
                outlier_iter = iter(outlier_loader)
                outlier_images, _ = next(outlier_iter)
            outlier_out = student(outlier_images.to(device, non_blocking=True))
            if alpha_outlier_uniform > 0.0:
                loss_outlier_uniform = outlier_exposure_uniform_loss(outlier_out["logits"])
            if alpha_outlier_energy > 0.0:
                loss_outlier_energy = energy_margin_loss(
                    s_out["logits"], outlier_out["logits"],
                    margin=energy_margin, temperature=energy_temperature,
                )
            if alpha_outlier_uncertainty > 0.0:
                loss_outlier_uncertainty = F.binary_cross_entropy(
                    outlier_out["uncertainty"].clamp(1e-6, 1.0 - 1e-6),
                    torch.ones_like(outlier_out["uncertainty"]),
                )
            if alpha_outlier_feature_margin > 0.0:
                loss_outlier_feature_margin = unknown_feature_margin_loss(
                    outlier_out["features"],
                    student.classifier.weight,
                    similarity_margin=outlier_feature_margin,
                )
        if novel_head is not None and alpha_joint_discovery > 0.0:
            main_novel_logits = novel_head(s_out["features"])
            if alpha_joint_proto_repulsion > 0.0:
                joint_proto_repulsion = prototype_repulsion_loss(
                    novel_head.prototypes,
                    similarity_margin=joint_proto_repulsion_margin,
                )
            if joint_space == "unified" and alpha_joint_known_ce > 0.0:
                unified_logits = combine_known_novel_logits(
                    s_out["logits"],
                    main_novel_logits,
                    known_temperature=joint_known_temperature,
                )
                loss_joint_known_ce = classification_loss(unified_logits, labels)
        discovery_selected_ratio = 0.0
        discovery_raw_selected_ratio = 0.0
        discovery_neighbor_agreement = 0.0
        discovery_weight_mean = 0.0
        feature_candidate_ratio = 0.0
        feature_candidate_weight_mean = 0.0
        if discovery_iter is not None and (
            alpha_discovery > 0.0
            or alpha_discovery_unknown > 0.0
            or alpha_discovery_energy > 0.0
            or alpha_discovery_uniform > 0.0
            or alpha_discovery_feature_margin > 0.0
            or alpha_discovery_objectosphere > 0.0
            or alpha_discovery_feature_separation > 0.0
            or alpha_discovery_boundary > 0.0
            or alpha_discovery_knn_boundary > 0.0
            or alpha_discovery_uncertainty_separation > 0.0
            or alpha_discovery_uncertainty_pu > 0.0
            or alpha_discovery_selective_unknown > 0.0
            or alpha_discovery_selective_energy > 0.0
            or alpha_reciprocal > 0.0
            or alpha_joint_discovery > 0.0
        ):
            try:
                discovery_images = next(discovery_iter)
            except StopIteration:
                discovery_iter = iter(discovery_loader)
                discovery_images = next(discovery_iter)
            first_view, second_view = discovery_images
            first_view = first_view.to(device)
            second_view = second_view.to(device)
            first_out = student(first_view)
            second_out = student(second_view)
            if reciprocal_points is not None and alpha_reciprocal > 0.0:
                loss_reciprocal = 0.5 * (
                    reciprocal_point_loss(
                        s_out["features"], first_out["features"],
                        student.classifier.weight, reciprocal_points,
                        margin=reciprocal_margin,
                    )
                    + reciprocal_point_loss(
                        s_out["features"], second_out["features"],
                        student.classifier.weight, reciprocal_points,
                        margin=reciprocal_margin,
                    )
                )
            joint_mask = None
            if joint_candidate_gating and alpha_joint_discovery > 0.0:
                if discovery_selection_model is None:
                    first_selection_out = first_out
                    second_selection_out = second_out
                else:
                    with torch.no_grad():
                        first_selection_out = discovery_selection_model(first_view)
                        second_selection_out = discovery_selection_model(second_view)
                first_selection_novel_logits = novel_head(first_selection_out["features"])
                second_selection_novel_logits = novel_head(second_selection_out["features"])
                if joint_candidate_soft_weighting:
                    first_weight = compute_joint_candidate_weights(
                        first_selection_out["logits"],
                        first_selection_out["uncertainty"],
                        novel_logits=first_selection_novel_logits,
                        ratio=joint_candidate_ratio,
                        mode=joint_candidate_mode,
                        floor=joint_candidate_weight_floor,
                        known_temperature=joint_known_temperature,
                    )
                    second_weight = compute_joint_candidate_weights(
                        second_selection_out["logits"],
                        second_selection_out["uncertainty"],
                        novel_logits=second_selection_novel_logits,
                        ratio=joint_candidate_ratio,
                        mode=joint_candidate_mode,
                        floor=joint_candidate_weight_floor,
                        known_temperature=joint_known_temperature,
                    )
                    joint_sample_weights = 0.5 * (first_weight + second_weight)
                    joint_candidate_ratio_value = (
                        joint_sample_weights >= 0.5
                    ).float().mean().item()
                else:
                    # Rank the paired views jointly.  Selecting top-k in each
                    # view and intersecting the masks can erase the whole
                    # training signal on small batches, even when both views
                    # agree on the same risk ordering.
                    paired_known_logits = 0.5 * (
                        first_selection_out["logits"] + second_selection_out["logits"]
                    )
                    paired_novel_logits = 0.5 * (
                        first_selection_novel_logits + second_selection_novel_logits
                    )
                    paired_uncertainty = 0.5 * (
                        first_selection_out["uncertainty"]
                        + second_selection_out["uncertainty"]
                    )
                    paired_features = 0.5 * (
                        first_selection_out["features"] + second_selection_out["features"]
                    )
                    joint_mask = select_joint_candidates(
                        paired_known_logits,
                        paired_novel_logits,
                        paired_uncertainty,
                        features=paired_features,
                        known_prototypes=student.classifier.weight,
                        ratio=joint_candidate_ratio,
                        mode=joint_candidate_mode,
                        known_temperature=joint_known_temperature,
                    )
                    if joint_candidate_neighbor_filter:
                        joint_mask, _ = filter_discovery_candidates_by_neighbors(
                            joint_mask,
                            0.5 * (
                                first_selection_out["proj"] + second_selection_out["proj"]
                            ),
                            k=joint_candidate_neighbor_k,
                            min_votes=joint_candidate_min_votes,
                        )
                    joint_candidate_ratio_value = joint_mask.float().mean().item()
            if (
                joint_mixed_known_consistency
                and alpha_joint_known_consistency > 0.0
                and joint_mask is not None
            ):
                # The complement of the high-risk gate is not automatically
                # known.  Retain only EMA/student predictions that agree
                # across views and exceed a confidence floor.
                if joint_known_target == "teacher":
                    with torch.no_grad():
                        first_target = teacher(first_view)
                        second_target = teacher(second_view)
                elif joint_known_target == "ema":
                    first_target = first_selection_out if first_selection_out is not None else first_out
                    second_target = second_selection_out if second_selection_out is not None else second_out
                else:
                    raise ValueError(
                        "joint_known_target must be either 'ema' or 'teacher'"
                    )
                first_probs = first_target["logits"].detach().softmax(dim=-1)
                second_probs = second_target["logits"].detach().softmax(dim=-1)
                first_labels = first_probs.argmax(dim=-1)
                second_labels = second_probs.argmax(dim=-1)
                confidence = 0.5 * (
                    first_probs.max(dim=-1).values + second_probs.max(dim=-1).values
                )
                known_anchor_mask = (
                    (~joint_mask)
                    & (first_labels == second_labels)
                    & (confidence >= float(joint_known_confidence_threshold))
                )
                loss_joint_known_consistency = known_pseudo_label_consistency_loss(
                    first_out["logits"],
                    second_out["logits"],
                    ((first_labels + second_labels) // 2),
                    mask=known_anchor_mask,
                )
            if novel_head is not None and alpha_joint_discovery > 0.0:
                first_novel_logits = novel_head(first_out["features"])
                second_novel_logits = novel_head(second_out["features"])
                first_weight_out = first_out
                second_weight_out = second_out
                first_weight_novel_logits = first_novel_logits
                second_weight_novel_logits = second_novel_logits
                if joint_novel_ema_weights and discovery_selection_model is not None:
                    with torch.no_grad():
                        first_weight_out = discovery_selection_model(first_view)
                        second_weight_out = discovery_selection_model(second_view)
                        first_weight_novel_logits = novel_head(first_weight_out["features"])
                        second_weight_novel_logits = novel_head(second_weight_out["features"])
                elif joint_novel_ema_weights and discovery_selection_model is None:
                    raise ValueError(
                        "--joint-novel-ema-weights requires "
                        "--discovery-selection-model ema"
                    )
                first_residual = None
                second_residual = None
                first_novel_mass = None
                second_novel_mass = None
                if joint_mixed_residual:
                    first_residual = known_residual_weights(
                        first_out["logits"],
                        temperature=joint_residual_temperature,
                        floor=joint_residual_floor,
                    )
                    second_residual = known_residual_weights(
                        second_out["logits"],
                        temperature=joint_residual_temperature,
                        floor=joint_residual_floor,
                    )
                if joint_novel_mass:
                    first_novel_mass = novel_mass_weights(
                        first_weight_out["logits"],
                        first_weight_novel_logits,
                        known_temperature=joint_known_temperature,
                        floor=joint_residual_floor,
                    )
                    second_novel_mass = novel_mass_weights(
                        second_weight_out["logits"],
                        second_weight_novel_logits,
                        known_temperature=joint_known_temperature,
                        floor=joint_residual_floor,
                    )
                if joint_mask is not None:
                    first_features = first_out["features"][joint_mask]
                    second_features = second_out["features"][joint_mask]
                    first_novel_logits = first_novel_logits[joint_mask]
                    second_novel_logits = second_novel_logits[joint_mask]
                    first_known_logits = first_out["logits"][joint_mask]
                    second_known_logits = second_out["logits"][joint_mask]
                    if joint_sample_weights is not None:
                        joint_sample_weights = joint_sample_weights[joint_mask]
                    if joint_mixed_residual:
                        residual_weights = 0.5 * (
                            first_residual[joint_mask] + second_residual[joint_mask]
                        )
                    if joint_novel_mass:
                        novel_mass_weights_value = 0.5 * (
                            first_novel_mass[joint_mask] + second_novel_mass[joint_mask]
                        )
                        novel_mass_agreement = 1.0 - (
                            first_novel_mass[joint_mask] - second_novel_mass[joint_mask]
                        ).abs()
                        mass_weights = bounded_novel_mass_weights(
                            novel_mass_weights_value,
                            novel_mass_agreement,
                            floor=joint_novel_weight_floor,
                        )
                else:
                    first_features = first_out["features"]
                    second_features = second_out["features"]
                    first_known_logits = first_out["logits"]
                    second_known_logits = second_out["logits"]
                    if joint_mixed_residual:
                        residual_weights = 0.5 * (first_residual + second_residual)
                    if joint_novel_mass:
                        novel_mass_weights_value = 0.5 * (
                            first_novel_mass + second_novel_mass
                        )
                        novel_mass_agreement = 1.0 - (
                            first_novel_mass - second_novel_mass
                        ).abs()
                        mass_weights = bounded_novel_mass_weights(
                            novel_mass_weights_value,
                            novel_mass_agreement,
                            floor=joint_novel_weight_floor,
                        )
                if joint_mixed_residual:
                    joint_residual_weight_value = (
                        residual_weights.mean().item() if residual_weights.numel() else 0.0
                    )
                    if joint_sample_weights is None:
                        joint_sample_weights = residual_weights
                    else:
                        joint_sample_weights = joint_sample_weights * residual_weights
                if joint_novel_mass:
                    joint_novel_mass_weight_value = (
                        mass_weights.mean().item() if mass_weights.numel() else 0.0
                    )
                    if joint_novel_neighbor_support:
                        if joint_mask is not None:
                            support_first = first_weight_out["features"][joint_mask]
                            support_second = second_weight_out["features"][joint_mask]
                        else:
                            support_first = first_weight_out["features"]
                            support_second = second_weight_out["features"]
                        support_features = 0.5 * (support_first + support_second)
                        mass_weights = neighbor_novel_support_weights(
                            support_features,
                            mass_weights,
                            k=joint_novel_neighbor_k,
                        ).clamp_min(
                            max(float(joint_residual_floor), float(joint_novel_weight_floor))
                        )
                        joint_novel_neighbor_support_value = (
                            mass_weights.mean().item() if mass_weights.numel() else 0.0
                        )
                    joint_novel_weight_min_meter.update(
                        mass_weights.min().item() if mass_weights.numel() else 0.0,
                        images.size(0),
                    )
                    if joint_sample_weights is None:
                        joint_sample_weights = mass_weights
                    else:
                        joint_sample_weights = joint_sample_weights * mass_weights
                    if alpha_joint_novel_margin > 0.0:
                        loss_joint_novel_margin = 0.5 * (
                            unknown_feature_margin_loss(
                                first_features,
                                student.classifier.weight,
                                similarity_margin=joint_novel_margin,
                                sample_weight=joint_sample_weights,
                            )
                            + unknown_feature_margin_loss(
                                second_features,
                                student.classifier.weight,
                                similarity_margin=joint_novel_margin,
                                sample_weight=joint_sample_weights,
                            )
                        )
                        joint_novel_margin_value = loss_joint_novel_margin.item()
                confidence_threshold = (
                    0.0
                    if joint_mask is not None or joint_sample_weights is not None
                    else joint_confidence_threshold
                )
                joint_losses = joint_discovery_loss(
                    first_features,
                    second_features,
                    first_novel_logits,
                    second_novel_logits,
                    confidence_threshold=confidence_threshold,
                    assignment_temperature=joint_assignment_temperature,
                    alpha_consistency=alpha_joint_consistency,
                    alpha_balance=alpha_joint_balance,
                    alpha_information=alpha_joint_information,
                    alpha_neighbor=alpha_joint_neighbor,
                    alpha_pseudo=alpha_joint_pseudo,
                    neighbor_k=joint_neighbor_k,
                    # In residual mode, known logits are used only to produce
                    # sample weights.  Keeping them in the Sinkhorn space
                    # would again force known and novel classes to compete in
                    # one balanced assignment problem.
                    first_known_logits=(
                        first_known_logits
                        if (
                            joint_space == "unified"
                            and not joint_mixed_residual
                            and not joint_novel_mass
                        )
                        else None
                    ),
                    second_known_logits=(
                        second_known_logits
                        if (
                            joint_space == "unified"
                            and not joint_mixed_residual
                            and not joint_novel_mass
                        )
                        else None
                    ),
                    known_temperature=joint_known_temperature,
                    sample_weights=joint_sample_weights,
                    weighted_assignments=joint_weighted_sinkhorn,
                    memory_features=(
                        joint_memory_features
                        if joint_memory_features is not None
                        and joint_memory_features.size(0) >= max(int(joint_memory_warmup_size), 1)
                        else None
                    ),
                    memory_logits=(
                        joint_memory_logits
                        if joint_memory_logits is not None
                        and joint_memory_logits.size(0) >= max(int(joint_memory_warmup_size), 1)
                        else None
                    ),
                    alpha_memory_neighbor=alpha_joint_memory_neighbor,
                    memory_neighbor_k=joint_memory_k,
                    memory_temperature=joint_memory_temperature,
                )
                loss_joint_discovery = joint_losses["total"]
                joint_consistency = joint_losses["consistency"]
                joint_balance = joint_losses["balance"]
                joint_information = joint_losses["information"]
                joint_neighbor = joint_losses["neighbor"]
                joint_memory_neighbor = joint_losses["memory_neighbor"]
                joint_pseudo = joint_losses["pseudo"]
            # Keep unknown gating independent from the novel-class logits.  In
            # a mixed discovery pool, only the paired high-risk candidates are
            # used as pseudo-unknowns; treating the whole pool as unknown would
            # contaminate the gate with known samples.
            if alpha_joint_gate > 0.0 and joint_mask is not None and joint_mask.any():
                first_unknown_unc = first_out["uncertainty"][joint_mask]
                second_unknown_unc = second_out["uncertainty"][joint_mask]
                joint_gate = 0.5 * (
                    uncertainty_ranking_loss(
                        s_out["uncertainty"],
                        first_unknown_unc,
                        margin=joint_gate_margin,
                    )
                    + uncertainty_ranking_loss(
                        s_out["uncertainty"],
                        second_unknown_unc,
                        margin=joint_gate_margin,
                    )
                )
            feature_candidate_masks = None
            feature_candidate_weights = None
            if (
                discovery_feature_candidate_gating
                and discovery_pool_mode == "mixed"
                and (
                    alpha_discovery_feature_margin > 0.0
                    or alpha_discovery_feature_separation > 0.0
                    or alpha_discovery_boundary > 0.0
                    or alpha_discovery_knn_boundary > 0.0
                    or alpha_discovery_objectosphere > 0.0
                )
            ):
                if discovery_selection_model is None:
                    first_feature_selection = first_out
                    second_feature_selection = second_out
                else:
                    with torch.no_grad():
                        first_feature_selection = discovery_selection_model(first_view)
                        second_feature_selection = discovery_selection_model(second_view)
                paired_feature_mask = select_discovery_candidates(
                    0.5 * (
                        first_feature_selection["logits"]
                        + second_feature_selection["logits"]
                    ),
                    0.5 * (
                        first_feature_selection["uncertainty"]
                        + second_feature_selection["uncertainty"]
                    ),
                    ratio=discovery_select_ratio,
                    mode=discovery_select_mode,
                    features=0.5 * (
                        first_feature_selection["features"]
                        + second_feature_selection["features"]
                    ),
                    prototypes=student.classifier.weight,
                )
                if discovery_cross_view_gating:
                    first_view_mask = select_discovery_candidates(
                        first_feature_selection["logits"],
                        first_feature_selection["uncertainty"],
                        ratio=discovery_select_ratio,
                        mode=discovery_select_mode,
                        features=first_feature_selection["features"],
                        prototypes=student.classifier.weight,
                    )
                    second_view_mask = select_discovery_candidates(
                        second_feature_selection["logits"],
                        second_feature_selection["uncertainty"],
                        ratio=discovery_select_ratio,
                        mode=discovery_select_mode,
                        features=second_feature_selection["features"],
                        prototypes=student.classifier.weight,
                    )
                    paired_feature_mask = first_view_mask & second_view_mask
                # Both augmentations represent the same unlabeled example. The
                # optional intersection gate trades recall for cleaner pseudo-unknowns.
                feature_candidate_masks = (paired_feature_mask, paired_feature_mask)
                feature_candidate_ratio = paired_feature_mask.float().mean().item()
                if discovery_knn_soft_weighting and alpha_discovery_knn_boundary > 0.0:
                    first_soft_mask, first_soft_weights, _ = compute_discovery_candidate_weights(
                        first_feature_selection["logits"],
                        first_feature_selection["uncertainty"],
                        features=first_feature_selection["proj"],
                        student_logits=first_out["logits"],
                        student_uncertainty=first_out["uncertainty"],
                        prototypes=student.classifier.weight,
                        ratio=discovery_select_ratio,
                        mode=discovery_select_mode,
                        neighbor_k=discovery_neighbor_k,
                        neighbor_temperature=discovery_neighbor_temperature,
                    )
                    second_soft_mask, second_soft_weights, _ = compute_discovery_candidate_weights(
                        second_feature_selection["logits"],
                        second_feature_selection["uncertainty"],
                        features=second_feature_selection["proj"],
                        student_logits=second_out["logits"],
                        student_uncertainty=second_out["uncertainty"],
                        prototypes=student.classifier.weight,
                        ratio=discovery_select_ratio,
                        mode=discovery_select_mode,
                        neighbor_k=discovery_neighbor_k,
                        neighbor_temperature=discovery_neighbor_temperature,
                    )
                    # Keep the paired hard gate, while assigning zero weight to
                    # a view that does not independently support the candidate.
                    feature_candidate_weights = (
                        (first_soft_weights * first_soft_mask.float())[paired_feature_mask],
                        (second_soft_weights * second_soft_mask.float())[paired_feature_mask],
                    )
                    if feature_candidate_weights[0].numel():
                        feature_candidate_weight_mean = 0.5 * (
                            feature_candidate_weights[0].mean().item()
                            + feature_candidate_weights[1].mean().item()
                        )

            loss_discovery = weighted_discovery_view_loss(
                first_out["proj"],
                second_out["proj"],
                alpha=alpha_discovery,
                mode=discovery_loss_mode,
                temperature=discovery_temperature,
            )
            if alpha_discovery_unknown > 0.0:
                loss_discovery_unknown = 0.5 * (
                    discovery_unknown_loss(first_out["logits"], first_out["uncertainty"])
                    + discovery_unknown_loss(second_out["logits"], second_out["uncertainty"])
                )
            if alpha_discovery_energy > 0.0:
                loss_discovery_energy = 0.5 * (
                    energy_margin_loss(
                        s_out["logits"],
                        first_out["logits"],
                        margin=energy_margin,
                        temperature=energy_temperature,
                    )
                    + energy_margin_loss(
                        s_out["logits"],
                        second_out["logits"],
                        margin=energy_margin,
                        temperature=energy_temperature,
                    )
                )
            if alpha_discovery_uniform > 0.0:
                loss_discovery_uniform = 0.5 * (
                    outlier_exposure_uniform_loss(first_out["logits"])
                    + outlier_exposure_uniform_loss(second_out["logits"])
                )
            if alpha_discovery_feature_margin > 0.0:
                first_feature_values = first_out["features"]
                second_feature_values = second_out["features"]
                if feature_candidate_masks is not None:
                    first_feature_values = first_feature_values[feature_candidate_masks[0]]
                    second_feature_values = second_feature_values[feature_candidate_masks[1]]
                loss_discovery_feature_margin = 0.5 * (
                    unknown_feature_margin_loss(
                        first_feature_values,
                        student.classifier.weight,
                        similarity_margin=discovery_feature_margin,
                    )
                    + unknown_feature_margin_loss(
                        second_feature_values,
                        student.classifier.weight,
                        similarity_margin=discovery_feature_margin,
                    )
                )
            if alpha_discovery_objectosphere > 0.0:
                if discovery_pool_mode == "mixed" and feature_candidate_masks is None:
                    raise ValueError(
                        "mixed-pool Objectosphere requires "
                        "--discovery-feature-candidate-gating"
                    )
                first_objectosphere_unknown = first_out["features"]
                second_objectosphere_unknown = second_out["features"]
                if feature_candidate_masks is not None:
                    first_objectosphere_unknown = first_objectosphere_unknown[
                        feature_candidate_masks[0]
                    ]
                    second_objectosphere_unknown = second_objectosphere_unknown[
                        feature_candidate_masks[1]
                    ]
                loss_discovery_objectosphere = 0.5 * (
                    objectosphere_loss(
                        s_out["features"],
                        first_objectosphere_unknown,
                        known_radius=objectosphere_known_radius,
                        unknown_weight=objectosphere_unknown_weight,
                    )
                    + objectosphere_loss(
                        s_out["features"],
                        second_objectosphere_unknown,
                        known_radius=objectosphere_known_radius,
                        unknown_weight=objectosphere_unknown_weight,
                    )
                )
            if alpha_discovery_feature_separation > 0.0:
                first_feature_values = first_out["features"]
                second_feature_values = second_out["features"]
                if feature_candidate_masks is not None:
                    first_feature_values = first_feature_values[feature_candidate_masks[0]]
                    second_feature_values = second_feature_values[feature_candidate_masks[1]]
                loss_discovery_feature_separation = 0.5 * (
                    unknown_feature_separation_loss(
                        s_out["features"],
                        first_feature_values,
                        similarity_margin=discovery_feature_separation_margin,
                        temperature=discovery_feature_separation_temperature,
                    )
                    + unknown_feature_separation_loss(
                        s_out["features"],
                        second_feature_values,
                        similarity_margin=discovery_feature_separation_margin,
                        temperature=discovery_feature_separation_temperature,
                    )
                )
            if alpha_discovery_boundary > 0.0:
                first_feature_values = first_out["features"]
                second_feature_values = second_out["features"]
                if feature_candidate_masks is not None:
                    first_feature_values = first_feature_values[feature_candidate_masks[0]]
                    second_feature_values = second_feature_values[feature_candidate_masks[1]]
                loss_discovery_boundary = 0.5 * (
                    unknown_feature_boundary_loss(
                        s_out["features"],
                        labels,
                        first_feature_values,
                        student.classifier.weight,
                        similarity_margin=discovery_boundary_margin,
                        prototype_weight=discovery_boundary_prototype_weight,
                        temperature=discovery_boundary_temperature,
                    )
                    + unknown_feature_boundary_loss(
                        s_out["features"],
                        labels,
                        second_feature_values,
                        student.classifier.weight,
                        similarity_margin=discovery_boundary_margin,
                        prototype_weight=discovery_boundary_prototype_weight,
                        temperature=discovery_boundary_temperature,
                    )
                )
            if alpha_discovery_knn_boundary > 0.0:
                if discovery_knn_support is None:
                    raise ValueError(
                        "kNN boundary loss requires an epoch-specific known support bank"
                    )
                first_knn_values = first_out["features"]
                second_knn_values = second_out["features"]
                if feature_candidate_masks is not None:
                    first_knn_values = first_knn_values[feature_candidate_masks[0]]
                    second_knn_values = second_knn_values[feature_candidate_masks[1]]
                first_knn_weights = None
                second_knn_weights = None
                if feature_candidate_weights is not None:
                    first_knn_weights, second_knn_weights = feature_candidate_weights
                loss_discovery_knn_boundary = 0.5 * (
                    knn_support_boundary_loss(
                        first_knn_values,
                        discovery_knn_support["features_by_class"],
                        discovery_knn_support["radii"],
                        k=discovery_knn_k,
                        margin=discovery_knn_margin,
                        sample_weights=first_knn_weights,
                    )
                    + knn_support_boundary_loss(
                        second_knn_values,
                        discovery_knn_support["features_by_class"],
                        discovery_knn_support["radii"],
                        k=discovery_knn_k,
                        margin=discovery_knn_margin,
                        sample_weights=second_knn_weights,
                    )
                )
            if alpha_discovery_uncertainty_separation > 0.0:
                separation_loss = uncertainty_separation_loss if discovery_uncertainty_loss == "bce" else uncertainty_ranking_loss
                loss_discovery_uncertainty_separation = 0.5 * (
                    separation_loss(
                        s_out["uncertainty"],
                        first_out["uncertainty"],
                        **({"margin": discovery_uncertainty_margin} if discovery_uncertainty_loss == "ranking" else {}),
                    )
                    + separation_loss(
                        s_out["uncertainty"],
                        second_out["uncertainty"],
                        **({"margin": discovery_uncertainty_margin} if discovery_uncertainty_loss == "ranking" else {}),
                    )
                )
            if alpha_discovery_uncertainty_pu > 0.0:
                if not 0.0 < float(discovery_uncertainty_known_prior) < 1.0:
                    raise ValueError(
                        "discovery_uncertainty_known_prior must be in (0, 1)"
                    )
                unlabeled_uncertainty = 0.5 * (
                    first_out["uncertainty"] + second_out["uncertainty"]
                )
                loss_discovery_uncertainty_pu = nnpu_known_uncertainty_loss(
                    s_out["uncertainty"],
                    unlabeled_uncertainty,
                    known_prior=discovery_uncertainty_known_prior,
                )
            if alpha_discovery_selective_unknown > 0.0 or alpha_discovery_selective_energy > 0.0:
                if discovery_selection_model is None:
                    first_selection_out = first_out
                    second_selection_out = second_out
                else:
                    with torch.no_grad():
                        first_selection_out = discovery_selection_model(first_view)
                        second_selection_out = discovery_selection_model(second_view)
                first_mask = select_discovery_candidates(
                    first_selection_out["logits"],
                    first_selection_out["uncertainty"],
                    ratio=discovery_select_ratio,
                    mode=discovery_select_mode,
                    features=first_selection_out["features"],
                    prototypes=student.classifier.weight,
                )
                second_mask = select_discovery_candidates(
                    second_selection_out["logits"],
                    second_selection_out["uncertainty"],
                    ratio=discovery_select_ratio,
                    mode=discovery_select_mode,
                    features=second_selection_out["features"],
                    prototypes=student.classifier.weight,
                )
                discovery_raw_selected_ratio = 0.5 * (
                    first_mask.float().mean().item() + second_mask.float().mean().item()
                )
                if discovery_neighbor_filter:
                    first_mask, first_agreement = filter_discovery_candidates_by_neighbors(
                        first_mask,
                        first_selection_out["proj"],
                        k=discovery_neighbor_k,
                        min_votes=discovery_neighbor_min_votes,
                    )
                    second_mask, second_agreement = filter_discovery_candidates_by_neighbors(
                        second_mask,
                        second_selection_out["proj"],
                        k=discovery_neighbor_k,
                        min_votes=discovery_neighbor_min_votes,
                    )
                    discovery_neighbor_agreement = 0.5 * (first_agreement + second_agreement)
                first_weight = first_mask.float()
                second_weight = second_mask.float()
                if discovery_soft_weighting:
                    first_mask, first_weight, first_agreement = compute_discovery_candidate_weights(
                        first_selection_out["logits"],
                        first_selection_out["uncertainty"],
                        features=first_selection_out["proj"],
                        student_logits=first_out["logits"],
                        student_uncertainty=first_out["uncertainty"],
                        ratio=discovery_select_ratio,
                        mode=discovery_select_mode,
                        neighbor_k=discovery_neighbor_k,
                        neighbor_temperature=discovery_neighbor_temperature,
                    )
                    second_mask, second_weight, second_agreement = compute_discovery_candidate_weights(
                        second_selection_out["logits"],
                        second_selection_out["uncertainty"],
                        features=second_selection_out["proj"],
                        student_logits=second_out["logits"],
                        student_uncertainty=second_out["uncertainty"],
                        ratio=discovery_select_ratio,
                        mode=discovery_select_mode,
                        neighbor_k=discovery_neighbor_k,
                        neighbor_temperature=discovery_neighbor_temperature,
                    )
                    discovery_neighbor_agreement = 0.5 * (first_agreement + second_agreement)
                discovery_selected_ratio = 0.5 * (
                    first_mask.float().mean().item() + second_mask.float().mean().item()
                )
                selected_weights = torch.cat(
                    [first_weight[first_mask], second_weight[second_mask]], dim=0
                )
                discovery_weight_mean = selected_weights.mean().item() if selected_weights.numel() else 0.0
                if alpha_discovery_selective_unknown > 0.0:
                    loss_discovery_selective_unknown = 0.5 * (
                        discovery_unknown_loss(
                            first_out["logits"][first_mask],
                            first_out["uncertainty"][first_mask],
                            first_weight[first_mask] if discovery_soft_weighting else None,
                        )
                        + discovery_unknown_loss(
                            second_out["logits"][second_mask],
                            second_out["uncertainty"][second_mask],
                            second_weight[second_mask] if discovery_soft_weighting else None,
                        )
                    )
                if alpha_discovery_selective_energy > 0.0:
                    loss_discovery_selective_energy = 0.5 * (
                        (
                            weighted_energy_margin_loss(
                                s_out["logits"],
                                first_out["logits"][first_mask],
                                first_weight[first_mask],
                                margin=energy_margin,
                                temperature=energy_temperature,
                            )
                            if discovery_soft_weighting
                            else energy_margin_loss(
                                s_out["logits"],
                                first_out["logits"][first_mask],
                                margin=energy_margin,
                                temperature=energy_temperature,
                            )
                        )
                        + (
                            weighted_energy_margin_loss(
                                s_out["logits"],
                                second_out["logits"][second_mask],
                                second_weight[second_mask],
                                margin=energy_margin,
                                temperature=energy_temperature,
                            )
                            if discovery_soft_weighting
                            else energy_margin_loss(
                                s_out["logits"],
                                second_out["logits"][second_mask],
                                margin=energy_margin,
                                temperature=energy_temperature,
                            )
                        )
                    )
        loss = (
            loss_ce
            + alpha_unc * loss_unc
            + alpha_kd * loss_kd
            + alpha_feat_kd * loss_feat_kd
            + alpha_supcon * loss_supcon
            + alpha_raw_supcon * loss_raw_supcon
            + alpha_proto * loss_proto
            + alpha_proto_repulsion * loss_proto_repulsion
            + alpha_proxy * loss_proxy
            + alpha_pseudo * loss_pseudo
            + alpha_energy * loss_energy
            + alpha_vos * loss_vos
            + alpha_discovery * loss_discovery
            + alpha_discovery_unknown * loss_discovery_unknown
            + alpha_discovery_energy * loss_discovery_energy
            + alpha_discovery_uniform * loss_discovery_uniform
            + alpha_discovery_feature_margin * loss_discovery_feature_margin
            + alpha_discovery_objectosphere * loss_discovery_objectosphere
            + alpha_discovery_feature_separation * loss_discovery_feature_separation
            + alpha_discovery_boundary * loss_discovery_boundary
            + alpha_discovery_knn_boundary * loss_discovery_knn_boundary
            + alpha_discovery_uncertainty_separation * loss_discovery_uncertainty_separation
            + alpha_discovery_uncertainty_pu * loss_discovery_uncertainty_pu
            + alpha_discovery_selective_unknown * loss_discovery_selective_unknown
            + alpha_discovery_selective_energy * loss_discovery_selective_energy
            + alpha_angular * loss_angular
            + alpha_proxy_anchor * loss_proxy_anchor
            + alpha_reciprocal * loss_reciprocal
            + alpha_joint_discovery * loss_joint_discovery
            + alpha_joint_proto_repulsion * joint_proto_repulsion
            + alpha_joint_known_ce * loss_joint_known_ce
            + alpha_joint_known_consistency * loss_joint_known_consistency
            + alpha_joint_gate * joint_gate
            + alpha_joint_novel_margin * loss_joint_novel_margin
            + alpha_outlier_uniform * loss_outlier_uniform
            + alpha_outlier_energy * loss_outlier_energy
            + alpha_outlier_uncertainty * loss_outlier_uncertainty
            + alpha_outlier_feature_margin * loss_outlier_feature_margin
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (
            novel_head is not None
            and alpha_joint_discovery > 0.0
            and alpha_joint_memory_neighbor > 0.0
            and first_novel_logits is not None
        ):
            with torch.no_grad():
                memory_features_batch = 0.5 * (
                    first_features.detach() + second_features.detach()
                )
                memory_known_first = (
                    first_known_logits.detach()
                    if (
                        joint_space == "unified"
                        and not joint_mixed_residual
                        and not joint_novel_mass
                        and first_known_logits is not None
                    )
                    else None
                )
                memory_known_second = (
                    second_known_logits.detach()
                    if (
                        joint_space == "unified"
                        and not joint_mixed_residual
                        and not joint_novel_mass
                        and second_known_logits is not None
                    )
                    else None
                )
                first_memory_logits = combine_known_novel_logits(
                    memory_known_first,
                    first_novel_logits.detach(),
                    known_temperature=joint_known_temperature,
                )
                second_memory_logits = combine_known_novel_logits(
                    memory_known_second,
                    second_novel_logits.detach(),
                    known_temperature=joint_known_temperature,
                )
                memory_logits_batch = 0.5 * (
                    first_memory_logits + second_memory_logits
                )
                if joint_sample_weights is not None:
                    keep = joint_sample_weights.detach() > 0.05
                    memory_features_batch = memory_features_batch[keep]
                    memory_logits_batch = memory_logits_batch[keep]
                if memory_features_batch.numel() > 0:
                    if joint_memory_features is None:
                        joint_memory_features = memory_features_batch
                        joint_memory_logits = memory_logits_batch
                    else:
                        joint_memory_features = torch.cat(
                            [joint_memory_features, memory_features_batch], dim=0
                        )[-max(int(joint_memory_size), 1):]
                        joint_memory_logits = torch.cat(
                            [joint_memory_logits, memory_logits_batch], dim=0
                        )[-max(int(joint_memory_size), 1):]
                    joint_memory_bank["features"] = joint_memory_features
                    joint_memory_bank["logits"] = joint_memory_logits
        if discovery_selection_model is not None and discovery_selection_model_updates_ema:
            update_ema_model(discovery_selection_model, student, decay=discovery_ema_decay)
        ce_meter.update(loss_ce.item(), images.size(0))
        kd_meter.update(loss_kd.item(), images.size(0))
        feat_kd_meter.update(loss_feat_kd.item(), images.size(0))
        unc_meter.update(loss_unc.item(), images.size(0))
        sc_meter.update(loss_supcon.item(), images.size(0))
        raw_sc_meter.update(loss_raw_supcon.item(), images.size(0))
        proto_meter.update(loss_proto.item(), images.size(0))
        proto_repulsion_meter.update(loss_proto_repulsion.item(), images.size(0))
        proxy_meter.update(loss_proxy.item(), images.size(0))
        pseudo_meter.update(loss_pseudo.item(), images.size(0))
        energy_meter.update(loss_energy.item(), images.size(0))
        vos_meter.update(loss_vos.item(), images.size(0))
        discovery_meter.update(loss_discovery.item(), images.size(0))
        discovery_unknown_meter.update(loss_discovery_unknown.item(), images.size(0))
        discovery_energy_meter.update(loss_discovery_energy.item(), images.size(0))
        discovery_uniform_meter.update(loss_discovery_uniform.item(), images.size(0))
        discovery_feature_margin_meter.update(loss_discovery_feature_margin.item(), images.size(0))
        discovery_objectosphere_meter.update(loss_discovery_objectosphere.item(), images.size(0))
        discovery_feature_separation_meter.update(loss_discovery_feature_separation.item(), images.size(0))
        discovery_boundary_meter.update(loss_discovery_boundary.item(), images.size(0))
        discovery_knn_boundary_meter.update(
            loss_discovery_knn_boundary.item(), images.size(0)
        )
        discovery_uncertainty_separation_meter.update(
            loss_discovery_uncertainty_separation.item(), images.size(0)
        )
        discovery_uncertainty_pu_meter.update(
            loss_discovery_uncertainty_pu.item(), images.size(0)
        )
        discovery_selective_unknown_meter.update(loss_discovery_selective_unknown.item(), images.size(0))
        discovery_selective_energy_meter.update(loss_discovery_selective_energy.item(), images.size(0))
        angular_meter.update(loss_angular.item(), images.size(0))
        proxy_anchor_meter.update(loss_proxy_anchor.item(), images.size(0))
        reciprocal_meter.update(loss_reciprocal.item(), images.size(0))
        discovery_selected_meter.update(discovery_selected_ratio, images.size(0))
        discovery_raw_selected_meter.update(discovery_raw_selected_ratio, images.size(0))
        discovery_neighbor_agreement_meter.update(discovery_neighbor_agreement, images.size(0))
        discovery_weight_mean_meter.update(discovery_weight_mean, images.size(0))
        feature_candidate_ratio_meter.update(feature_candidate_ratio, images.size(0))
        feature_candidate_weight_mean_meter.update(
            feature_candidate_weight_mean, images.size(0)
        )
        joint_meter.update(loss_joint_discovery.item(), images.size(0))
        joint_consistency_meter.update(joint_consistency.item(), images.size(0))
        joint_balance_meter.update(joint_balance.item(), images.size(0))
        joint_information_meter.update(joint_information.item(), images.size(0))
        joint_neighbor_meter.update(joint_neighbor.item(), images.size(0))
        joint_memory_neighbor_meter.update(
            joint_memory_neighbor.item(), images.size(0)
        )
        joint_pseudo_meter.update(joint_pseudo.item(), images.size(0))
        joint_gate_meter.update(joint_gate.item(), images.size(0))
        joint_proto_repulsion_meter.update(joint_proto_repulsion.item(), images.size(0))
        joint_known_ce_meter.update(loss_joint_known_ce.item(), images.size(0))
        joint_candidate_meter.update(joint_candidate_ratio_value, images.size(0))
        joint_known_consistency_meter.update(
            loss_joint_known_consistency.item(), images.size(0)
        )
        joint_residual_weight_meter.update(joint_residual_weight_value, images.size(0))
        joint_novel_mass_weight_meter.update(joint_novel_mass_weight_value, images.size(0))
        joint_novel_margin_meter.update(joint_novel_margin_value, images.size(0))
        joint_novel_neighbor_support_meter.update(
            joint_novel_neighbor_support_value, images.size(0)
        )
        outlier_uniform_meter.update(loss_outlier_uniform.item(), images.size(0))
        outlier_energy_meter.update(loss_outlier_energy.item(), images.size(0))
        outlier_uncertainty_meter.update(loss_outlier_uncertainty.item(), images.size(0))
        outlier_feature_margin_meter.update(loss_outlier_feature_margin.item(), images.size(0))
    return {
        "ce": ce_meter.avg,
        "kd": kd_meter.avg,
        "feat_kd": feat_kd_meter.avg,
        "unc": unc_meter.avg,
        "supcon": sc_meter.avg,
        "raw_supcon": raw_sc_meter.avg,
        "proto": proto_meter.avg,
        "proto_repulsion": proto_repulsion_meter.avg,
        "proxy": proxy_meter.avg,
        "pseudo": pseudo_meter.avg,
        "energy": energy_meter.avg,
        "vos": vos_meter.avg,
        "discovery": discovery_meter.avg,
        "discovery_unknown": discovery_unknown_meter.avg,
        "discovery_energy": discovery_energy_meter.avg,
        "discovery_uniform": discovery_uniform_meter.avg,
        "discovery_feature_margin": discovery_feature_margin_meter.avg,
        "discovery_objectosphere": discovery_objectosphere_meter.avg,
        "discovery_feature_separation": discovery_feature_separation_meter.avg,
        "discovery_boundary": discovery_boundary_meter.avg,
        "discovery_knn_boundary": discovery_knn_boundary_meter.avg,
        "discovery_uncertainty_separation": discovery_uncertainty_separation_meter.avg,
        "discovery_uncertainty_pu": discovery_uncertainty_pu_meter.avg,
        "discovery_selective_unknown": discovery_selective_unknown_meter.avg,
        "discovery_selective_energy": discovery_selective_energy_meter.avg,
        "angular": angular_meter.avg,
        "angular_active_ratio": angular_active_meter.avg,
        "proxy_anchor": proxy_anchor_meter.avg,
        "reciprocal": reciprocal_meter.avg,
        "discovery_selected_ratio": discovery_selected_meter.avg,
        "discovery_raw_selected_ratio": discovery_raw_selected_meter.avg,
        "discovery_neighbor_agreement": discovery_neighbor_agreement_meter.avg,
        "discovery_weight_mean": discovery_weight_mean_meter.avg,
        "feature_candidate_ratio": feature_candidate_ratio_meter.avg,
        "feature_candidate_weight_mean": feature_candidate_weight_mean_meter.avg,
        "joint_discovery": joint_meter.avg,
        "joint_consistency": joint_consistency_meter.avg,
        "joint_balance": joint_balance_meter.avg,
        "joint_information": joint_information_meter.avg,
        "joint_neighbor": joint_neighbor_meter.avg,
        "joint_memory_neighbor": joint_memory_neighbor_meter.avg,
        "joint_pseudo": joint_pseudo_meter.avg,
        "joint_gate": joint_gate_meter.avg,
        "joint_proto_repulsion": joint_proto_repulsion_meter.avg,
        "joint_known_ce": joint_known_ce_meter.avg,
        "joint_candidate_ratio": joint_candidate_meter.avg,
        "joint_known_consistency": joint_known_consistency_meter.avg,
        "joint_residual_weight": joint_residual_weight_meter.avg,
        "joint_novel_mass_weight": joint_novel_mass_weight_meter.avg,
        "joint_novel_weight_min": joint_novel_weight_min_meter.avg,
        "joint_novel_margin": joint_novel_margin_meter.avg,
        "joint_novel_neighbor_support": joint_novel_neighbor_support_meter.avg,
        "outlier_uniform": outlier_uniform_meter.avg,
        "outlier_energy": outlier_energy_meter.avg,
        "outlier_uncertainty": outlier_uncertainty_meter.avg,
        "outlier_feature_margin": outlier_feature_margin_meter.avg,
    }


@torch.no_grad()
def evaluate_classification(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="eval-cls", leave=False):
        images, labels, raw_labels, is_known, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        pred = out["logits"].argmax(dim=-1)
        mask = labels >= 0
        if mask.any():
            correct += (pred[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
    return {"known_acc": correct / max(total, 1)}


@torch.no_grad()
def evaluate_representation_geometry(model, loader, device, num_classes: int):
    """Measure known-only feature geometry without using unknown labels.

    The values are diagnostics for checkpoint selection, not an open-set
    benchmark.  Class centers are computed from the validation features, then
    each sample is compared with its own center and each class center with its
    nearest *other* class center.
    """
    was_training = model.training
    model.eval()
    feature_chunks = []
    label_chunks = []
    try:
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            valid = (labels >= 0) & (labels < int(num_classes))
            if valid.any():
                feature_chunks.append(F.normalize(outputs["features"][valid].float(), dim=-1).cpu())
                label_chunks.append(labels[valid].cpu())
    finally:
        model.train(was_training)

    if not feature_chunks:
        return {
            "geometry_samples": 0,
            "feature_within_mean": float("nan"),
            "feature_within_q95": float("nan"),
            "nearest_center_distance": float("nan"),
            "center_margin": float("nan"),
        }

    features = torch.cat(feature_chunks, dim=0).numpy()
    labels = torch.cat(label_chunks, dim=0).numpy()
    centers = []
    within_distances = []
    for cls in range(int(num_classes)):
        class_features = features[labels == cls]
        if len(class_features) == 0:
            centers.append(np.zeros(features.shape[1], dtype=np.float32))
            continue
        center = class_features.mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-6)
        centers.append(center)
        within_distances.extend((1.0 - class_features @ center).tolist())

    centers = np.asarray(centers, dtype=np.float32)
    center_similarities = centers @ centers.T
    np.fill_diagonal(center_similarities, -np.inf)
    available_classes = np.flatnonzero(np.isfinite(center_similarities).any(axis=1))
    if len(available_classes) < 2:
        nearest_center_distance = float("nan")
    else:
        nearest_center_distance = float(
            np.mean(1.0 - np.max(center_similarities[available_classes], axis=1))
        )
    within_mean = float(np.mean(within_distances)) if within_distances else float("nan")
    within_q95 = float(np.quantile(within_distances, 0.95)) if within_distances else float("nan")
    return {
        "geometry_samples": int(len(features)),
        "feature_within_mean": within_mean,
        "feature_within_q95": within_q95,
        "nearest_center_distance": nearest_center_distance,
        "center_margin": float(nearest_center_distance - within_mean)
        if np.isfinite(nearest_center_distance) and np.isfinite(within_mean)
        else float("nan"),
    }


@torch.no_grad()
def collect_prototypes(model, loader, device, num_classes: int):
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, device=device)
    for batch in tqdm(loader, desc="collect-proto", leave=False):
        images, labels, raw_labels, is_known, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        feat = out["features"]
        if sums is None:
            sums = torch.zeros(num_classes, feat.size(1), device=device)
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                sums[c] += feat[mask].sum(dim=0)
                counts[c] += mask.sum()
    counts = counts.clamp_min(1.0).unsqueeze(1)
    return sums / counts


@torch.no_grad()
def collect_classwise_knn_support(
    model,
    loader,
    device,
    num_classes: int,
    k: int = 5,
    quantile: float = 0.95,
):
    """Build an epoch-specific class-wise known-feature support bank.

    Per-class radii are quantiles of leave-one-out mean-k-neighbor cosine
    distances computed only from labeled known training examples.
    """
    if int(k) <= 0:
        raise ValueError("k must be positive")
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("quantile must be in [0.5, 1.0)")
    was_training = model.training
    model.eval()
    feature_chunks = [[] for _ in range(int(num_classes))]
    try:
        for batch in tqdm(loader, desc="known-knn-bank", leave=False):
            images, labels = batch[0].to(device), batch[1].to(device)
            features = F.normalize(model(images)["features"].float(), dim=-1)
            valid = (labels >= 0) & (labels < int(num_classes))
            for cls in torch.unique(labels[valid]).tolist():
                mask = labels == int(cls)
                feature_chunks[int(cls)].append(features[mask].detach())
    finally:
        model.train(was_training)

    banks = []
    radii = torch.zeros(int(num_classes), dtype=torch.float32, device=device)
    counts = torch.zeros(int(num_classes), dtype=torch.long)
    for cls, chunks in enumerate(feature_chunks):
        if not chunks:
            banks.append(torch.empty(0, 0, dtype=torch.float32, device=device))
            continue
        bank = F.normalize(torch.cat(chunks, dim=0), dim=-1)
        banks.append(bank.detach())
        counts[cls] = bank.size(0)
        if bank.size(0) <= 1:
            radii[cls] = 0.0
            continue
        similarities = bank @ bank.T
        similarities.fill_diagonal_(float("-inf"))
        neighbor_count = min(int(k), bank.size(0) - 1)
        distances = 1.0 - similarities.topk(neighbor_count, dim=-1).values.mean(dim=-1)
        radii[cls] = torch.quantile(distances.clamp_min(0.0), float(quantile))
    if not counts.any():
        raise ValueError("Known training loader produced no class features")
    return {"features_by_class": banks, "radii": radii, "counts": counts}


def _odin_batch_score(
    model,
    images: torch.Tensor,
    epsilon: float = 0.0,
    temperature: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ODIN-style confidence score after temperature scaling and input perturbation."""
    temperature = max(float(temperature), 1e-6)
    images = images.detach().clone().requires_grad_(True)
    logits = model(images, stochastic=False)["logits"] / temperature
    pseudo_labels = logits.argmax(dim=-1)
    loss = F.cross_entropy(logits, pseudo_labels)
    grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
    perturbed = images - float(epsilon) * grad.sign()
    with torch.no_grad():
        odin_logits = model(perturbed, stochastic=False)["logits"] / temperature
        odin_probs = odin_logits.softmax(dim=-1)
        odin_msp = 1.0 - odin_probs.max(dim=-1).values
    return odin_msp.detach(), odin_logits.detach()


def extract_outputs(
    model,
    loader,
    device,
    mc_samples: int = 8,
    odin_epsilon: float = 0.0,
    odin_temperature: float = 1000.0,
    react_clip_value: float | None = None,
):
    model.eval()
    all_logits = []
    all_probs = []
    all_entropy = []
    all_expected_entropy = []
    all_epistemic = []
    all_aleatoric = []
    all_head_uncertainty = []
    all_feature_norm = []
    all_features = []
    all_projections = []
    all_odin_msp = []
    all_odin_logits = []
    all_react_energy = []
    all_labels = []
    all_raw = []
    all_known = []
    for batch in tqdm(loader, desc="extract", leave=False):
        images, labels, raw_labels, is_known, _ = batch
        images = images.to(device)
        with torch.no_grad():
            mc = model.mc_predict(images, mc_samples=mc_samples)
            out = model(images)
        all_logits.append(mc["mean_logits"].cpu())
        all_probs.append(mc["mean_probs"].cpu())
        all_entropy.append(mc["predictive_entropy"].cpu())
        all_expected_entropy.append(mc["expected_entropy"].cpu())
        all_epistemic.append(mc["epistemic"].cpu())
        all_aleatoric.append(mc["aleatoric"].cpu())
        all_head_uncertainty.append(mc["head_uncertainty"].cpu())
        all_features.append(out["features"].cpu())
        all_feature_norm.append(out["features"].norm(dim=-1).cpu())
        all_projections.append(out["proj"].cpu())
        if react_clip_value is not None:
            clipped_features = out["features"].clamp(max=float(react_clip_value))
            react_logits = F.linear(
                clipped_features,
                model.classifier.weight,
                model.classifier.bias,
            )
            all_react_energy.append(-torch.logsumexp(react_logits, dim=-1).detach().cpu())
        if odin_epsilon > 0.0:
            odin_msp, odin_logits = _odin_batch_score(
                model,
                images,
                epsilon=odin_epsilon,
                temperature=odin_temperature,
            )
            all_odin_msp.append(odin_msp.cpu())
            all_odin_logits.append(odin_logits.cpu())
        all_labels.append(labels)
        all_raw.append(raw_labels)
        all_known.append(is_known)
    outputs = {
        "logits": torch.cat(all_logits).numpy(),
        "probs": torch.cat(all_probs).numpy(),
        "entropy": torch.cat(all_entropy).numpy(),
        "expected_entropy": torch.cat(all_expected_entropy).numpy(),
        "epistemic": torch.cat(all_epistemic).numpy(),
        "aleatoric": torch.cat(all_aleatoric).numpy(),
        "head_uncertainty": torch.cat(all_head_uncertainty).numpy(),
        "feature_norm": torch.cat(all_feature_norm).numpy(),
        "features": torch.cat(all_features).numpy(),
        "projections": torch.cat(all_projections).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "raw_labels": torch.cat(all_raw).numpy(),
        "is_known": torch.cat(all_known).numpy(),
    }
    if all_odin_msp:
        outputs["odin_msp"] = torch.cat(all_odin_msp).numpy()
        outputs["odin_logits"] = torch.cat(all_odin_logits).numpy()
    if all_react_energy:
        outputs["react_energy"] = torch.cat(all_react_energy).numpy()
    return outputs


def build_rejector_features(
    outputs: Dict[str, np.ndarray],
    feature_mode: str = "embedding",
) -> np.ndarray:
    """Construct the representation used by the optional binary rejector."""
    features = np.asarray(outputs["features"], dtype=np.float32)
    features = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-6, None)
    if feature_mode == "embedding":
        return features
    if feature_mode not in {
        "augmented",
        "support_augmented",
        "uncertainty_augmented",
        "support_uncertainty_augmented",
    }:
        raise ValueError(f"Unsupported rejector feature mode: {feature_mode}")
    logits = np.asarray(outputs["logits"], dtype=np.float32)
    probs = np.asarray(outputs["probs"], dtype=np.float32)
    entropy = np.asarray(outputs["entropy"], dtype=np.float32).reshape(-1, 1)
    max_prob = probs.max(axis=1, keepdims=True)
    sorted_probs = np.sort(probs, axis=1)
    margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
    uncertainty = np.asarray(outputs["head_uncertainty"], dtype=np.float32).reshape(-1, 1)
    summary = np.concatenate([logits, entropy, max_prob, margin, uncertainty], axis=1)
    if feature_mode in {"uncertainty_augmented", "support_uncertainty_augmented"}:
        # MC-dropout signals are intentionally kept separate from the learned
        # uncertainty head.  The former estimates epistemic uncertainty while
        # the latter is an auxiliary aleatoric/confidence signal.  Using both
        # lets the rejector test the uncertainty decomposition directly.
        epistemic = np.asarray(outputs.get("epistemic", np.zeros(len(features))), dtype=np.float32).reshape(-1, 1)
        expected_entropy = np.asarray(
            outputs.get("expected_entropy", entropy[:, 0]), dtype=np.float32
        ).reshape(-1, 1)
        aleatoric = np.asarray(
            outputs.get("aleatoric", uncertainty[:, 0]), dtype=np.float32
        ).reshape(-1, 1)
        summary = np.concatenate(
            [summary, epistemic, expected_entropy, aleatoric], axis=1
        )
    if feature_mode in {"support_augmented", "support_uncertainty_augmented"}:
        if "known_support_score" not in outputs:
            raise ValueError("support_augmented requires known_support_score")
        support_score = np.asarray(outputs["known_support_score"], dtype=np.float32).reshape(-1, 1)
        summary = np.concatenate([summary, support_score], axis=1)
    return np.concatenate([features, summary], axis=1)


def fit_feature_rejector(
    known_outputs: Dict[str, np.ndarray],
    unknown_outputs: Dict[str, np.ndarray],
    max_samples: int = 5000,
    seed: int = 42,
    feature_mode: str = "embedding",
    model_type: str = "logistic",
):
    """Fit a frozen-feature known/unknown rejector for controlled ablations."""
    known_features = build_rejector_features(known_outputs, feature_mode=feature_mode)
    unknown_features = build_rejector_features(unknown_outputs, feature_mode=feature_mode)
    if known_features.ndim != 2 or unknown_features.ndim != 2:
        raise ValueError("rejector features must be 2-D")
    if known_features.shape[1] != unknown_features.shape[1]:
        raise ValueError("known and unknown rejector features must have matching dimensions")
    rng = np.random.default_rng(int(seed))

    def sample_rows(values):
        if len(values) <= max_samples:
            return values
        return values[rng.choice(len(values), size=max_samples, replace=False)]

    known_features = sample_rows(known_features)
    unknown_features = sample_rows(unknown_features)
    features = np.concatenate([known_features, unknown_features], axis=0)
    labels = np.concatenate(
        [np.zeros(len(known_features), dtype=np.int64), np.ones(len(unknown_features), dtype=np.int64)]
    )
    if model_type not in {"logistic", "mlp"}:
        raise ValueError(f"Unsupported rejector model type: {model_type}")
    if model_type == "mlp":
        rejector = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=int(seed),
            ),
        )
    elif feature_mode in {
        "augmented",
        "support_augmented",
        "uncertainty_augmented",
        "support_uncertainty_augmented",
    }:
        rejector = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=int(seed),
            ),
        )
    else:
        rejector = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=int(seed),
        )
    rejector.fit(features, labels)
    return rejector


def fit_virtual_outlier_rejector(
    known_outputs: Dict[str, np.ndarray],
    max_samples: int = 5000,
    seed: int = 42,
):
    """Fit a known-only rejector with class-boundary virtual outliers.

    This is a feature-space auxiliary experiment inspired by virtual-outlier
    methods such as VOS/NPOS. It does not use unknown labels or an unknown
    discovery pool. Pairs of different known classes are mixed on the unit
    sphere to create hard negatives near the known decision boundaries.
    """
    features = np.asarray(known_outputs["features"], dtype=np.float32)
    labels = np.asarray(known_outputs.get("labels"), dtype=np.int64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("known outputs must contain a non-empty 2-D feature array")
    if labels.shape != (len(features),):
        raise ValueError("known outputs labels must match the feature count")
    if np.any(labels < 0):
        raise ValueError("virtual outlier synthesis requires known class labels")

    rng = np.random.default_rng(int(seed))
    normalized = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-6, None)
    groups = [np.flatnonzero(labels == cls) for cls in np.unique(labels)]
    groups = [group for group in groups if len(group) > 0]
    if len(groups) < 2:
        raise ValueError("virtual outlier synthesis requires at least two known classes")
    count = min(int(max_samples), len(normalized))
    left = np.empty(count, dtype=np.int64)
    right = np.empty(count, dtype=np.int64)
    for index in range(count):
        first_group, second_group = rng.choice(len(groups), size=2, replace=False)
        left[index] = rng.choice(groups[first_group])
        right[index] = rng.choice(groups[second_group])

    first = normalized[left]
    second = normalized[right]
    midpoint = first + second
    midpoint /= np.clip(np.linalg.norm(midpoint, axis=1, keepdims=True), 1e-6, None)
    extrapolated = 1.5 * first - 0.5 * second
    extrapolated /= np.clip(np.linalg.norm(extrapolated, axis=1, keepdims=True), 1e-6, None)
    virtual = np.concatenate([midpoint, extrapolated], axis=0).astype(np.float32)
    known = normalized.astype(np.float32)
    train_features = np.concatenate([known, virtual], axis=0)
    train_labels = np.concatenate(
        [np.zeros(len(known), dtype=np.int64), np.ones(len(virtual), dtype=np.int64)]
    )
    rejector = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=int(seed),
        ),
    )
    rejector.fit(train_features, train_labels)
    return rejector


def fit_known_support_rejector(
    known_outputs: Dict[str, np.ndarray],
    quantile: float = 0.95,
):
    """Fit a known-only class-conditional support boundary."""
    features = np.asarray(known_outputs["features"], dtype=np.float32)
    labels = np.asarray(known_outputs.get("labels"), dtype=np.int64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("known outputs must contain a non-empty 2-D feature array")
    if labels.shape != (len(features),) or np.any(labels < 0):
        raise ValueError("known support fitting requires one valid label per feature")
    quantile = float(quantile)
    if not 0.5 <= quantile < 1.0:
        raise ValueError("support quantile must be in [0.5, 1.0)")

    normalized = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-6, None)
    classes = np.unique(labels)
    centers = []
    radii = []
    all_distances = []
    for cls in classes:
        class_features = normalized[labels == cls]
        center = class_features.mean(axis=0)
        center /= np.clip(np.linalg.norm(center), 1e-6, None)
        distances = 1.0 - class_features @ center
        centers.append(center)
        radii.append(float(np.quantile(distances, quantile)))
        all_distances.append(distances)
    fallback = float(np.quantile(np.concatenate(all_distances), quantile))
    radii = np.maximum(np.asarray(radii, dtype=np.float32), max(fallback * 0.25, 1e-3))
    return {
        "classes": np.asarray(classes, dtype=np.int64),
        "centers": np.asarray(centers, dtype=np.float32),
        "radii": radii,
        "quantile": quantile,
    }


def attach_known_support_score(
    outputs: Dict[str, np.ndarray], support_model: Dict[str, np.ndarray]
) -> None:
    """Attach a higher-is-more-unknown class-conditional support score."""
    features = np.asarray(outputs["features"], dtype=np.float32)
    features = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-6, None)
    centers = np.asarray(support_model["centers"], dtype=np.float32)
    centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-6, None)
    radii = np.asarray(support_model["radii"], dtype=np.float32)
    distances = 1.0 - features @ centers.T
    normalized_distances = distances / np.clip(radii[None, :], 1e-6, None)
    outputs["known_support_score"] = normalized_distances.min(axis=1).astype(np.float64)


def fit_pu_feature_rejector(
    known_outputs: Dict[str, np.ndarray],
    unlabeled_outputs: Dict[str, np.ndarray],
    max_samples: int = 5000,
    iterations: int = 4,
    seed: int = 42,
    feature_mode: str = "embedding",
):
    """Fit a soft positive-unlabeled feature rejector.

    Known features are the reliable in-distribution class. The discovery pool
    is treated as unlabeled: it contributes to the unknown class only through
    a continuously updated probability weight, rather than a hard unknown
    label. This reduces the damage caused by known samples contaminating a
    mixed discovery pool.
    """
    known_features = build_rejector_features(known_outputs, feature_mode=feature_mode)
    unlabeled_features = build_rejector_features(unlabeled_outputs, feature_mode=feature_mode)
    if known_features.ndim != 2 or unlabeled_features.ndim != 2:
        raise ValueError("rejector features must be 2-D")
    if known_features.shape[1] != unlabeled_features.shape[1]:
        raise ValueError("known and unlabeled rejector features must have matching dimensions")
    rng = np.random.default_rng(int(seed))

    def sample_rows(values):
        if len(values) <= max_samples:
            return values
        return values[rng.choice(len(values), size=max_samples, replace=False)]

    known_features = sample_rows(known_features)
    unlabeled_features = sample_rows(unlabeled_features)
    known_features = known_features / np.clip(
        np.linalg.norm(known_features, axis=1, keepdims=True), 1e-6, None
    )
    unlabeled_features = unlabeled_features / np.clip(
        np.linalg.norm(unlabeled_features, axis=1, keepdims=True), 1e-6, None
    )
    features = np.concatenate([known_features, unlabeled_features], axis=0)
    labels = np.concatenate(
        [np.zeros(len(known_features), dtype=np.int64), np.ones(len(unlabeled_features), dtype=np.int64)]
    )
    unlabeled_weight = np.ones(len(unlabeled_features), dtype=np.float64)
    rejector = None
    for _ in range(max(1, int(iterations))):
        sample_weight = np.concatenate(
            [np.ones(len(known_features), dtype=np.float64), unlabeled_weight]
        )
        if feature_mode in {"augmented", "support_augmented"}:
            rejector = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=int(seed),
                ),
            )
        else:
            rejector = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=int(seed),
            )
        rejector.fit(features, labels, sample_weight=sample_weight)
        probabilities = rejector.predict_proba(unlabeled_features)[:, 1]
        # Keep a small floor so the model does not discard an entire region of
        # the unlabeled pool after one unstable iteration.
        updated = np.clip(probabilities, 0.05, 1.0)
        unlabeled_weight = 0.5 * unlabeled_weight + 0.5 * updated
    return rejector


class NnPUFeatureRejector:
    """Linear unknown-score model fitted with a non-negative PU risk.

    The positive distribution is the known training set.  The discovery pool
    remains unlabeled and is treated as a mixture of known and unknown data;
    ``known_prior`` is the estimated known fraction in that mixture.  This is
    a controlled mixed-pool baseline, not a claim that the deployment prior is
    known exactly.
    """

    def __init__(self, mean, scale, weight, bias):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)
        self.weight_ = np.asarray(weight, dtype=np.float32)
        self.bias_ = float(bias)

    def _transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return (values - self.mean_) / self.scale_

    def decision_function(self, values):
        transformed = self._transform(values)
        return transformed @ self.weight_ + self.bias_

    def predict_proba(self, values):
        scores = self.decision_function(values)
        probability = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
        return np.stack([1.0 - probability, probability], axis=1)


def fit_nnpu_feature_rejector(
    known_outputs: Dict[str, np.ndarray],
    unlabeled_outputs: Dict[str, np.ndarray],
    known_prior: float = 0.2,
    max_samples: int = 5000,
    iterations: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 42,
    feature_mode: str = "embedding",
):
    """Fit a mixed-pool rejector with non-negative PU risk estimation."""
    prior = float(known_prior)
    if not 0.0 < prior < 1.0:
        raise ValueError("known_prior must be in (0, 1) for nnPU")
    known = build_rejector_features(known_outputs, feature_mode=feature_mode)
    unlabeled = build_rejector_features(unlabeled_outputs, feature_mode=feature_mode)
    if known.ndim != 2 or unlabeled.ndim != 2 or known.shape[1] != unlabeled.shape[1]:
        raise ValueError("known and unlabeled rejector features must be matching 2-D arrays")
    if len(known) == 0 or len(unlabeled) == 0:
        raise ValueError("nnPU requires non-empty known and unlabeled features")
    rng = np.random.default_rng(int(seed))

    def sample_rows(values):
        if len(values) <= max_samples:
            return values
        return values[rng.choice(len(values), size=max_samples, replace=False)]

    known = sample_rows(known).astype(np.float32, copy=False)
    unlabeled = sample_rows(unlabeled).astype(np.float32, copy=False)
    combined = np.concatenate([known, unlabeled], axis=0)
    mean = combined.mean(axis=0)
    scale = combined.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    known = (known - mean) / scale
    unlabeled = (unlabeled - mean) / scale

    torch.manual_seed(int(seed))
    known_tensor = torch.from_numpy(known)
    unlabeled_tensor = torch.from_numpy(unlabeled)
    weight = torch.zeros(known.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=float(lr), weight_decay=float(weight_decay))
    known_zero = torch.zeros(len(known_tensor))
    known_one = torch.ones(len(known_tensor))
    unknown_one = torch.ones(len(unlabeled_tensor))
    for _ in range(max(1, int(iterations))):
        known_logits = known_tensor @ weight + bias
        unlabeled_logits = unlabeled_tensor @ weight + bias
        known_as_unknown = torch.nn.functional.binary_cross_entropy_with_logits(
            known_logits, known_zero
        )
        known_as_known = torch.nn.functional.binary_cross_entropy_with_logits(
            known_logits, known_one
        )
        unlabeled_as_unknown = torch.nn.functional.binary_cross_entropy_with_logits(
            unlabeled_logits, unknown_one
        )
        negative_risk = (
            unlabeled_as_unknown - prior * known_as_known
        ) / max(1.0 - prior, 1e-6)
        loss = prior * known_as_unknown + torch.relu(negative_risk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return NnPUFeatureRejector(
        mean,
        scale,
        weight.detach().cpu().numpy(),
        bias.detach().cpu().item(),
    )


def attach_feature_rejector_score(
    outputs: Dict[str, np.ndarray], rejector, feature_mode: str = "embedding"
) -> None:
    """Attach a higher-is-more-unknown decision score to extracted outputs."""
    features = build_rejector_features(outputs, feature_mode=feature_mode)
    if hasattr(rejector, "decision_function"):
        score = rejector.decision_function(features)
    else:
        probability = rejector.predict_proba(features)[:, 1]
        probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
        score = np.log(probability) - np.log1p(-probability)
    outputs["feature_rejector_score"] = np.asarray(score).astype(np.float64)


@torch.no_grad()
def collect_activation_clip_value(model, loader, device, percentile: float) -> float:
    """Estimate a ReAct activation cap from known training features only."""
    values = []
    for batch in tqdm(loader, desc="react-stats", leave=False):
        images = batch[0].to(device)
        features = model(images)["features"].detach().flatten().cpu()
        values.append(features)
    if not values:
        return float("inf")
    joined = torch.cat(values).numpy()
    return float(np.percentile(joined, float(np.clip(percentile, 0.0, 100.0))))


@torch.no_grad()
def collect_knn_feature_bank(
    model,
    loader,
    device,
    max_samples: int = 0,
    seed: int = 0,
    feature_key: str = "features",
    support_k: int = 10,
    support_quantile: float = 0.95,
):
    """Collect normalized known-training vectors for KNN-OOD."""
    model.eval()
    bank = []
    bank_labels = []
    for batch in tqdm(loader, desc="knn-bank", leave=False):
        images, labels, *_ = batch
        known = labels >= 0
        if not torch.any(known):
            continue
        images = images[known].to(device)
        if feature_key not in {"features", "proj"}:
            raise ValueError("KNN feature_key must be 'features' or 'proj'.")
        bank.append(F.normalize(model(images)[feature_key], dim=-1).cpu())
        bank_labels.append(labels[known].detach().cpu().long())
    if not bank:
        raise ValueError("KNN-OOD requires at least one known training feature.")
    bank = torch.cat(bank, dim=0)
    labels = torch.cat(bank_labels, dim=0)
    if max_samples > 0 and len(bank) > max_samples:
        generator = torch.Generator().manual_seed(int(seed))
        indices = torch.randperm(len(bank), generator=generator)[:max_samples]
        bank = bank[indices]
        labels = labels[indices]
    features = bank.numpy()
    label_array = labels.numpy()
    return {
        "features": features,
        "labels": label_array,
        "class_support_radii": estimate_classwise_knn_radii(
            features,
            label_array,
            k=support_k,
            quantile=support_quantile,
        ),
    }


def estimate_classwise_knn_radii(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    quantile: float = 0.95,
) -> np.ndarray:
    """Estimate per-class cosine support radii from leave-one-out train kNN.

    Each training vector is excluded from its own neighborhood. Classes with
    fewer than two bank samples use the pooled known-class radius as fallback.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("kNN radius features and labels must be aligned 2-D/1-D arrays")
    if not 0.5 <= float(quantile) < 1.0:
        raise ValueError("kNN support quantile must be in [0.5, 1.0)")
    if int(k) <= 0:
        raise ValueError("kNN radius neighbor count must be positive")
    if not len(features):
        raise ValueError("cannot estimate kNN radii from an empty feature bank")
    features = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    unique_labels = np.unique(labels)
    all_loo_distances = []
    class_distances = {}
    for class_index in unique_labels:
        class_features = features[labels == class_index]
        if len(class_features) < 2:
            continue
        similarities = class_features @ class_features.T
        np.fill_diagonal(similarities, -np.inf)
        neighbor_count = min(int(k), len(class_features) - 1)
        nearest = np.partition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:]
        loo_distances = 1.0 - nearest.mean(axis=1)
        class_distances[int(class_index)] = loo_distances
        all_loo_distances.append(loo_distances)
    if not all_loo_distances:
        raise ValueError("at least one class needs two samples to estimate kNN support")
    pooled_radius = float(np.quantile(np.concatenate(all_loo_distances), quantile))
    radii = np.full(int(labels.max()) + 1, pooled_radius, dtype=np.float32)
    for class_index, loo_distances in class_distances.items():
        radii[class_index] = float(np.quantile(loo_distances, quantile))
    return np.clip(radii, 1e-6, None)


def attach_knn_distances(
    outputs: Dict[str, np.ndarray],
    feature_bank: np.ndarray,
    k: int = 10,
    feature_key: str = "features",
):
    """Attach mean cosine distance to the k nearest known-training features."""
    output_key = "features" if feature_key == "features" else "projections"
    if feature_key not in {"features", "proj"} or output_key not in outputs:
        raise ValueError("KNN-OOD requires matching extracted features or projections.")
    if k <= 0:
        raise ValueError("KNN-OOD k must be positive.")
    bank_labels = None
    if isinstance(feature_bank, dict):
        bank_labels = np.asarray(feature_bank["labels"], dtype=np.int64)
        bank = np.array(feature_bank["features"], dtype=np.float32, copy=True)
        if len(bank_labels) != len(bank):
            raise ValueError("KNN feature bank labels must match its feature count.")
    else:  # Backwards compatibility for callers with an unlabeled feature bank.
        bank = np.array(feature_bank, dtype=np.float32, copy=True)
    bank /= np.clip(np.linalg.norm(bank, axis=1, keepdims=True), 1e-8, None)
    queries = np.array(outputs[output_key], dtype=np.float32, copy=True)
    queries /= np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8, None)
    neighbor_count = min(int(k), len(bank))
    distances = np.empty(len(queries), dtype=np.float32)
    support = np.empty(len(queries), dtype=np.float32) if bank_labels is not None else None
    predicted_class_distance = (
        np.empty(len(queries), dtype=np.float32)
        if bank_labels is not None and "logits" in outputs
        else None
    )
    predicted_class_relative_distance = (
        np.empty(len(queries), dtype=np.float32)
        if bank_labels is not None
        and predicted_class_distance is not None
        and isinstance(feature_bank, dict)
        and "class_support_radii" in feature_bank
        else None
    )
    # Unlike the predicted-class distance, this statistic checks every known
    # class support region. It is useful when an unknown sample is confidently
    # assigned to the wrong known class.
    min_class_distance = (
        np.full(len(queries), np.inf, dtype=np.float32)
        if bank_labels is not None
        else None
    )
    min_class_relative_distance = (
        np.full(len(queries), np.inf, dtype=np.float32)
        if (
            bank_labels is not None
            and isinstance(feature_bank, dict)
            and "class_support_radii" in feature_bank
        )
        else None
    )
    predicted_classes = (
        np.asarray(outputs["logits"]).argmax(axis=1)
        if predicted_class_distance is not None
        else None
    )
    predicted_class_support = (
        np.empty(len(queries), dtype=np.float32)
        if bank_labels is not None and predicted_classes is not None
        else None
    )
    batch_size = 256
    for start in range(0, len(queries), batch_size):
        similarities = queries[start : start + batch_size] @ bank.T
        nearest_indices = np.argpartition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:]
        nearest = np.take_along_axis(similarities, nearest_indices, axis=1)
        distances[start : start + batch_size] = 1.0 - nearest.mean(axis=1)
        if bank_labels is not None:
            neighbor_labels = bank_labels[nearest_indices]
            # Similarity-weighted class votes reduce sensitivity to a single
            # accidental neighbor while retaining the known class identity.
            weights = np.exp((nearest - nearest.max(axis=1, keepdims=True)) / 0.07)
            class_support = np.zeros((len(nearest), int(bank_labels.max()) + 1), dtype=np.float32)
            rows = np.arange(len(nearest))[:, None]
            np.add.at(class_support, (rows, neighbor_labels), weights)
            support[start : start + batch_size] = class_support.max(axis=1) / np.clip(
                weights.sum(axis=1), 1e-12, None
            )
            if predicted_class_support is not None:
                predicted_class_support[start : start + len(nearest)] = (
                    weights
                    * (neighbor_labels == predicted_classes[start : start + len(nearest), None])
                ).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-12, None)
            if predicted_class_distance is not None:
                batch_predictions = predicted_classes[start : start + batch_size]
                for class_index in np.unique(batch_predictions):
                    rows = np.flatnonzero(batch_predictions == class_index)
                    class_bank = bank[bank_labels == class_index]
                    if len(class_bank) == 0:
                        predicted_class_distance[start + rows] = distances[start + rows]
                        if predicted_class_relative_distance is not None:
                            radii = np.asarray(feature_bank["class_support_radii"], dtype=float)
                            predicted_class_relative_distance[start + rows] = (
                                distances[start + rows] / max(float(np.median(radii)), 1e-6)
                            )
                        continue
                    class_similarities = queries[start + rows] @ class_bank.T
                    class_k = min(int(k), len(class_bank))
                    class_neighbors = np.partition(
                        class_similarities, -class_k, axis=1
                    )[:, -class_k:]
                    predicted_class_distance[start + rows] = 1.0 - class_neighbors.mean(axis=1)
                    if predicted_class_relative_distance is not None:
                        radii = np.asarray(feature_bank["class_support_radii"], dtype=float)
                        radius = radii[class_index] if class_index < len(radii) else np.median(radii)
                        predicted_class_relative_distance[start + rows] = (
                            predicted_class_distance[start + rows] / max(float(radius), 1e-6)
                        )
            if min_class_distance is not None:
                # Compute the closest class-conditional support, rather than
                # trusting the classifier's predicted class.
                radii = (
                    np.asarray(feature_bank["class_support_radii"], dtype=float)
                    if min_class_relative_distance is not None
                    else None
                )
                for class_index in np.unique(bank_labels):
                    class_bank = bank[bank_labels == class_index]
                    if len(class_bank) == 0:
                        continue
                    class_similarities = queries[start : start + len(nearest)] @ class_bank.T
                    class_k = min(int(k), len(class_bank))
                    class_neighbors = np.partition(
                        class_similarities, -class_k, axis=1
                    )[:, -class_k:]
                    class_distance = 1.0 - class_neighbors.mean(axis=1)
                    min_class_distance[start : start + len(nearest)] = np.minimum(
                        min_class_distance[start : start + len(nearest)], class_distance
                    )
                    if min_class_relative_distance is not None:
                        radius = float(radii[class_index]) if class_index < len(radii) else float(np.median(radii))
                        min_class_relative_distance[start : start + len(nearest)] = np.minimum(
                            min_class_relative_distance[start : start + len(nearest)],
                            class_distance / max(radius, 1e-6),
                        )
    outputs["knn_distance"] = distances
    if support is not None:
        outputs["knn_class_support"] = support
    if predicted_class_distance is not None:
        outputs["knn_predicted_class_distance"] = predicted_class_distance
    if predicted_class_support is not None:
        outputs["knn_predicted_class_support"] = predicted_class_support
    if predicted_class_relative_distance is not None:
        outputs["knn_predicted_class_relative_distance"] = predicted_class_relative_distance
    if min_class_distance is not None:
        outputs["knn_min_class_distance"] = min_class_distance
    if min_class_relative_distance is not None:
        outputs["knn_min_class_relative_distance"] = min_class_relative_distance
    return outputs


@torch.no_grad()
def collect_vim_stats(model, loader, device, rank: int = 64):
    """Fit a VIM-style known feature principal subspace from train data only."""
    model.eval()
    features = []
    for batch in tqdm(loader, desc="vim-stats", leave=False):
        images, labels, *_ = batch
        known = labels >= 0
        if not torch.any(known):
            continue
        images = images[known].to(device)
        features.append(model(images)["features"].cpu())
    if not features:
        raise ValueError("VIM requires at least one known training feature.")
    features = torch.cat(features, dim=0).numpy().astype(np.float32)
    center = features.mean(axis=0)
    centered = features - center
    max_rank = min(centered.shape[0] - 1, centered.shape[1])
    rank = max(1, min(int(rank), max_rank))
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:rank].astype(np.float32)
    return {"center": center.astype(np.float32), "components": components}


def attach_vim_residual(outputs: Dict[str, np.ndarray], vim_stats: Dict[str, np.ndarray]):
    """Attach distance from the known-training principal feature subspace."""
    features = np.asarray(outputs["features"], dtype=np.float32)
    center = np.asarray(vim_stats["center"], dtype=np.float32)
    components = np.asarray(vim_stats["components"], dtype=np.float32)
    centered = features - center
    projected = centered @ components.T @ components
    outputs["vim_residual"] = np.linalg.norm(centered - projected, axis=1)
    return outputs


def calibration_diagnostics(probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> dict:
    """Compute ECE-style reliability bins on labeled known-validation samples."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid = labels >= 0
    probabilities = probabilities[valid]
    labels = labels[valid]
    if len(labels) == 0:
        return {"ece": float("nan"), "accuracy": float("nan"), "bins": []}

    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins = []
    ece = 0.0
    for index in range(num_bins):
        upper_inclusive = index == num_bins - 1
        mask = (confidence >= edges[index]) & (
            (confidence <= edges[index + 1]) if upper_inclusive else (confidence < edges[index + 1])
        )
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(mean_confidence - accuracy)
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": float(ece), "accuracy": float(correct.mean()), "bins": bins}


def uncertainty_error_diagnostics(outputs: Dict[str, np.ndarray]) -> dict:
    """Measure whether the auxiliary uncertainty head tracks known-class errors."""
    labels = np.asarray(outputs["labels"], dtype=int)
    valid = labels >= 0
    if int(valid.sum()) < 2:
        return {"error_rate": float("nan"), "correlation": float("nan")}
    errors = (np.asarray(outputs["logits"])[valid].argmax(axis=1) != labels[valid]).astype(float)
    uncertainty = np.asarray(outputs["head_uncertainty"], dtype=float)[valid]
    if np.std(errors) == 0 or np.std(uncertainty) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(errors, uncertainty)[0, 1])
    return {"error_rate": float(errors.mean()), "correlation": correlation}


def fit_temperature(outputs: Dict[str, np.ndarray]) -> float:
    """Fit a scalar temperature on known-validation logits."""
    labels = torch.as_tensor(outputs["labels"], dtype=torch.long)
    logits = torch.as_tensor(outputs["logits"], dtype=torch.float32)
    valid = labels >= 0
    if int(valid.sum()) < 2:
        return 1.0

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(logits[valid] / temperature, labels[valid])
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 20.0).item())


def apply_temperature(outputs: Dict[str, np.ndarray], temperature: float) -> Dict[str, np.ndarray]:
    """Return calibrated logits/probabilities without shifting logits used by Energy."""
    calibrated = dict(outputs)
    scaled_logits = np.asarray(outputs["logits"], dtype=float) / max(float(temperature), 1e-6)
    stable_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    probs = np.exp(stable_logits)
    probs /= probs.sum(axis=1, keepdims=True)
    calibrated["logits"] = scaled_logits
    calibrated["probs"] = probs
    calibrated["entropy"] = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=1)
    return calibrated


def compute_prototype_distance(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    feat = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    proto = prototypes / np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8, None)
    sim = feat @ proto.T
    return 1.0 - sim.max(axis=1)


def collect_diagonal_gaussian_stats(
    model, loader, device, num_classes: int, include_shared_covariance: bool = True
):
    """Collect class means plus diagonal and shared-covariance statistics."""
    model.eval()
    features_by_class = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for batch in loader:
            images, labels, *_ = batch
            images = images.to(device)
            features = model(images)["features"].cpu().numpy()
            labels = labels.numpy()
            for feat, label in zip(features, labels):
                if 0 <= int(label) < num_classes:
                    features_by_class[int(label)].append(feat)
    means = []
    variances = []
    global_features = np.concatenate(
        [np.asarray(items) for items in features_by_class if items], axis=0
    )
    global_var = np.var(global_features, axis=0) + 1e-2
    for items in features_by_class:
        values = np.asarray(items)
        if len(values) < 2:
            means.append(np.zeros(global_features.shape[1], dtype=np.float32))
            variances.append(global_var)
        else:
            means.append(values.mean(axis=0))
            variances.append(np.var(values, axis=0) + 1e-2)
    means = np.asarray(means, dtype=np.float32)
    variances = np.asarray(variances, dtype=np.float32)

    # OpenMax-style EVT calibration: fit a Weibull model to the largest
    # distances of each known class from its mean activation vector. This is
    # used only when ``score_mode=openmax`` and does not alter training.
    weibull_shapes = []
    weibull_scales = []
    weibull_locations = []
    for class_index, items in enumerate(features_by_class):
        values = np.asarray(items, dtype=np.float32)
        if len(values) < 2:
            distances = np.array([1.0, 1.0], dtype=np.float64)
        else:
            distances = np.linalg.norm(values - means[class_index], axis=1).astype(np.float64)
        tail_size = min(len(distances), max(5, int(np.ceil(0.2 * len(distances)))))
        tail = np.sort(distances)[-tail_size:]
        tail = np.clip(tail, 1e-6, None)
        try:
            shape, location, scale = weibull_min.fit(tail, floc=0.0)
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("invalid Weibull fit")
        except Exception:
            shape, location, scale = 1.0, 0.0, float(max(np.mean(tail), 1e-3))
        weibull_shapes.append(float(shape))
        weibull_locations.append(float(location))
        weibull_scales.append(float(scale))

    centered = []
    for class_index, items in enumerate(features_by_class):
        if items:
            centered.append(np.asarray(items, dtype=np.float32) - means[class_index])
    centered_features = np.concatenate(centered, axis=0) if centered else global_features - global_features.mean(axis=0)
    feature_dim = centered_features.shape[1]
    precision = None
    shrinkage = 0.1
    if include_shared_covariance:
        if len(centered_features) <= 1:
            covariance = np.diag(global_var)
        else:
            covariance = (centered_features.T @ centered_features) / max(len(centered_features) - 1, 1)
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        covariance = covariance + np.eye(feature_dim, dtype=np.float32) * 1e-3
        with threadpool_limits(limits=1):
            precision = np.linalg.pinv(covariance).astype(np.float32)
    # Use the law of total variance with equal class priors to estimate the
    # background distribution diagonally. This avoids a second dense inverse
    # and is appropriate for the class-balanced CIFAR protocol.
    global_mean = means.mean(axis=0).astype(np.float32)
    global_variances = (
        np.mean(variances + (means - global_mean[None, :]) ** 2, axis=0) + 1e-2
    ).astype(np.float32)
    result = {
        "means": means,
        "variances": variances,
        "global_mean": global_mean,
        "global_variances": global_variances,
        "weibull_shapes": np.asarray(weibull_shapes, dtype=np.float32),
        "weibull_locations": np.asarray(weibull_locations, dtype=np.float32),
        "weibull_scales": np.asarray(weibull_scales, dtype=np.float32),
    }
    if precision is not None:
        result["precision"] = precision
    return result


def compute_mahalanobis_distance(
    features: np.ndarray,
    gaussian_stats: Dict[str, np.ndarray],
    covariance: str = "auto",
    chunk_size: int = 512,
) -> np.ndarray:
    distances = compute_mahalanobis_class_distances(
        features,
        gaussian_stats,
        covariance=covariance,
        chunk_size=chunk_size,
    )
    return distances.min(axis=1) if len(distances) else np.empty(0, dtype=float)


def compute_mahalanobis_class_distances(
    features: np.ndarray,
    gaussian_stats: Dict[str, np.ndarray],
    covariance: str = "auto",
    chunk_size: int = 512,
) -> np.ndarray:
    """Return one Mahalanobis distance per sample and known class.

    Keeping the class dimension is useful for class-conditional calibration:
    a distance of 2.0 can be normal for one class and highly atypical for
    another.  Computation remains chunked to avoid materialising the full
    sample-by-class-by-feature tensor.
    """
    features = np.asarray(features)
    means = np.asarray(gaussian_stats["means"])
    if features.ndim != 2 or means.ndim != 2:
        raise ValueError("features and gaussian means must be 2-D arrays")
    if features.shape[1] != means.shape[1]:
        raise ValueError("feature dimension does not match Gaussian means")
    chunk_size = max(1, int(chunk_size))
    use_shared = covariance == "shared" or (covariance == "auto" and "precision" in gaussian_stats)
    distance_chunks = []
    if use_shared:
        if "precision" not in gaussian_stats:
            raise ValueError("shared Mahalanobis score requires gaussian_stats['precision']")
        precision = np.asarray(gaussian_stats["precision"], dtype=float)
        # Expand (x - mu_c)^T P (x - mu_c) before applying the class
        # dimension.  The direct einsum over [sample, class, feature,
        # feature] is mathematically correct but needlessly costs O(N*C*D^2)
        # and becomes impractical for full CIFAR-100 feature extraction.
        transformed = np.asarray(features, dtype=float) @ precision
        quadratic_x = np.sum(transformed * np.asarray(features, dtype=float), axis=1)[:, None]
        transformed_means = np.asarray(means, dtype=float) @ precision
        cross = transformed @ np.asarray(means, dtype=float).T
        quadratic_means = np.sum(transformed_means * np.asarray(means, dtype=float), axis=1)[None, :]
        distances = (quadratic_x - 2.0 * cross + quadratic_means) / max(features.shape[1], 1)
        return np.maximum(distances, 0.0)
    else:
        variances = np.asarray(gaussian_stats["variances"])
        for start in range(0, len(features), chunk_size):
            feat = features[start : start + chunk_size, None, :]
            diff = feat - means[None, :, :]
            distances = (diff**2 / np.clip(variances[None, :, :], 1e-6, None)).mean(axis=-1)
            distance_chunks.append(distances)
    return np.concatenate(distance_chunks, axis=0) if distance_chunks else np.empty((0, len(means)), dtype=float)


def compute_predicted_classwise_mahalanobis(
    outputs: Dict[str, np.ndarray],
    gaussian_stats: Dict[str, np.ndarray],
    covariance: str = "auto",
) -> np.ndarray:
    """Distance to the class selected by the known classifier.

    Unlike the usual minimum-over-classes distance, this preserves the
    classifier's decision boundary.  The resulting distance can then be
    standardised with statistics for the corresponding predicted class.
    """
    distances = compute_mahalanobis_class_distances(
        outputs["features"], gaussian_stats, covariance=covariance
    )
    if len(distances) == 0:
        return np.empty(0, dtype=float)
    predicted = np.asarray(outputs["logits"]).argmax(axis=1)
    predicted = np.clip(predicted, 0, distances.shape[1] - 1)
    return distances[np.arange(len(distances)), predicted]


def compute_relative_mahalanobis_distance(
    features: np.ndarray,
    gaussian_stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """Class-conditional Mahalanobis distance relative to the global feature density.

    This follows the relative-Mahalanobis idea: subtract the distance to a
    background/global feature distribution, reducing the bias toward regions
    that are globally low-density but not specifically associated with any
    known class.
    """
    required = ("means", "variances")
    if any(key not in gaussian_stats for key in required):
        raise ValueError("relative Mahalanobis requires class and global Gaussian statistics")
    features = np.asarray(features, dtype=float)
    if "precision" in gaussian_stats:
        class_distance = compute_mahalanobis_distance(features, gaussian_stats, covariance="shared")
    else:
        class_distance = compute_mahalanobis_distance(features, gaussian_stats, covariance="diag")
    means = np.asarray(gaussian_stats["means"], dtype=float)
    variances = np.asarray(gaussian_stats["variances"], dtype=float)
    global_mean = np.asarray(
        gaussian_stats.get("global_mean", means.mean(axis=0)), dtype=float
    )
    global_variances = gaussian_stats.get("global_variances")
    if global_variances is None:
        global_variances = np.mean(
            variances + (means - global_mean[None, :]) ** 2, axis=0
        ) + 1e-2
    centered = features - global_mean[None, :]
    background_distance = np.mean(
        (centered**2) / np.clip(np.asarray(global_variances, dtype=float)[None, :], 1e-6, None),
        axis=1,
    )
    return class_distance - background_distance


def compute_gaussian_nll(features: np.ndarray, gaussian_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Class-conditional diagonal Gaussian negative log-likelihood.

    This is a DDU-style density score: unlike plain Mahalanobis distance it
    also includes each class covariance's log-volume term.
    """
    feat = np.asarray(features, dtype=float)[:, None, :]
    means = np.asarray(gaussian_stats["means"], dtype=float)[None, :, :]
    variances = np.clip(np.asarray(gaussian_stats["variances"], dtype=float)[None, :, :], 1e-6, None)
    nll = 0.5 * (((feat - means) ** 2) / variances + np.log(2.0 * np.pi * variances))
    return nll.mean(axis=-1).min(axis=1)


def compute_openmax_score(features: np.ndarray, gaussian_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute an OpenMax-style EVT distance score.

    A sample is considered unknown when it is in the fitted tail for every
    known class. The minimum classwise Weibull CDF is therefore used: a close
    class keeps the score low, while a sample far from all class centers gets
    a high score.
    """
    required = ("means", "weibull_shapes", "weibull_locations", "weibull_scales")
    if any(key not in gaussian_stats for key in required):
        raise ValueError("openmax requires Weibull statistics from known training features")
    feat = np.asarray(features, dtype=float)[:, None, :]
    means = np.asarray(gaussian_stats["means"], dtype=float)[None, :, :]
    distances = np.linalg.norm(feat - means, axis=-1)
    shapes = np.clip(np.asarray(gaussian_stats["weibull_shapes"], dtype=float), 1e-3, None)
    locations = np.asarray(gaussian_stats["weibull_locations"], dtype=float)
    scales = np.clip(np.asarray(gaussian_stats["weibull_scales"], dtype=float), 1e-6, None)
    cdf = weibull_min.cdf(distances, shapes[None, :], loc=locations[None, :], scale=scales[None, :])
    return np.clip(np.nanmin(cdf, axis=1), 0.0, 1.0)


def compute_reciprocal_score(features: np.ndarray, reciprocal_points: np.ndarray) -> np.ndarray:
    """Score attraction to learned ARPL-inspired reciprocal points."""
    feat = np.asarray(features, dtype=float)
    points = np.asarray(reciprocal_points, dtype=float)
    if feat.ndim != 2 or points.ndim != 2 or feat.shape[1] != points.shape[1]:
        raise ValueError("features and reciprocal_points must be 2-D with matching dimensions")
    feat = feat / np.clip(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8, None)
    points = points / np.clip(np.linalg.norm(points, axis=1, keepdims=True), 1e-8, None)
    return (feat @ points.T).max(axis=1)


def compute_open_score(
    outputs: Dict[str, np.ndarray],
    prototypes: np.ndarray | None = None,
    weights=(0.5, 0.3, 0.2),
    score_mode: str = "full",
    normalization: Dict[str, Dict[str, float]] | None = None,
    gaussian_stats: Dict[str, np.ndarray] | None = None,
):
    entropy = outputs["entropy"]
    epistemic = outputs["epistemic"]
    aleatoric = outputs["aleatoric"]
    proto_dist = None
    if prototypes is not None:
        proto_dist = compute_prototype_distance(outputs["features"], prototypes)
    mahalanobis = None
    mahalanobis_mode = "auto"
    if score_mode.endswith("_diag"):
        mahalanobis_mode = "diag"
    elif score_mode.endswith("_shared"):
        mahalanobis_mode = "shared"
    mahalanobis_score_modes = {
        "mahalanobis",
        "mahalanobis_diag",
        "mahalanobis_shared",
        "entropy_mahalanobis",
        "entropy_mahalanobis_diag",
        "entropy_mahalanobis_shared",
        "normalized_entropy_mahalanobis",
        "normalized_entropy_mahalanobis_diag",
        "normalized_entropy_mahalanobis_shared",
        "normalized_entropy_mahalanobis_knn",
        "classwise_mahalanobis",
        "normalized_entropy_classwise_mahalanobis",
        "relative_mahalanobis",
        "normalized_entropy_relative_mahalanobis",
    }
    if gaussian_stats is not None and score_mode in mahalanobis_score_modes:
        if score_mode in {"relative_mahalanobis", "normalized_entropy_relative_mahalanobis"}:
            mahalanobis = compute_relative_mahalanobis_distance(
                outputs["features"], gaussian_stats
            )
        elif score_mode in {"classwise_mahalanobis", "normalized_entropy_classwise_mahalanobis"}:
            mahalanobis = compute_predicted_classwise_mahalanobis(
                outputs, gaussian_stats, covariance=mahalanobis_mode
            )
        else:
            mahalanobis = compute_mahalanobis_distance(
                outputs["features"], gaussian_stats, covariance=mahalanobis_mode
            )
    gaussian_nll = None
    gaussian_nll_score_modes = {"gaussian_nll", "normalized_entropy_gaussian_nll"}
    if gaussian_stats is not None and score_mode in gaussian_nll_score_modes:
        gaussian_nll = compute_gaussian_nll(outputs["features"], gaussian_stats)
    openmax_score = None
    if gaussian_stats is not None and score_mode == "openmax":
        openmax_score = compute_openmax_score(outputs["features"], gaussian_stats)
    reciprocal_score = None
    if gaussian_stats is not None and score_mode == "reciprocal":
        if "reciprocal_points" not in gaussian_stats:
            raise ValueError("reciprocal score requires a checkpoint trained with reciprocal points")
        reciprocal_score = compute_reciprocal_score(
            outputs["features"], gaussian_stats["reciprocal_points"]
        )

    if score_mode == "full":
        score = weights[0] * entropy + weights[1] * epistemic + weights[2] * aleatoric
        if proto_dist is not None:
            score = score + proto_dist
    elif score_mode == "normalized_full":
        if normalization is None:
            raise ValueError("normalized_full requires known-validation normalization")

        def zscore(values, key):
            if key not in normalization:
                raise ValueError(f"normalized_full requires normalization for {key}")
            stats = normalization[key]
            return (values - stats["mean"]) / max(float(stats["std"]), 1e-6)

        score = (
            weights[0] * zscore(entropy, "entropy")
            + weights[1] * zscore(epistemic, "epistemic")
            + weights[2] * zscore(aleatoric, "aleatoric")
        )
        if proto_dist is not None:
            score = score + zscore(proto_dist, "proto_dist")
    elif score_mode == "max_softmax":
        score = 1.0 - outputs["probs"].max(axis=1)
    elif score_mode == "max_logit":
        score = -np.asarray(outputs["logits"], dtype=float).max(axis=1)
    elif score_mode == "logit_margin":
        logits = np.asarray(outputs["logits"], dtype=float)
        top2 = np.sort(np.partition(logits, -2, axis=1)[:, -2:], axis=1)
        score = -(top2[:, 1] - top2[:, 0])
    elif score_mode == "novel_msp":
        if "novel_probs" not in outputs:
            raise ValueError("novel_msp score requires --novel-head-ckpt")
        score = outputs["novel_probs"].max(axis=1)
    elif score_mode == "novel_entropy":
        if "novel_probs" not in outputs:
            raise ValueError("novel_entropy score requires --novel-head-ckpt")
        probs = np.asarray(outputs["novel_probs"], dtype=float)
        entropy = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=1)
        score = 1.0 - entropy / np.log(max(probs.shape[1], 2))
    elif score_mode == "unified_novel_mass":
        if "unified_novel_mass" not in outputs:
            raise ValueError("unified_novel_mass requires --novel-head-ckpt")
        score = np.asarray(outputs["unified_novel_mass"], dtype=float)
    elif score_mode == "classwise_unified_novel_mass":
        if "unified_novel_mass" not in outputs:
            raise ValueError("classwise_unified_novel_mass requires --novel-head-ckpt")
        if normalization is None or "unified_novel_mass_classwise" not in normalization:
            raise ValueError(
                "classwise_unified_novel_mass requires known-validation classwise statistics"
            )
        raw_score = np.asarray(outputs["unified_novel_mass"], dtype=float)
        predicted_class = np.asarray(outputs["logits"]).argmax(axis=1)
        class_stats = normalization["unified_novel_mass_classwise"]
        means = np.asarray(class_stats["mean"], dtype=float)
        stds = np.asarray(class_stats["std"], dtype=float)
        predicted_class = np.clip(predicted_class, 0, len(means) - 1)
        score = (raw_score - means[predicted_class]) / np.maximum(stds[predicted_class], 1e-6)
    elif score_mode == "normalized_unified_novel_mass_entropy":
        if "unified_novel_mass" not in outputs or normalization is None:
            raise ValueError(
                "normalized_unified_novel_mass_entropy requires a joint novel head "
                "and known-validation normalization"
            )
        if "unified_novel_mass" not in normalization:
            raise ValueError(
                "normalized_unified_novel_mass_entropy requires unified novel-mass statistics"
            )
        novel_stats = normalization["unified_novel_mass"]
        entropy_stats = normalization["entropy"]
        novel_z = (
            np.asarray(outputs["unified_novel_mass"], dtype=float) - novel_stats["mean"]
        ) / max(float(novel_stats["std"]), 1e-6)
        entropy_z = (entropy - entropy_stats["mean"]) / max(float(entropy_stats["std"]), 1e-6)
        # The novel head is the primary signal; entropy contributes a smaller
        # uncertainty-aware correction for samples that remain overconfident.
        score = novel_z + 0.5 * entropy_z
    elif score_mode == "odin_msp":
        if "odin_msp" not in outputs:
            raise ValueError("odin_msp score requires --odin-epsilon greater than 0")
        score = np.asarray(outputs["odin_msp"], dtype=float)
    elif score_mode == "energy":
        logits = outputs["logits"]
        temperature = 1.0
        score = -temperature * np.logaddexp.reduce(logits / temperature, axis=1)
    elif score_mode == "react_energy":
        if "react_energy" not in outputs:
            raise ValueError("react_energy requires --react-percentile greater than 0")
        score = np.asarray(outputs["react_energy"], dtype=float)
    elif score_mode == "knn_distance":
        if "knn_distance" not in outputs:
            raise ValueError("knn_distance requires --knn-ood")
        score = np.asarray(outputs["knn_distance"], dtype=float)
    elif score_mode == "normalized_entropy_knn":
        if "knn_distance" not in outputs or normalization is None:
            raise ValueError("normalized_entropy_knn requires --knn-ood and known-validation normalization")
        score = (
            (entropy - normalization["entropy"]["mean"]) / normalization["entropy"]["std"]
            + (outputs["knn_distance"] - normalization["knn_distance"]["mean"])
            / normalization["knn_distance"]["std"]
        )
    elif score_mode == "predicted_class_knn_distance":
        if "knn_predicted_class_distance" not in outputs:
            raise ValueError(
                "predicted_class_knn_distance requires --knn-ood and labeled known training features"
            )
        score = np.asarray(outputs["knn_predicted_class_distance"], dtype=float)
    elif score_mode == "normalized_entropy_predicted_class_knn_global":
        if "knn_predicted_class_distance" not in outputs or normalization is None:
            raise ValueError(
                "normalized_entropy_predicted_class_knn_global requires --knn-ood "
                "and known-validation normalization"
            )
        score = (
            (entropy - normalization["entropy"]["mean"])
            / max(float(normalization["entropy"]["std"]), 1e-6)
            + (
                outputs["knn_predicted_class_distance"]
                - normalization["knn_predicted_class_distance"]["mean"]
            )
            / max(float(normalization["knn_predicted_class_distance"]["std"]), 1e-6)
        )
    elif score_mode == "normalized_entropy_predicted_class_knn":
        if "knn_predicted_class_distance" not in outputs or normalization is None:
            raise ValueError(
                "normalized_entropy_predicted_class_knn requires --knn-ood "
                "and known-validation normalization"
            )
        class_stats = normalization.get("knn_predicted_class_distance_classwise")
        if class_stats is None:
            raise ValueError(
                "normalized_entropy_predicted_class_knn requires classwise "
                "known-validation distance statistics"
            )
        predicted = np.asarray(outputs["logits"]).argmax(axis=1)
        means = np.asarray(class_stats["mean"], dtype=float)
        stds = np.asarray(class_stats["std"], dtype=float)
        predicted = np.clip(predicted, 0, len(means) - 1)
        class_distance_z = (
            np.asarray(outputs["knn_predicted_class_distance"], dtype=float)
            - means[predicted]
        ) / np.maximum(stds[predicted], 1e-6)
        score = (
            (entropy - normalization["entropy"]["mean"])
            / max(float(normalization["entropy"]["std"]), 1e-6)
            + class_distance_z
        )
    elif score_mode == "normalized_entropy_relative_predicted_class_knn":
        if "knn_predicted_class_relative_distance" not in outputs or normalization is None:
            raise ValueError(
                "normalized_entropy_relative_predicted_class_knn requires "
                "--knn-ood and known-validation normalization"
            )
        distance_stats = normalization["knn_predicted_class_relative_distance"]
        score = (
            (entropy - normalization["entropy"]["mean"])
            / max(float(normalization["entropy"]["std"]), 1e-6)
            + (
                outputs["knn_predicted_class_relative_distance"]
                - distance_stats["mean"]
            )
            / max(float(distance_stats["std"]), 1e-6)
        )
    elif score_mode in {
        "normalized_entropy_min_class_knn",
        "normalized_entropy_min_class_relative_knn",
    }:
        distance_key = (
            "knn_min_class_relative_distance"
            if score_mode.endswith("relative_knn")
            else "knn_min_class_distance"
        )
        if distance_key not in outputs or normalization is None:
            raise ValueError(
                f"{score_mode} requires --knn-ood and known-validation normalization"
            )
        distance_stats = normalization.get(distance_key)
        if distance_stats is None:
            raise ValueError(f"{score_mode} requires {distance_key} statistics")
        score = (
            (entropy - normalization["entropy"]["mean"])
            / max(float(normalization["entropy"]["std"]), 1e-6)
            + (
                np.asarray(outputs[distance_key], dtype=float)
                - distance_stats["mean"]
            )
            / max(float(distance_stats["std"]), 1e-6)
        )
    elif score_mode == "normalized_entropy_uncertainty_min_class_knn":
        if "knn_min_class_distance" not in outputs or normalization is None:
            raise ValueError(
                "normalized_entropy_uncertainty_min_class_knn requires --knn-ood "
                "and known-validation normalization"
            )
        distance_stats = normalization.get("knn_min_class_distance")
        uncertainty_stats = normalization.get("head_uncertainty")
        if distance_stats is None or uncertainty_stats is None:
            raise ValueError(
                "normalized_entropy_uncertainty_min_class_knn requires "
                "distance and head-uncertainty statistics"
            )
        entropy_z = (
            entropy - normalization["entropy"]["mean"]
        ) / max(float(normalization["entropy"]["std"]), 1e-6)
        distance_z = (
            np.asarray(outputs["knn_min_class_distance"], dtype=float)
            - distance_stats["mean"]
        ) / max(float(distance_stats["std"]), 1e-6)
        uncertainty_z = (
            np.asarray(outputs["head_uncertainty"], dtype=float)
            - uncertainty_stats["mean"]
        ) / max(float(uncertainty_stats["std"]), 1e-6)
        # Keep uncertainty as a correction instead of letting a noisy head
        # overwhelm the two independently useful geometry signals.
        score = entropy_z + distance_z + 0.5 * uncertainty_z
    elif score_mode == "normalized_entropy_knn_conflict":
        if "knn_predicted_class_support" not in outputs or normalization is None:
            raise ValueError(
                "normalized_entropy_knn_conflict requires --knn-ood "
                "and known-validation normalization"
            )
        support_stats = normalization["knn_predicted_class_support"]
        conflict = 1.0 - np.asarray(outputs["knn_predicted_class_support"], dtype=float)
        score = (
            (entropy - normalization["entropy"]["mean"])
            / max(float(normalization["entropy"]["std"]), 1e-6)
            + (conflict - (1.0 - support_stats["mean"]))
            / max(float(support_stats["std"]), 1e-6)
        )
    elif score_mode == "normalized_entropy_mahalanobis_knn":
        if "knn_distance" not in outputs or normalization is None or gaussian_stats is None:
            raise ValueError(
                "normalized_entropy_mahalanobis_knn requires --knn-ood, "
                "Gaussian statistics, and known-validation normalization"
            )
        mahalanobis_stats = normalization["mahalanobis"]
        mahalanobis = compute_mahalanobis_distance(outputs["features"], gaussian_stats)
        score = (
            (entropy - normalization["entropy"]["mean"]) / normalization["entropy"]["std"]
            + (mahalanobis - mahalanobis_stats["mean"]) / mahalanobis_stats["std"]
            + (outputs["knn_distance"] - normalization["knn_distance"]["mean"])
            / normalization["knn_distance"]["std"]
        )
    elif score_mode == "gaussian_nll":
        if gaussian_nll is None:
            raise ValueError("gaussian_nll requires known-training Gaussian statistics")
        score = gaussian_nll
    elif score_mode == "openmax":
        if openmax_score is None:
            raise ValueError("openmax requires Weibull statistics from known training features")
        score = openmax_score
    elif score_mode == "reciprocal":
        if reciprocal_score is None:
            raise ValueError("reciprocal score requires learned reciprocal points")
        score = reciprocal_score
    elif score_mode == "normalized_entropy_gaussian_nll":
        if gaussian_nll is None or normalization is None:
            raise ValueError("normalized_entropy_gaussian_nll requires statistics")
        score = (
            (entropy - normalization["entropy"]["mean"]) / normalization["entropy"]["std"]
            + (gaussian_nll - normalization["gaussian_nll"]["mean"])
            / normalization["gaussian_nll"]["std"]
        )
    elif score_mode == "relative_mahalanobis":
        if mahalanobis is None:
            raise ValueError("relative_mahalanobis requires class and global Gaussian statistics")
        score = mahalanobis
    elif score_mode == "normalized_entropy_relative_mahalanobis":
        if mahalanobis is None or normalization is None:
            raise ValueError("normalized_entropy_relative_mahalanobis requires statistics")
        score = (
            (entropy - normalization["entropy"]["mean"]) / normalization["entropy"]["std"]
            + (mahalanobis - normalization["relative_mahalanobis"]["mean"])
            / normalization["relative_mahalanobis"]["std"]
        )
    elif score_mode == "classwise_mahalanobis":
        if mahalanobis is None:
            raise ValueError("classwise_mahalanobis requires Gaussian statistics")
        if normalization is not None and "classwise_mahalanobis" in normalization:
            class_stats = normalization["classwise_mahalanobis"]
            predicted = np.asarray(outputs["logits"]).argmax(axis=1)
            means = np.asarray(class_stats["mean"], dtype=float)
            stds = np.maximum(np.asarray(class_stats["std"], dtype=float), 1e-6)
            predicted = np.clip(predicted, 0, len(means) - 1)
            score = (mahalanobis - means[predicted]) / stds[predicted]
        else:
            score = mahalanobis
    elif score_mode == "normalized_entropy_classwise_mahalanobis":
        if mahalanobis is None or normalization is None:
            raise ValueError(
                "normalized_entropy_classwise_mahalanobis requires statistics"
            )
        if "classwise_mahalanobis" not in normalization:
            raise ValueError(
                "normalized_entropy_classwise_mahalanobis requires classwise statistics"
            )
        class_stats = normalization["classwise_mahalanobis"]
        predicted = np.asarray(outputs["logits"]).argmax(axis=1)
        means = np.asarray(class_stats["mean"], dtype=float)
        stds = np.maximum(np.asarray(class_stats["std"], dtype=float), 1e-6)
        predicted = np.clip(predicted, 0, len(means) - 1)
        distance_z = (mahalanobis - means[predicted]) / stds[predicted]
        score = (
            (entropy - normalization["entropy"]["mean"])
            / normalization["entropy"]["std"]
            + distance_z
        )
    elif score_mode == "vim_residual":
        if "vim_residual" not in outputs:
            raise ValueError("vim_residual requires --vim-ood")
        score = np.asarray(outputs["vim_residual"], dtype=float)
    elif score_mode == "entropy_only":
        score = entropy
    elif score_mode == "proto_only":
        score = proto_dist if proto_dist is not None else np.zeros_like(entropy)
    elif score_mode == "entropy_proto":
        score = entropy
        if proto_dist is not None:
            score = score + proto_dist
    elif score_mode == "normalized_entropy_proto":
        if proto_dist is None or normalization is None:
            raise ValueError("normalized_entropy_proto requires prototypes and known-validation normalization")
        entropy_mean = normalization["entropy"]["mean"]
        entropy_std = normalization["entropy"]["std"]
        proto_mean = normalization["proto_dist"]["mean"]
        proto_std = normalization["proto_dist"]["std"]
        score = (entropy - entropy_mean) / entropy_std + (proto_dist - proto_mean) / proto_std
    elif score_mode == "entropy_epistemic":
        score = entropy + epistemic
        if proto_dist is not None:
            score = score + proto_dist
    elif score_mode == "entropy_aleatoric":
        score = outputs.get("expected_entropy", aleatoric)
        if proto_dist is not None:
            score = score + proto_dist
    elif score_mode == "expected_entropy":
        score = outputs.get("expected_entropy", aleatoric)
    elif score_mode == "epistemic":
        score = epistemic
    elif score_mode == "feature_norm":
        score = -np.asarray(outputs["feature_norm"], dtype=float)
    elif score_mode == "normalized_entropy_feature_norm":
        if "feature_norm" not in outputs or normalization is None:
            raise ValueError(
                "normalized_entropy_feature_norm requires feature norms and normalization"
            )
        score = (
            (entropy - normalization["entropy"]["mean"])
            / normalization["entropy"]["std"]
            - (np.asarray(outputs["feature_norm"], dtype=float) - normalization["feature_norm"]["mean"])
            / normalization["feature_norm"]["std"]
        )
    elif score_mode == "head_uncertainty":
        score = np.asarray(outputs["head_uncertainty"], dtype=float)
    elif score_mode == "feature_rejector":
        if "feature_rejector_score" not in outputs:
            raise ValueError("feature_rejector requires a fitted rejector")
        score = np.asarray(outputs["feature_rejector_score"], dtype=float)
    elif score_mode == "virtual_rejector":
        if "feature_rejector_score" not in outputs:
            raise ValueError("virtual_rejector requires a fitted rejector")
        score = np.asarray(outputs["feature_rejector_score"], dtype=float)
    elif score_mode == "known_support":
        if "known_support_score" not in outputs:
            raise ValueError("known_support requires a fitted support detector")
        score = np.asarray(outputs["known_support_score"], dtype=float)
    elif score_mode == "margin_uncertainty":
        logits = np.asarray(outputs["logits"], dtype=float)
        top2 = np.sort(np.partition(logits, -2, axis=1)[:, -2:], axis=1)
        margin_risk = -(top2[:, 1] - top2[:, 0])
        head = np.asarray(outputs["head_uncertainty"], dtype=float)
        score = margin_risk + head
    elif score_mode in {"mahalanobis", "mahalanobis_diag", "mahalanobis_shared"}:
        if mahalanobis is None:
            raise ValueError("mahalanobis score requires gaussian statistics")
        score = mahalanobis
    elif score_mode in {"entropy_mahalanobis", "entropy_mahalanobis_diag", "entropy_mahalanobis_shared"}:
        if mahalanobis is None:
            raise ValueError("entropy_mahalanobis score requires gaussian statistics")
        score = entropy + mahalanobis
    elif score_mode in {
        "normalized_entropy_mahalanobis",
        "normalized_entropy_mahalanobis_diag",
        "normalized_entropy_mahalanobis_shared",
    }:
        if mahalanobis is None or normalization is None:
            raise ValueError("normalized_entropy_mahalanobis requires statistics")
        stat_key = "mahalanobis"
        if score_mode.endswith("_diag"):
            stat_key = "mahalanobis_diag"
        elif score_mode.endswith("_shared"):
            stat_key = "mahalanobis_shared"
        score = (
            (entropy - normalization["entropy"]["mean"]) / normalization["entropy"]["std"]
            + (mahalanobis - normalization[stat_key]["mean"])
            / normalization[stat_key]["std"]
        )
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    return score, proto_dist


def fit_score_normalization(
    outputs_known: Dict[str, np.ndarray],
    prototypes: np.ndarray | None,
    gaussian_stats: Dict[str, np.ndarray] | None = None,
    include_mahalanobis: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Fit score statistics using known validation samples only."""
    entropy = np.asarray(outputs_known["entropy"], dtype=float)

    def stats(values):
        values = np.asarray(values, dtype=float)
        std = float(np.std(values))
        return {"mean": float(np.mean(values)), "std": max(std, 1e-6)}

    result = {"entropy": stats(entropy)}
    if "feature_norm" in outputs_known:
        result["feature_norm"] = stats(outputs_known["feature_norm"])
    result["epistemic"] = stats(outputs_known.get("epistemic", np.zeros_like(entropy)))
    result["aleatoric"] = stats(outputs_known.get("aleatoric", np.zeros_like(entropy)))
    result["head_uncertainty"] = stats(
        outputs_known.get("head_uncertainty", np.zeros_like(entropy))
    )
    if "knn_distance" in outputs_known:
        result["knn_distance"] = stats(outputs_known["knn_distance"])
    if "knn_predicted_class_distance" in outputs_known:
        values = np.asarray(outputs_known["knn_predicted_class_distance"], dtype=float)
        result["knn_predicted_class_distance"] = stats(values)
        if "labels" in outputs_known and "logits" in outputs_known:
            labels = np.asarray(outputs_known["labels"], dtype=int)
            predicted = np.asarray(outputs_known["logits"]).argmax(axis=1)
            num_classes = int(np.asarray(outputs_known["logits"]).shape[1])
            fallback = stats(values)
            means = np.full(num_classes, fallback["mean"], dtype=float)
            stds = np.full(num_classes, fallback["std"], dtype=float)
            counts = np.zeros(num_classes, dtype=np.int64)
            for class_index in range(num_classes):
                class_values = values[(labels == class_index) & (predicted == class_index)]
                counts[class_index] = len(class_values)
                if len(class_values) >= 2:
                    class_stats = stats(class_values)
                    means[class_index] = class_stats["mean"]
                    stds[class_index] = class_stats["std"]
            result["knn_predicted_class_distance_classwise"] = {
                "mean": means.tolist(),
                "std": stds.tolist(),
                "count": counts.tolist(),
                "fallback_mean": fallback["mean"],
                "fallback_std": fallback["std"],
            }
    if "knn_predicted_class_relative_distance" in outputs_known:
            result["knn_predicted_class_relative_distance"] = stats(
                outputs_known["knn_predicted_class_relative_distance"]
            )
    if "knn_min_class_distance" in outputs_known:
        result["knn_min_class_distance"] = stats(outputs_known["knn_min_class_distance"])
    if "knn_min_class_relative_distance" in outputs_known:
        result["knn_min_class_relative_distance"] = stats(
            outputs_known["knn_min_class_relative_distance"]
        )
    if "knn_predicted_class_support" in outputs_known:
        result["knn_predicted_class_support"] = stats(
            outputs_known["knn_predicted_class_support"]
        )
    if "vim_residual" in outputs_known:
        result["vim_residual"] = stats(outputs_known["vim_residual"])
    if prototypes is not None:
        _, proto_dist = compute_open_score(
            outputs_known,
            prototypes=prototypes,
            score_mode="proto_only",
        )
        result["proto_dist"] = stats(proto_dist)
    if gaussian_stats is not None and include_mahalanobis:
        result["mahalanobis"] = stats(compute_mahalanobis_distance(outputs_known["features"], gaussian_stats))
        result["mahalanobis_diag"] = stats(
            compute_mahalanobis_distance(outputs_known["features"], gaussian_stats, covariance="diag")
        )
        if "precision" in gaussian_stats:
            result["mahalanobis_shared"] = stats(
                compute_mahalanobis_distance(outputs_known["features"], gaussian_stats, covariance="shared")
            )
        if "precision" in gaussian_stats:
            result["relative_mahalanobis"] = stats(
                compute_relative_mahalanobis_distance(outputs_known["features"], gaussian_stats)
            )
        result["gaussian_nll"] = stats(compute_gaussian_nll(outputs_known["features"], gaussian_stats))
        if "labels" in outputs_known and "logits" in outputs_known:
            classwise_distance = compute_predicted_classwise_mahalanobis(
                outputs_known, gaussian_stats
            )
            labels = np.asarray(outputs_known["labels"], dtype=int)
            num_classes = int(np.asarray(gaussian_stats["means"]).shape[0])
            global_distance = stats(classwise_distance)
            means = np.full(num_classes, global_distance["mean"], dtype=float)
            stds = np.full(num_classes, global_distance["std"], dtype=float)
            counts = np.zeros(num_classes, dtype=np.int64)
            # Calibration uses the ground-truth known label. At test time the
            # corresponding predicted class is used, so no unknown labels are
            # involved in score construction.
            for class_index in range(num_classes):
                values = classwise_distance[labels == class_index]
                counts[class_index] = len(values)
                if len(values) >= 2:
                    class_stats = stats(values)
                    means[class_index] = class_stats["mean"]
                    stds[class_index] = class_stats["std"]
            result["classwise_mahalanobis"] = {
                "mean": means.tolist(),
                "std": stds.tolist(),
                "count": counts.tolist(),
                "fallback_mean": global_distance["mean"],
                "fallback_std": global_distance["std"],
            }
    if "unified_novel_mass" in outputs_known and "logits" in outputs_known:
        raw_score = np.asarray(outputs_known["unified_novel_mass"], dtype=float)
        result["unified_novel_mass"] = stats(raw_score)
        predicted_class = np.asarray(outputs_known["logits"]).argmax(axis=1)
        num_classes = int(np.asarray(outputs_known["logits"]).shape[1])
        global_stats = stats(raw_score)
        means = np.full(num_classes, global_stats["mean"], dtype=float)
        stds = np.full(num_classes, global_stats["std"], dtype=float)
        counts = np.zeros(num_classes, dtype=np.int64)
        for class_index in range(num_classes):
            values = raw_score[predicted_class == class_index]
            counts[class_index] = len(values)
            if len(values) >= 2:
                class_stats = stats(values)
                means[class_index] = class_stats["mean"]
                stds[class_index] = class_stats["std"]
        result["unified_novel_mass_classwise"] = {
            "mean": means.tolist(),
            "std": stds.tolist(),
            "count": counts.tolist(),
            "fallback_mean": global_stats["mean"],
            "fallback_std": global_stats["std"],
        }
    return result


def calibrate_threshold(scores_known: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(scores_known, percentile))


def calibrate_coverage_threshold(
    scores_known: np.ndarray,
    target_known_coverage: float = 0.95,
) -> tuple[float, dict]:
    """Choose a known-only threshold for a target known acceptance rate.

    Scores at or below the threshold are accepted as known.  The threshold is
    estimated only from known validation scores, so this policy can compare
    unknown rejection at a matched known-coverage operating point without
    using unknown test labels.
    """
    scores_known = np.asarray(scores_known, dtype=float)
    if scores_known.ndim != 1 or scores_known.size == 0:
        raise ValueError("scores_known must be a non-empty 1-D array")
    target = float(target_known_coverage)
    if not 0.0 < target <= 1.0:
        raise ValueError("target_known_coverage must be in (0, 1]")
    threshold = float(np.quantile(scores_known, target, method="linear"))
    known_acceptance = float(np.mean(scores_known <= threshold))
    return threshold, {
        "target_known_coverage": target,
        "known_accept_rate": known_acceptance,
    }


def calibrate_open_threshold(
    scores: np.ndarray,
    is_known: np.ndarray,
    objective: str = "balanced_accuracy",
    target_known_coverage: float = 0.95,
) -> tuple[float, dict]:
    """Choose an operating threshold on a labeled open-validation split.

    Lower scores are accepted as known. This is an explicit validation-only
    operating-point calibration and must not be used as a replacement for
    AUROC, which is threshold-independent.
    """
    if objective not in {"balanced_accuracy", "unknown_f1", "known_coverage"}:
        raise ValueError(f"Unsupported open threshold objective: {objective}")
    scores = np.asarray(scores, dtype=float)
    is_known = np.asarray(is_known, dtype=bool)
    if scores.ndim != 1 or scores.shape != is_known.shape:
        raise ValueError("scores and is_known must be matching 1-D arrays")
    if not np.any(is_known) or np.all(is_known):
        raise ValueError("Open threshold calibration requires known and unknown samples")
    target = float(target_known_coverage)
    if not 0.0 < target <= 1.0:
        raise ValueError("target_known_coverage must be in (0, 1]")
    candidates = np.unique(scores)
    candidates = np.concatenate(
        [np.array([np.nextafter(candidates[0], -np.inf)]), candidates, np.array([np.nextafter(candidates[-1], np.inf)])]
    )
    best = None
    for threshold in candidates:
        pred_known = scores <= threshold
        known_accept = float(np.mean(pred_known[is_known]))
        unknown_reject = float(np.mean(~pred_known[~is_known]))
        if objective == "known_coverage":
            if known_accept + 1e-12 < target:
                continue
            value = unknown_reject
        elif objective == "balanced_accuracy":
            value = 0.5 * (known_accept + unknown_reject)
        else:
            predicted_unknown = ~pred_known
            true_unknown = ~is_known
            tp = float(np.sum(predicted_unknown & true_unknown))
            precision = tp / max(float(predicted_unknown.sum()), 1.0)
            recall = tp / max(float(true_unknown.sum()), 1.0)
            value = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if objective == "known_coverage":
            candidate = (value, -known_accept, -float(threshold))
        else:
            candidate = (value, known_accept, unknown_reject, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if objective == "known_coverage":
        threshold = -best[2]
        pred_known = scores <= threshold
        known_accept = float(np.mean(pred_known[is_known]))
        unknown_reject = float(np.mean(~pred_known[~is_known]))
    else:
        _, known_accept, unknown_reject, threshold = best
    return threshold, {
        "objective": objective,
        "known_accept_rate": known_accept,
        "unknown_reject_rate": unknown_reject,
        "target_known_coverage": target if objective == "known_coverage" else None,
    }


def calibrate_class_thresholds(
    scores_known: np.ndarray,
    predicted_classes: np.ndarray,
    num_classes: int,
    percentile: float = 95.0,
    min_samples: int = 5,
) -> np.ndarray:
    """Calibrate known-only thresholds separately for predicted classes.

    Classes with too few validation predictions use the global threshold. The
    calibration never consumes unknown labels, so it remains valid for open
    set evaluation.
    """
    scores_known = np.asarray(scores_known, dtype=float)
    predicted_classes = np.asarray(predicted_classes, dtype=int)
    if scores_known.ndim != 1 or predicted_classes.shape != scores_known.shape:
        raise ValueError("scores_known and predicted_classes must be matching 1-D arrays")
    global_threshold = calibrate_threshold(scores_known, percentile)
    thresholds = np.full(int(num_classes), global_threshold, dtype=float)
    required_samples = max(int(min_samples), 1)
    for class_id in range(int(num_classes)):
        values = scores_known[predicted_classes == class_id]
        if len(values) >= required_samples:
            thresholds[class_id] = np.percentile(values, percentile)
    return thresholds


def _rank_normalize_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.ones_like(values)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return (order + 1.0) / len(values)


def candidate_purification_score(
    outputs: Dict[str, np.ndarray],
    mode: str = "none",
    open_score: np.ndarray | None = None,
) -> np.ndarray:
    """Build a label-free ranking used only to clean the cluster candidate pool."""
    if mode == "none":
        return np.zeros(len(outputs["entropy"]), dtype=float)
    if mode == "open_score":
        if open_score is None:
            raise ValueError("open_score purification requires an open-set score")
        return _rank_normalize_numpy(open_score)
    if mode == "entropy":
        return _rank_normalize_numpy(outputs["entropy"])
    if mode == "head_uncertainty":
        return _rank_normalize_numpy(outputs["head_uncertainty"])
    if mode == "uncertainty_consensus":
        components = [
            _rank_normalize_numpy(outputs["entropy"]),
            _rank_normalize_numpy(outputs["epistemic"]),
            _rank_normalize_numpy(outputs["head_uncertainty"]),
            _rank_normalize_numpy(1.0 - outputs["probs"].max(axis=1)),
        ]
        return np.mean(components, axis=0)
    if mode == "knn_support":
        if "knn_class_support" not in outputs:
            raise ValueError("knn_support purification requires labeled KNN feature-bank outputs")
        # High class-consensus among known neighbors is evidence that a
        # rejected sample may be a known-class false rejection.
        return 1.0 - np.asarray(outputs["knn_class_support"], dtype=float)
    if mode == "open_knn_support":
        if open_score is None or "knn_class_support" not in outputs:
            raise ValueError("open_knn_support requires open_score and labeled KNN outputs")
        return 0.5 * (
            _rank_normalize_numpy(open_score)
            + _rank_normalize_numpy(1.0 - outputs["knn_class_support"])
        )
    raise ValueError(f"Unsupported candidate purification mode: {mode}")


def evaluate_candidate_purification(
    outputs: Dict[str, np.ndarray],
    candidate_mask: np.ndarray,
    purified_mask: np.ndarray,
) -> dict:
    """Measure candidate-pool quality using known/unknown identity labels.

    Intended for the reserved open-validation split only; test labels must not
    be used to select purification rules or keep ratios.
    """
    known = np.asarray(outputs["is_known"], dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    purified_mask = np.asarray(purified_mask, dtype=bool)

    def summarize(mask):
        count = int(mask.sum())
        unknown_count = int(np.sum(mask & ~known))
        return {
            "count": count,
            "unknown_precision": float(unknown_count / max(count, 1)),
            "unknown_recall": float(unknown_count / max(int((~known).sum()), 1)),
            "known_contamination": float(np.sum(mask & known) / max(count, 1)),
        }

    return {"before": summarize(candidate_mask), "after": summarize(purified_mask)}


def purify_candidate_mask(
    outputs: Dict[str, np.ndarray],
    candidate_mask: np.ndarray,
    mode: str = "none",
    keep_ratio: float = 1.0,
    open_score: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Keep the highest-ranked rejected samples for clustering only."""
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    before_count = int(candidate_mask.sum())
    if mode == "none" or before_count <= 1:
        return candidate_mask, {
            "mode": mode,
            "keep_ratio": 1.0,
            "before_count": before_count,
            "after_count": before_count,
            "removed_count": 0,
        }
    ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    keep_count = min(before_count, max(2, int(np.ceil(before_count * ratio))))
    scores = candidate_purification_score(outputs, mode, open_score=open_score)
    indices = np.flatnonzero(candidate_mask)
    order = np.argsort(scores[indices], kind="stable")[::-1]
    purified = np.zeros_like(candidate_mask)
    purified[indices[order[:keep_count]]] = True
    return purified, {
        "mode": mode,
        "keep_ratio": float(keep_count / max(before_count, 1)),
        "before_count": before_count,
        "after_count": int(purified.sum()),
        "removed_count": before_count - int(purified.sum()),
    }


def prepare_cluster_features(
    features: np.ndarray,
    use_pca: bool = False,
    pca_dim: int = 32,
    whiten: bool = True,
) -> np.ndarray:
    """Prepare features for clustering without using unknown labels."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        return values
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    if use_pca and len(values) > 2 and values.shape[1] > 1:
        n_components = min(int(pca_dim), values.shape[1], len(values) - 1)
        if n_components >= 2:
            values = PCA(n_components=n_components, whiten=whiten, random_state=42).fit_transform(values)
            values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def cluster_unknown_samples(
    features: np.ndarray,
    num_clusters: int,
    method: str = "kmeans",
    n_init: int = 10,
    random_state: int = 42,
):
    """Cluster candidate unknown samples with a selectable label-free method."""
    values = np.asarray(features, dtype=np.float32)
    num_clusters = int(max(1, min(num_clusters, len(values))))
    if num_clusters == 1 or len(values) <= 1:
        return np.zeros(len(values), dtype=np.int64)
    if method == "kmeans":
        return KMeans(
            n_clusters=num_clusters,
            n_init=max(1, int(n_init)),
            random_state=random_state,
        ).fit_predict(values)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=num_clusters, linkage="ward").fit_predict(values)
    if method == "spectral":
        neighbors = max(1, min(10, len(values) - 1))
        return SpectralClustering(
            n_clusters=num_clusters,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            assign_labels="kmeans",
            n_init=max(1, int(n_init)),
            random_state=random_state,
        ).fit_predict(values)
    raise ValueError(f"Unsupported cluster method: {method}")


def _scale_metric(values: list[float], higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not higher_is_better:
        array = -array
    if np.ptp(array) < 1e-12:
        return np.ones_like(array)
    return (array - array.min()) / (array.max() - array.min())


def evaluate_cluster_candidates(
    features: np.ndarray,
    max_clusters: int,
    method: str = "kmeans",
    selection: str = "silhouette",
    n_init: int = 10,
    stability_repeats: int = 5,
    random_state: int = 42,
) -> tuple[int, list[dict]]:
    """Select K using only candidate-pool geometry and report diagnostics."""
    n = len(features)
    if n < 4:
        return max(1, n), []
    # Respect the configured search bound. A hidden cap of 20 silently made
    # larger novel-class counts impossible to recover.
    upper = min(int(max_clusters), n - 1)
    if upper < 2:
        return 1, []
    values = np.asarray(features, dtype=np.float32)
    eval_values = values
    eval_indices = np.arange(n)
    if n > 2000:
        rng = np.random.default_rng(random_state)
        eval_indices = rng.choice(n, 2000, replace=False)
        eval_values = values[eval_indices]

    rows: list[dict] = []
    for k in range(2, upper + 1):
        labels = cluster_unknown_samples(values, k, method, n_init, random_state)
        eval_labels = labels[eval_indices]
        if len(np.unique(eval_labels)) < 2:
            continue
        row = {
            "k": int(k),
            "silhouette": float(silhouette_score(eval_values, eval_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(eval_values, eval_labels)),
            "davies_bouldin": float(davies_bouldin_score(eval_values, eval_labels)),
        }
        if selection == "gmm_bic":
            mixture = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                reg_covar=1e-4,
                n_init=1,
                random_state=random_state,
            ).fit(eval_values)
            row["gmm_bic"] = float(mixture.bic(eval_values))
        if selection == "stability":
            repeat_labels = []
            for repeat in range(max(2, int(stability_repeats))):
                repeat_labels.append(
                    cluster_unknown_samples(
                        values, k, method, n_init, random_state + repeat + 1
                    )[eval_indices]
                )
            stability_values = [
                normalized_mutual_info_score(repeat_labels[i], repeat_labels[j])
                for i in range(len(repeat_labels))
                for j in range(i + 1, len(repeat_labels))
            ]
            row["stability_nmi"] = float(np.mean(stability_values)) if stability_values else 1.0
        rows.append(row)

    if not rows:
        return 1, []
    silhouette = _scale_metric([row["silhouette"] for row in rows], True)
    ch = _scale_metric([row["calinski_harabasz"] for row in rows], True)
    db = _scale_metric([row["davies_bouldin"] for row in rows], False)
    internal = 0.5 * silhouette + 0.25 * ch + 0.25 * db
    if selection == "silhouette":
        combined = silhouette
    elif selection == "stability":
        stability = np.asarray([row["stability_nmi"] for row in rows], dtype=float)
        combined = 0.5 * internal + 0.5 * stability
    elif selection == "composite":
        combined = internal
    elif selection == "gmm_bic":
        combined = _scale_metric(
            [row["gmm_bic"] for row in rows], higher_is_better=False
        )
    else:
        raise ValueError(f"Unsupported cluster selection: {selection}")
    for row, internal_value, value in zip(rows, internal, combined):
        row["internal_score"] = float(internal_value)
        row["selection_score"] = float(value)
    best_index = int(np.argmax(combined))
    return int(rows[best_index]["k"]), rows


def estimate_num_clusters(
    features: np.ndarray,
    max_clusters: int,
    method: str = "kmeans",
    selection: str = "silhouette",
    n_init: int = 10,
    stability_repeats: int = 5,
):
    """Backward-compatible K estimator; diagnostics are available separately."""
    selected, _ = evaluate_cluster_candidates(
        features, max_clusters, method, selection, n_init, stability_repeats
    )
    return selected


def run_discovery(
    outputs: Dict[str, np.ndarray],
    threshold: float,
    num_novel: int,
    prototypes: np.ndarray | None = None,
    score_mode: str = "full",
    normalization: Dict[str, Dict[str, float]] | None = None,
    gaussian_stats: Dict[str, np.ndarray] | None = None,
    cluster_k: int | str = "oracle",
    cluster_method: str = "kmeans",
    cluster_feature: str = "projection",
    cluster_selection: str = "silhouette",
    cluster_pca_dim: int = 32,
    cluster_normalize: bool = False,
    cluster_whiten: bool = True,
    cluster_n_init: int = 10,
    cluster_stability_repeats: int = 5,
    max_auto_clusters: int = 50,
    candidate_purify: str = "none",
    candidate_keep_ratio: float = 1.0,
    enable_clustering: bool = True,
):
    entropy = outputs["entropy"]
    epistemic = outputs["epistemic"]
    aleatoric = outputs["aleatoric"]
    score, proto_dist = compute_open_score(
        outputs,
        prototypes=prototypes,
        score_mode=score_mode,
        normalization=normalization,
        gaussian_stats=gaussian_stats,
    )
    proto_dist = np.zeros_like(score) if proto_dist is None else proto_dist
    mahalanobis_score_modes = {
        "mahalanobis",
        "mahalanobis_diag",
        "mahalanobis_shared",
        "entropy_mahalanobis",
        "entropy_mahalanobis_diag",
        "entropy_mahalanobis_shared",
        "normalized_entropy_mahalanobis",
        "normalized_entropy_mahalanobis_diag",
        "normalized_entropy_mahalanobis_shared",
        "relative_mahalanobis",
        "normalized_entropy_relative_mahalanobis",
    }
    if gaussian_stats is not None and score_mode in mahalanobis_score_modes:
        if score_mode in {"relative_mahalanobis", "normalized_entropy_relative_mahalanobis"}:
            mahalanobis = compute_relative_mahalanobis_distance(outputs["features"], gaussian_stats)
        else:
            mahalanobis = compute_mahalanobis_distance(outputs["features"], gaussian_stats)
    else:
        mahalanobis = np.zeros_like(score)
    known_mask = outputs["is_known"].astype(bool)
    open_labels = (~known_mask).astype(int)
    auroc = compute_auroc(open_labels, score)
    aupr = compute_aupr(open_labels, score)
    fpr95 = compute_fpr95(open_labels, score)
    pred_class = outputs["logits"].argmax(axis=1)
    threshold_values = np.asarray(threshold)
    if threshold_values.ndim == 0:
        sample_threshold = float(threshold_values)
        threshold_type = "global"
    elif threshold_values.ndim == 1 and len(threshold_values) > 0:
        sample_threshold = threshold_values[np.clip(pred_class, 0, len(threshold_values) - 1)]
        threshold_type = "class_conditional"
    else:
        raise ValueError("threshold must be a scalar or non-empty class threshold vector")
    pred_known = score <= sample_threshold
    open_confusion = open_set_confusion(known_mask, pred_known)
    true_labels = outputs["labels"]
    class_correct = pred_class == true_labels
    oscr = compute_oscr(known_mask, pred_known, class_correct, score)
    known_class_correct = int(np.sum(known_mask & pred_known & (pred_class == true_labels)))
    known_class_wrong = int(np.sum(known_mask & pred_known & (pred_class != true_labels)))
    known_total = int(known_mask.sum())
    result = {
        "auroc": auroc,
        "aupr": aupr,
        "fpr95": fpr95,
        "oscr": oscr,
        "known_ratio": float(pred_known.mean()),
        "known_class_correct": known_class_correct,
        "known_class_wrong": known_class_wrong,
        "known_class_accuracy_after_accept": float(known_class_correct / max(known_class_correct + known_class_wrong, 1)),
        "known_class_accuracy_all_known": float(known_class_correct / max(known_total, 1)),
        "threshold_type": threshold_type,
    }
    result.update(open_confusion)
    if not enable_clustering:
        result.update(
            {
                "cluster_k": None,
                "cluster_method": cluster_method,
                "cluster_feature": cluster_feature,
                "cluster_selection": "skipped",
                "cluster_pca_dim": int(cluster_pca_dim),
                "cluster_normalized": bool(cluster_normalize or cluster_feature.endswith("_pca")),
                "cluster_n_init": int(cluster_n_init),
                "cluster_stability_repeats": int(cluster_stability_repeats),
                "cluster_diagnostics": [],
                "candidate_purify": candidate_purify,
                "candidate_keep_ratio": float(candidate_keep_ratio),
                "candidate_purification": {
                    "mode": candidate_purify,
                    "keep_ratio": 1.0,
                    "before_count": 0,
                    "after_count": 0,
                    "removed_count": 0,
                },
                "clustering_skipped": True,
            }
        )
        detail = {
            "score": score,
            "score_mode": score_mode,
            "feature_norm": np.asarray(
                outputs.get("feature_norm", np.zeros_like(score)), dtype=float
            ),
            "entropy": entropy,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "expected_entropy": outputs.get("expected_entropy", aleatoric),
            "head_uncertainty": outputs.get("head_uncertainty", aleatoric),
            "proto_dist": proto_dist,
            "mahalanobis": mahalanobis,
            "odin_msp": np.asarray(outputs.get("odin_msp", np.zeros_like(score)), dtype=float),
            "react_energy": np.asarray(outputs.get("react_energy", np.zeros_like(score)), dtype=float),
            "knn_distance": np.asarray(outputs.get("knn_distance", np.zeros_like(score)), dtype=float),
            "knn_min_class_distance": np.asarray(
                outputs.get("knn_min_class_distance", np.zeros_like(score)), dtype=float
            ),
            "knn_min_class_relative_distance": np.asarray(
                outputs.get("knn_min_class_relative_distance", np.zeros_like(score)), dtype=float
            ),
            "vim_residual": np.asarray(outputs.get("vim_residual", np.zeros_like(score)), dtype=float),
            "pred_known": pred_known,
            "true_known": known_mask,
            "pred_class": pred_class,
            "true_label": true_labels,
            "raw_labels": outputs["raw_labels"],
            "pred_cluster": None,
            "cluster_k": None,
            "cluster_method": cluster_method,
            "cluster_feature": cluster_feature,
            "cluster_selection": "skipped",
            "cluster_diagnostics": [],
            "candidate_purify": candidate_purify,
            "candidate_keep_ratio": float(candidate_keep_ratio),
            "candidate_purification": {
                "mode": candidate_purify,
                "keep_ratio": float(candidate_keep_ratio),
                "before_count": int((~pred_known).sum()),
                "after_count": int((~pred_known).sum()),
                "removed_count": 0,
            },
        }
        return result, score, pred_known, detail
    novel_mask = ~pred_known
    cluster_mask, purification = purify_candidate_mask(
        outputs,
        novel_mask,
        mode=candidate_purify,
        keep_ratio=candidate_keep_ratio,
        open_score=score,
    )
    selected_cluster_k = None
    selected_cluster_predictions = None
    cluster_diagnostics = []
    true_unknown_k = int(np.unique(outputs["raw_labels"][~known_mask]).size)
    if cluster_feature in {"projection", "projection_pca"}:
        raw_cluster_features = outputs.get("projections", outputs["features"])
    elif cluster_feature in {"feature", "feature_pca"}:
        raw_cluster_features = outputs["features"]
    elif cluster_feature in {"novel", "novel_pca"}:
        if "novel_probs" not in outputs:
            raise ValueError("novel clustering features require --novel-head-ckpt")
        raw_cluster_features = outputs["novel_probs"]
    elif cluster_feature in {"unified", "unified_pca"}:
        if "unified_probs" not in outputs:
            raise ValueError("unified clustering features require --novel-head-ckpt")
        raw_cluster_features = outputs["unified_probs"]
    else:
        raise ValueError(f"Unsupported cluster feature: {cluster_feature}")
    use_pca = cluster_feature.endswith("_pca")

    def prepare_subset(mask):
        values = np.asarray(raw_cluster_features[mask], dtype=np.float32)
        if len(values) == 0:
            return values
        if cluster_normalize or use_pca:
            return prepare_cluster_features(
                values,
                use_pca=use_pca,
                pca_dim=cluster_pca_dim,
                whiten=cluster_whiten,
            )
        return values

    def add_oracle_report(prefix, mask):
        labels = np.asarray(outputs["raw_labels"])[mask]
        if len(labels) < 2:
            return
        oracle_k = int(np.unique(labels).size)
        if oracle_k < 1:
            return
        features = prepare_subset(mask)
        predictions = cluster_unknown_samples(
            features,
            num_clusters=min(oracle_k, len(labels)),
            method=cluster_method,
            n_init=cluster_n_init,
        )
        result.update(
            {f"{prefix}_{key}": value for key, value in clustering_report(labels, predictions).items()}
        )

    add_oracle_report("cluster_all_unknown_oracle", ~known_mask)
    add_oracle_report("cluster_candidate_unknown_oracle", novel_mask & ~known_mask)
    if cluster_mask.sum() > 1 and num_novel > 0:
        raw_novel_features = np.asarray(raw_cluster_features[cluster_mask], dtype=np.float32)
        novel_features = (
            prepare_cluster_features(
                raw_novel_features,
                use_pca=use_pca,
                pca_dim=cluster_pca_dim,
                whiten=cluster_whiten,
            )
            if cluster_normalize or use_pca
            else raw_novel_features
        )
        novel_true = outputs["raw_labels"][cluster_mask]
        selected_cluster_k = (
            min(num_novel, len(novel_features))
            if cluster_k == "oracle"
            else None
        )
        if cluster_k != "oracle":
            selected_cluster_k, cluster_diagnostics = evaluate_cluster_candidates(
                novel_features,
                max_clusters=max_auto_clusters,
                method=cluster_method,
                selection=cluster_selection,
                n_init=cluster_n_init,
                stability_repeats=cluster_stability_repeats,
            )
        novel_pred = cluster_unknown_samples(
            novel_features,
            num_clusters=selected_cluster_k,
            method=cluster_method,
            n_init=cluster_n_init,
        )
        # Reuse this fit when serializing per-sample cluster assignments below.
        selected_cluster_predictions = novel_pred
        if len(np.unique(novel_pred)) > 0:
            result.update({f"cluster_{k}": v for k, v in clustering_report(novel_true, novel_pred).items()})
            true_unknown_candidates = (~known_mask)[cluster_mask]
            if true_unknown_candidates.sum() > 1:
                result.update(
                    {
                        f"cluster_unknown_only_{k}": v
                        for k, v in clustering_report(
                            novel_true[true_unknown_candidates],
                            novel_pred[true_unknown_candidates],
                        ).items()
                    }
                )
            result["cluster_candidate_count"] = int(len(novel_true))
            result["cluster_candidate_count_before_purification"] = purification["before_count"]
            result["cluster_candidate_removed_count"] = purification["removed_count"]
            true_unknown_before = int((~known_mask & novel_mask).sum())
            result["cluster_candidate_true_unknown_count_before_purification"] = true_unknown_before
            result["cluster_candidate_false_reject_count_before_purification"] = int(
                (known_mask & novel_mask).sum()
            )
            result["cluster_candidate_purity_before_purification"] = float(
                true_unknown_before / max(purification["before_count"], 1)
            )
            result["cluster_candidate_unknown_recall"] = float(
                true_unknown_candidates.sum() / max(true_unknown_before, 1)
            )
            result["cluster_true_unknown_count"] = int(true_unknown_candidates.sum())
            result["cluster_false_reject_count"] = int((~true_unknown_candidates).sum())
            result["cluster_candidate_purity"] = float(
                true_unknown_candidates.sum() / max(len(novel_true), 1)
            )
            result["cluster_true_k"] = true_unknown_k
            result["cluster_k_abs_error"] = abs(int(selected_cluster_k) - true_unknown_k)
    result.update(
        {
            "cluster_k": None if selected_cluster_k is None else int(selected_cluster_k),
            "cluster_method": cluster_method,
            "cluster_feature": cluster_feature,
            "cluster_selection": cluster_selection if cluster_k != "oracle" else "oracle",
            "cluster_pca_dim": int(cluster_pca_dim),
            "cluster_normalized": bool(cluster_normalize or cluster_feature.endswith("_pca")),
            "cluster_n_init": int(cluster_n_init),
            "cluster_stability_repeats": int(cluster_stability_repeats),
            "max_auto_clusters": int(max_auto_clusters),
            "cluster_diagnostics": cluster_diagnostics,
            "candidate_purify": candidate_purify,
            "candidate_keep_ratio": float(candidate_keep_ratio),
            "candidate_purification": purification,
        }
    )
    detail = {
        "score": score,
        "score_mode": score_mode,
        "feature_norm": np.asarray(
            outputs.get("feature_norm", np.zeros_like(score)), dtype=float
        ),
        "entropy": entropy,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "expected_entropy": outputs.get("expected_entropy", aleatoric),
        "head_uncertainty": outputs.get("head_uncertainty", aleatoric),
        "proto_dist": proto_dist,
        "mahalanobis": mahalanobis,
        "odin_msp": np.asarray(outputs.get("odin_msp", np.zeros_like(score)), dtype=float),
        "react_energy": np.asarray(outputs.get("react_energy", np.zeros_like(score)), dtype=float),
        "knn_distance": np.asarray(outputs.get("knn_distance", np.zeros_like(score)), dtype=float),
        "knn_min_class_distance": np.asarray(
            outputs.get("knn_min_class_distance", np.zeros_like(score)), dtype=float
        ),
        "knn_min_class_relative_distance": np.asarray(
            outputs.get("knn_min_class_relative_distance", np.zeros_like(score)), dtype=float
        ),
        "vim_residual": np.asarray(outputs.get("vim_residual", np.zeros_like(score)), dtype=float),
        "pred_known": pred_known,
        "true_known": known_mask,
        "pred_class": pred_class,
        "true_label": true_labels,
        "raw_labels": outputs["raw_labels"],
        "pred_cluster": None,
        "cluster_k": selected_cluster_k,
        "cluster_method": cluster_method,
        "cluster_feature": cluster_feature,
        "cluster_selection": cluster_selection,
        "cluster_diagnostics": cluster_diagnostics,
        "candidate_purify": candidate_purify,
        "candidate_keep_ratio": float(candidate_keep_ratio),
        "candidate_purification": purification,
    }
    if cluster_mask.sum() > 1 and num_novel > 0:
        pred_cluster = np.full(len(score), -1, dtype=np.int64)
        if selected_cluster_predictions is None:
            # Defensive fallback for an unusual short-candidate path.
            cluster_features = prepare_subset(cluster_mask)
            selected_cluster_predictions = cluster_unknown_samples(
                cluster_features,
                num_clusters=selected_cluster_k,
                method=cluster_method,
                n_init=cluster_n_init,
            )
        pred_cluster[cluster_mask] = selected_cluster_predictions
        detail["pred_cluster"] = pred_cluster
    return result, score, pred_known, detail
