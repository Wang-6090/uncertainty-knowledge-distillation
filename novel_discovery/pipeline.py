from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from .losses import (
    classification_loss,
    discovery_unknown_loss,
    discovery_view_loss,
    distillation_loss,
    energy_margin_loss,
    feature_distillation_loss,
    pseudo_unknown_loss,
    proxy_contrastive_loss,
    prototype_alignment_loss,
    supervised_contrastive_loss,
    uncertainty_alignment_loss,
    weighted_energy_margin_loss,
)
from .joint_discovery import joint_discovery_loss
from .metrics import (
    clustering_report,
    compute_aupr,
    compute_auroc,
    compute_fpr95,
    compute_oscr,
    open_set_confusion,
)
from .utils import AverageMeter


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


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
        if mode == "entropy":
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
    mask = select_discovery_candidates(logits, uncertainty, ratio=ratio, mode=mode)
    weights = torch.zeros_like(logits[:, 0], dtype=torch.float32)
    if not mask.any():
        return mask, weights, 0.0
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
        )
        agreement = (student_mask == mask).float()
        candidate_weight = candidate_weight * (0.5 + 0.5 * agreement)
    weights[mask] = candidate_weight[mask].clamp(0.0, 1.0)
    selected_agreement = neighbor_agreement[mask].mean().item() if mask.any() else 0.0
    return mask, weights, selected_agreement


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
    alpha_proxy: float = 0.0,
    alpha_pseudo: float = 0.0,
    alpha_energy: float = 0.0,
    pseudo_mode: str = "strong",
    pseudo_feature_noise: float = 0.05,
    uncertainty_target_mode: str = "confidence",
    energy_margin: float = 1.0,
    energy_temperature: float = 1.0,
    proxy_temperature: float = 0.1,
):
    model.train()
    ce_meter = AverageMeter()
    unc_meter = AverageMeter()
    proto_meter = AverageMeter()
    proxy_meter = AverageMeter()
    pseudo_meter = AverageMeter()
    energy_meter = AverageMeter()
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
        loss = (
            loss_ce
            + alpha_unc * loss_unc
            + alpha_proto * loss_proto
            + alpha_proxy * loss_proxy
            + alpha_pseudo * loss_pseudo
            + alpha_energy * loss_energy
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_meter.update(loss_ce.item(), images.size(0))
        unc_meter.update(loss_unc.item(), images.size(0))
        proto_meter.update(loss_proto.item(), images.size(0))
        proxy_meter.update(loss_proxy.item(), images.size(0))
        pseudo_meter.update(loss_pseudo.item(), images.size(0))
        energy_meter.update(loss_energy.item(), images.size(0))
    return {
        "ce": ce_meter.avg,
        "unc": unc_meter.avg,
        "proto": proto_meter.avg,
        "proxy": proxy_meter.avg,
        "pseudo": pseudo_meter.avg,
        "energy": energy_meter.avg,
    }


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
    alpha_proto: float = 0.0,
    alpha_proxy: float = 0.0,
    alpha_pseudo: float = 0.0,
    alpha_energy: float = 0.0,
    pseudo_mode: str = "strong",
    pseudo_feature_noise: float = 0.05,
    uncertainty_target_mode: str = "confidence",
    temperature: float = 2.0,
    kd_mode: str = "uncertainty",
    uncertainty_weight_mode: str = "raw",
    discovery_loader=None,
    alpha_discovery: float = 0.0,
    alpha_discovery_unknown: float = 0.0,
    alpha_discovery_energy: float = 0.0,
    alpha_discovery_selective_unknown: float = 0.0,
    alpha_discovery_selective_energy: float = 0.0,
    discovery_select_ratio: float = 0.25,
    discovery_select_mode: str = "entropy_uncertainty",
    discovery_loss_mode: str = "nt_xent",
    discovery_temperature: float = 0.2,
    energy_margin: float = 1.0,
    energy_temperature: float = 1.0,
    proxy_temperature: float = 0.1,
    discovery_selection_model=None,
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
    alpha_joint_neighbor: float = 0.1,
    joint_neighbor_k: int = 5,
):
    student.train()
    teacher.eval()
    ce_meter = AverageMeter()
    kd_meter = AverageMeter()
    feat_kd_meter = AverageMeter()
    unc_meter = AverageMeter()
    sc_meter = AverageMeter()
    proto_meter = AverageMeter()
    proxy_meter = AverageMeter()
    pseudo_meter = AverageMeter()
    energy_meter = AverageMeter()
    discovery_meter = AverageMeter()
    discovery_unknown_meter = AverageMeter()
    discovery_energy_meter = AverageMeter()
    discovery_selective_unknown_meter = AverageMeter()
    discovery_selective_energy_meter = AverageMeter()
    discovery_selected_meter = AverageMeter()
    discovery_raw_selected_meter = AverageMeter()
    discovery_neighbor_agreement_meter = AverageMeter()
    discovery_weight_mean_meter = AverageMeter()
    joint_meter = AverageMeter()
    joint_consistency_meter = AverageMeter()
    joint_balance_meter = AverageMeter()
    joint_neighbor_meter = AverageMeter()
    discovery_iter = iter(discovery_loader) if discovery_loader is not None else None
    for batch in tqdm(loader, desc="student-train", leave=False):
        images, labels, raw_labels, is_known, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            t_out = teacher(images)
        s_out = student(images)
        loss_ce = classification_loss(s_out["logits"], labels)
        loss_unc = uncertainty_alignment_loss(
            s_out["uncertainty"], s_out["logits"], labels, target_mode=uncertainty_target_mode
        )
        loss_kd = distillation_loss(
            s_out["logits"],
            t_out["logits"],
            teacher_uncertainty=t_out["uncertainty"],
            temperature=temperature,
            uncertainty_weighted=(kd_mode == "uncertainty"),
            uncertainty_weight_mode=uncertainty_weight_mode,
        )
        feature_teacher_uncertainty = t_out["uncertainty"] if kd_mode == "uncertainty" else None
        loss_feat_kd = feature_distillation_loss(
            s_out["proj"],
            t_out["proj"],
            teacher_uncertainty=feature_teacher_uncertainty,
            uncertainty_weight_mode=uncertainty_weight_mode,
        )
        loss_supcon = supervised_contrastive_loss(s_out["proj"], labels)
        loss_proto = prototype_alignment_loss(s_out["features"], labels, student.classifier.weight)
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
        loss_discovery = s_out["logits"].new_tensor(0.0)
        loss_discovery_unknown = s_out["logits"].new_tensor(0.0)
        loss_discovery_energy = s_out["logits"].new_tensor(0.0)
        loss_discovery_selective_unknown = s_out["logits"].new_tensor(0.0)
        loss_discovery_selective_energy = s_out["logits"].new_tensor(0.0)
        loss_joint_discovery = s_out["logits"].new_tensor(0.0)
        joint_consistency = s_out["logits"].new_tensor(0.0)
        joint_balance = s_out["logits"].new_tensor(0.0)
        joint_neighbor = s_out["logits"].new_tensor(0.0)
        discovery_selected_ratio = 0.0
        discovery_raw_selected_ratio = 0.0
        discovery_neighbor_agreement = 0.0
        discovery_weight_mean = 0.0
        if discovery_iter is not None and (
            alpha_discovery > 0.0
            or alpha_discovery_unknown > 0.0
            or alpha_discovery_energy > 0.0
            or alpha_discovery_selective_unknown > 0.0
            or alpha_discovery_selective_energy > 0.0
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
            if novel_head is not None and alpha_joint_discovery > 0.0:
                first_novel_logits = novel_head(first_out["features"])
                second_novel_logits = novel_head(second_out["features"])
                joint_losses = joint_discovery_loss(
                    first_out["features"],
                    second_out["features"],
                    first_novel_logits,
                    second_novel_logits,
                    confidence_threshold=joint_confidence_threshold,
                    assignment_temperature=joint_assignment_temperature,
                    alpha_consistency=alpha_joint_consistency,
                    alpha_balance=alpha_joint_balance,
                    alpha_neighbor=alpha_joint_neighbor,
                    neighbor_k=joint_neighbor_k,
                )
                loss_joint_discovery = joint_losses["total"]
                joint_consistency = joint_losses["consistency"]
                joint_balance = joint_losses["balance"]
                joint_neighbor = joint_losses["neighbor"]
            loss_discovery = discovery_view_loss(
                first_out["proj"],
                second_out["proj"],
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
                )
                second_mask = select_discovery_candidates(
                    second_selection_out["logits"],
                    second_selection_out["uncertainty"],
                    ratio=discovery_select_ratio,
                    mode=discovery_select_mode,
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
            + alpha_proto * loss_proto
            + alpha_proxy * loss_proxy
            + alpha_pseudo * loss_pseudo
            + alpha_energy * loss_energy
            + alpha_discovery * loss_discovery
            + alpha_discovery_unknown * loss_discovery_unknown
            + alpha_discovery_energy * loss_discovery_energy
            + alpha_discovery_selective_unknown * loss_discovery_selective_unknown
            + alpha_discovery_selective_energy * loss_discovery_selective_energy
            + alpha_joint_discovery * loss_joint_discovery
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if discovery_selection_model is not None:
            update_ema_model(discovery_selection_model, student, decay=discovery_ema_decay)
        ce_meter.update(loss_ce.item(), images.size(0))
        kd_meter.update(loss_kd.item(), images.size(0))
        feat_kd_meter.update(loss_feat_kd.item(), images.size(0))
        unc_meter.update(loss_unc.item(), images.size(0))
        sc_meter.update(loss_supcon.item(), images.size(0))
        proto_meter.update(loss_proto.item(), images.size(0))
        proxy_meter.update(loss_proxy.item(), images.size(0))
        pseudo_meter.update(loss_pseudo.item(), images.size(0))
        energy_meter.update(loss_energy.item(), images.size(0))
        discovery_meter.update(loss_discovery.item(), images.size(0))
        discovery_unknown_meter.update(loss_discovery_unknown.item(), images.size(0))
        discovery_energy_meter.update(loss_discovery_energy.item(), images.size(0))
        discovery_selective_unknown_meter.update(loss_discovery_selective_unknown.item(), images.size(0))
        discovery_selective_energy_meter.update(loss_discovery_selective_energy.item(), images.size(0))
        discovery_selected_meter.update(discovery_selected_ratio, images.size(0))
        discovery_raw_selected_meter.update(discovery_raw_selected_ratio, images.size(0))
        discovery_neighbor_agreement_meter.update(discovery_neighbor_agreement, images.size(0))
        discovery_weight_mean_meter.update(discovery_weight_mean, images.size(0))
        joint_meter.update(loss_joint_discovery.item(), images.size(0))
        joint_consistency_meter.update(joint_consistency.item(), images.size(0))
        joint_balance_meter.update(joint_balance.item(), images.size(0))
        joint_neighbor_meter.update(joint_neighbor.item(), images.size(0))
    return {
        "ce": ce_meter.avg,
        "kd": kd_meter.avg,
        "feat_kd": feat_kd_meter.avg,
        "unc": unc_meter.avg,
        "supcon": sc_meter.avg,
        "proto": proto_meter.avg,
        "proxy": proxy_meter.avg,
        "pseudo": pseudo_meter.avg,
        "energy": energy_meter.avg,
        "discovery": discovery_meter.avg,
        "discovery_unknown": discovery_unknown_meter.avg,
        "discovery_energy": discovery_energy_meter.avg,
        "discovery_selective_unknown": discovery_selective_unknown_meter.avg,
        "discovery_selective_energy": discovery_selective_energy_meter.avg,
        "discovery_selected_ratio": discovery_selected_meter.avg,
        "discovery_raw_selected_ratio": discovery_raw_selected_meter.avg,
        "discovery_neighbor_agreement": discovery_neighbor_agreement_meter.avg,
        "discovery_weight_mean": discovery_weight_mean_meter.avg,
        "joint_discovery": joint_meter.avg,
        "joint_consistency": joint_consistency_meter.avg,
        "joint_balance": joint_balance_meter.avg,
        "joint_neighbor": joint_neighbor_meter.avg,
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
):
    model.eval()
    all_logits = []
    all_probs = []
    all_entropy = []
    all_expected_entropy = []
    all_epistemic = []
    all_aleatoric = []
    all_head_uncertainty = []
    all_features = []
    all_projections = []
    all_odin_msp = []
    all_odin_logits = []
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
        all_projections.append(out["proj"].cpu())
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
        "features": torch.cat(all_features).numpy(),
        "projections": torch.cat(all_projections).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "raw_labels": torch.cat(all_raw).numpy(),
        "is_known": torch.cat(all_known).numpy(),
    }
    if all_odin_msp:
        outputs["odin_msp"] = torch.cat(all_odin_msp).numpy()
        outputs["odin_logits"] = torch.cat(all_odin_logits).numpy()
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


def collect_diagonal_gaussian_stats(model, loader, device, num_classes: int):
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

    centered = []
    for class_index, items in enumerate(features_by_class):
        if items:
            centered.append(np.asarray(items, dtype=np.float32) - means[class_index])
    centered_features = np.concatenate(centered, axis=0) if centered else global_features - global_features.mean(axis=0)
    feature_dim = centered_features.shape[1]
    if len(centered_features) <= 1:
        covariance = np.diag(global_var)
    else:
        covariance = (centered_features.T @ centered_features) / max(len(centered_features) - 1, 1)
    # A small diagonal shrinkage keeps the shared covariance stable and much
    # faster than iterative covariance estimators during repeated experiments.
    shrinkage = 0.1
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance = covariance + np.eye(feature_dim, dtype=np.float32) * 1e-3
    precision = np.linalg.pinv(covariance).astype(np.float32)
    return {"means": means, "variances": variances, "precision": precision}


def compute_mahalanobis_distance(
    features: np.ndarray,
    gaussian_stats: Dict[str, np.ndarray],
    covariance: str = "auto",
) -> np.ndarray:
    feat = np.asarray(features)[:, None, :]
    means = np.asarray(gaussian_stats["means"])[None, :, :]
    use_shared = covariance == "shared" or (covariance == "auto" and "precision" in gaussian_stats)
    if use_shared:
        if "precision" not in gaussian_stats:
            raise ValueError("shared Mahalanobis score requires gaussian_stats['precision']")
        precision = np.asarray(gaussian_stats["precision"], dtype=float)
        diff = feat - means
        distances = np.einsum("ncd,df,ncf->nc", diff, precision, diff) / max(diff.shape[-1], 1)
        distances = np.maximum(distances, 0.0)
    else:
        variances = np.asarray(gaussian_stats["variances"])[None, :, :]
        distances = ((feat - means) ** 2 / np.clip(variances, 1e-6, None)).mean(axis=-1)
    return distances.min(axis=1)


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
    if gaussian_stats is not None:
        mahalanobis = compute_mahalanobis_distance(
            outputs["features"], gaussian_stats, covariance=mahalanobis_mode
        )

    if score_mode == "full":
        score = weights[0] * entropy + weights[1] * epistemic + weights[2] * aleatoric
        if proto_dist is not None:
            score = score + proto_dist
    elif score_mode == "max_softmax":
        score = 1.0 - outputs["probs"].max(axis=1)
    elif score_mode == "odin_msp":
        if "odin_msp" not in outputs:
            raise ValueError("odin_msp score requires --odin-epsilon greater than 0")
        score = np.asarray(outputs["odin_msp"], dtype=float)
    elif score_mode == "energy":
        logits = outputs["logits"]
        temperature = 1.0
        score = -temperature * np.logaddexp.reduce(logits / temperature, axis=1)
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
) -> Dict[str, Dict[str, float]]:
    """Fit score statistics using known validation samples only."""
    entropy = np.asarray(outputs_known["entropy"], dtype=float)

    def stats(values):
        values = np.asarray(values, dtype=float)
        std = float(np.std(values))
        return {"mean": float(np.mean(values)), "std": max(std, 1e-6)}

    result = {"entropy": stats(entropy)}
    if prototypes is not None:
        _, proto_dist = compute_open_score(
            outputs_known,
            prototypes=prototypes,
            score_mode="proto_only",
        )
        result["proto_dist"] = stats(proto_dist)
    if gaussian_stats is not None:
        result["mahalanobis"] = stats(compute_mahalanobis_distance(outputs_known["features"], gaussian_stats))
        result["mahalanobis_diag"] = stats(
            compute_mahalanobis_distance(outputs_known["features"], gaussian_stats, covariance="diag")
        )
        if "precision" in gaussian_stats:
            result["mahalanobis_shared"] = stats(
                compute_mahalanobis_distance(outputs_known["features"], gaussian_stats, covariance="shared")
            )
    return result


def calibrate_threshold(scores_known: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(scores_known, percentile))


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
    upper = min(int(max_clusters), n - 1, 20)
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
        repeat_labels = []
        for repeat in range(max(1, int(stability_repeats))):
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
    stability = np.asarray([row["stability_nmi"] for row in rows], dtype=float)
    internal = 0.5 * silhouette + 0.25 * ch + 0.25 * db
    if selection == "silhouette":
        combined = silhouette
    elif selection == "stability":
        combined = 0.5 * internal + 0.5 * stability
    elif selection == "composite":
        combined = internal
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
    mahalanobis = (
        compute_mahalanobis_distance(outputs["features"], gaussian_stats)
        if gaussian_stats is not None
        else np.zeros_like(score)
    )
    pred_known = score <= threshold
    known_mask = outputs["is_known"].astype(bool)
    open_labels = (~known_mask).astype(int)
    auroc = compute_auroc(open_labels, score)
    aupr = compute_aupr(open_labels, score)
    fpr95 = compute_fpr95(open_labels, score)
    open_confusion = open_set_confusion(known_mask, pred_known)
    pred_class = outputs["logits"].argmax(axis=1)
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
    }
    result.update(open_confusion)
    novel_mask = ~pred_known
    selected_cluster_k = None
    cluster_diagnostics = []
    true_unknown_k = int(np.unique(outputs["raw_labels"][~known_mask]).size)
    if cluster_feature in {"projection", "projection_pca"}:
        raw_cluster_features = outputs.get("projections", outputs["features"])
    elif cluster_feature in {"feature", "feature_pca"}:
        raw_cluster_features = outputs["features"]
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
    if novel_mask.sum() > 1 and num_novel > 0:
        raw_novel_features = np.asarray(raw_cluster_features[novel_mask], dtype=np.float32)
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
        novel_true = outputs["raw_labels"][novel_mask]
        selected_cluster_k = (
            min(num_novel, len(novel_features))
            if cluster_k == "oracle"
            else None
        )
        if cluster_k != "oracle":
            selected_cluster_k, cluster_diagnostics = evaluate_cluster_candidates(
                novel_features,
                max_clusters=num_novel,
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
        if len(np.unique(novel_pred)) > 0:
            result.update({f"cluster_{k}": v for k, v in clustering_report(novel_true, novel_pred).items()})
            true_unknown_candidates = (~known_mask)[novel_mask]
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
            "cluster_diagnostics": cluster_diagnostics,
        }
    )
    detail = {
        "score": score,
        "score_mode": score_mode,
        "entropy": entropy,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "expected_entropy": outputs.get("expected_entropy", aleatoric),
        "head_uncertainty": outputs.get("head_uncertainty", aleatoric),
        "proto_dist": proto_dist,
        "mahalanobis": mahalanobis,
        "odin_msp": np.asarray(outputs.get("odin_msp", np.zeros_like(score)), dtype=float),
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
    }
    if novel_mask.sum() > 1 and num_novel > 0:
        pred_cluster = np.full(len(score), -1, dtype=np.int64)
        raw_cluster_features = (
            outputs.get("projections", outputs["features"])
            if cluster_feature.startswith("projection")
            else outputs["features"]
        )
        raw_cluster_features = np.asarray(raw_cluster_features[novel_mask], dtype=np.float32)
        cluster_features = (
            prepare_cluster_features(
                raw_cluster_features,
                use_pca=cluster_feature.endswith("_pca"),
                pca_dim=cluster_pca_dim,
                whiten=cluster_whiten,
            )
            if cluster_normalize or cluster_feature.endswith("_pca")
            else raw_cluster_features
        )
        pred_cluster[novel_mask] = cluster_unknown_samples(
            cluster_features,
            num_clusters=selected_cluster_k,
            method=cluster_method,
            n_init=cluster_n_init,
        )
        detail["pred_cluster"] = pred_cluster
    return result, score, pred_known, detail
