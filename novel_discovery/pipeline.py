from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import (
    classification_loss,
    cluster_prototype_consistency_loss,
    distillation_loss,
    feature_distillation_loss,
    nt_xent_loss,
    pseudo_label_consistency_loss,
    pseudo_unknown_loss,
    prototype_alignment_loss,
    supervised_contrastive_loss,
    uncertainty_alignment_loss,
    view_consistency_loss,
)
from .metrics import (
    clustering_report,
    compute_aupr,
    compute_auroc,
    compute_fpr95,
    compute_oscr,
    detection_cluster_split,
    expected_calibration_error,
    open_set_confusion,
)
from .utils import AverageMeter


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 4):
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
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


def train_one_epoch_teacher(
    model,
    loader,
    optimizer,
    device,
    alpha_unc: float = 0.1,
    alpha_proto: float = 0.0,
    alpha_pseudo: float = 0.0,
    pseudo_mode: str = "strong",
    pseudo_feature_noise: float = 0.05,
    uncertainty_target_mode: str = "confidence",
):
    model.train()
    ce_meter = AverageMeter()
    unc_meter = AverageMeter()
    proto_meter = AverageMeter()
    pseudo_meter = AverageMeter()
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
        pseudo_images = make_pseudo_unknown(images, mode=pseudo_mode)
        pseudo_out = model(pseudo_images)
        pseudo_features = pseudo_out["features"] + pseudo_feature_noise * torch.randn_like(pseudo_out["features"])
        pseudo_logits, pseudo_uncertainty = pseudo_forward_from_features(model, pseudo_features)
        loss_pseudo = pseudo_unknown_loss(pseudo_logits, pseudo_uncertainty)
        loss = loss_ce + alpha_unc * loss_unc + alpha_proto * loss_proto + alpha_pseudo * loss_pseudo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_meter.update(loss_ce.item(), images.size(0))
        unc_meter.update(loss_unc.item(), images.size(0))
        proto_meter.update(loss_proto.item(), images.size(0))
        pseudo_meter.update(loss_pseudo.item(), images.size(0))
    return {"ce": ce_meter.avg, "unc": unc_meter.avg, "proto": proto_meter.avg, "pseudo": pseudo_meter.avg}


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
    alpha_pseudo: float = 0.0,
    pseudo_mode: str = "strong",
    pseudo_feature_noise: float = 0.05,
    uncertainty_target_mode: str = "confidence",
    temperature: float = 2.0,
    kd_mode: str = "uncertainty",
    kd_weight_clip_min: float = 0.05,
    kd_weight_clip_max: float = 1.0,
    normalize_kd_weights: bool = True,
):
    student.train()
    teacher.eval()
    ce_meter = AverageMeter()
    kd_meter = AverageMeter()
    feat_kd_meter = AverageMeter()
    unc_meter = AverageMeter()
    sc_meter = AverageMeter()
    proto_meter = AverageMeter()
    pseudo_meter = AverageMeter()
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
            weight_clip_min=kd_weight_clip_min,
            weight_clip_max=kd_weight_clip_max,
            normalize_weights=normalize_kd_weights,
        )
        loss_feat_kd = feature_distillation_loss(
            s_out["proj"],
            t_out["proj"],
            teacher_uncertainty=t_out["uncertainty"] if kd_mode == "uncertainty" else None,
            weight_clip_min=kd_weight_clip_min,
            weight_clip_max=kd_weight_clip_max,
            normalize_weights=normalize_kd_weights,
        )
        loss_supcon = supervised_contrastive_loss(s_out["proj"], labels)
        loss_proto = prototype_alignment_loss(s_out["features"], labels, student.classifier.weight)
        pseudo_images = make_pseudo_unknown(images, mode=pseudo_mode)
        pseudo_out = student(pseudo_images)
        pseudo_features = pseudo_out["features"] + pseudo_feature_noise * torch.randn_like(pseudo_out["features"])
        pseudo_logits, pseudo_uncertainty = pseudo_forward_from_features(student, pseudo_features)
        loss_pseudo = pseudo_unknown_loss(pseudo_logits, pseudo_uncertainty)
        loss = (
            loss_ce
            + alpha_unc * loss_unc
            + alpha_kd * loss_kd
            + alpha_feat_kd * loss_feat_kd
            + alpha_supcon * loss_supcon
            + alpha_proto * loss_proto
            + alpha_pseudo * loss_pseudo
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ce_meter.update(loss_ce.item(), images.size(0))
        kd_meter.update(loss_kd.item(), images.size(0))
        feat_kd_meter.update(loss_feat_kd.item(), images.size(0))
        unc_meter.update(loss_unc.item(), images.size(0))
        sc_meter.update(loss_supcon.item(), images.size(0))
        proto_meter.update(loss_proto.item(), images.size(0))
        pseudo_meter.update(loss_pseudo.item(), images.size(0))
    return {
        "ce": ce_meter.avg,
        "kd": kd_meter.avg,
        "feat_kd": feat_kd_meter.avg,
        "unc": unc_meter.avg,
        "supcon": sc_meter.avg,
        "proto": proto_meter.avg,
        "pseudo": pseudo_meter.avg,
    }


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-8, None)


def assign_discovery_pseudo_labels(
    features: np.ndarray,
    num_clusters: int | str,
    max_clusters: int | None = None,
    confidence_percentile: float = 50.0,
    criterion: str = "combined",
):
    """Cluster unlabeled discovery-pool features and keep only stable assignments."""
    features = _l2_normalize(features)
    n = len(features)
    if n == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, np.zeros((0, 0), dtype=np.float32), {
            "selected_k": 0,
            "n_confident": 0,
        }
    if isinstance(num_clusters, str) and num_clusters == "auto":
        selected_k, criterion_scores = estimate_num_clusters(
            features,
            max_clusters=max_clusters,
            criterion=criterion,
        )
    else:
        selected_k = int(max(1, min(int(num_clusters), n)))
        criterion_scores = {}
    km = KMeans(n_clusters=max(1, selected_k), n_init=10, random_state=42)
    fitted = km.fit(features)
    labels = fitted.labels_.astype(np.int64)
    centroids = fitted.cluster_centers_.astype(np.float32)
    distances = np.linalg.norm(features - centroids[labels], axis=1)
    threshold = float(np.percentile(distances, confidence_percentile)) if n > 1 else float(distances[0])
    confident = distances <= threshold
    cluster_ids = np.full(n, -1, dtype=np.int64)
    cluster_ids[confident] = labels[confident]
    return cluster_ids, labels, centroids, {
        "selected_k": int(selected_k),
        "n_confident": int(confident.sum()),
        "confidence_percentile": float(confidence_percentile),
        "distance_threshold": float(threshold),
        "criterion_scores": criterion_scores,
    }


@torch.no_grad()
def collect_discovery_features(model, loader, device):
    model.eval()
    n = len(loader.dataset)
    projections = None
    for batch in tqdm(loader, desc="discovery-features", leave=False):
        images, labels, raw_labels, is_known, sample_ids = batch
        images = images.to(device)
        sample_ids = np.asarray(sample_ids, dtype=np.int64)
        out = model(images)
        proj = out["proj"].cpu().numpy()
        if projections is None:
            projections = np.zeros((n, proj.shape[1]), dtype=np.float32)
        projections[sample_ids] = proj
    if projections is None:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return projections, np.arange(n, dtype=np.int64)


def train_one_epoch_discovery(
    student,
    known_loader,
    discovery_loader,
    optimizer,
    device,
    teacher=None,
    cluster_ids: np.ndarray | None = None,
    cluster_centroids: np.ndarray | None = None,
    alpha_unc: float = 0.1,
    alpha_kd: float = 1.0,
    alpha_feat_kd: float = 0.0,
    alpha_supcon: float = 0.1,
    alpha_proto: float = 0.0,
    alpha_consistency: float = 0.5,
    alpha_contrast: float = 0.1,
    alpha_cluster: float = 0.0,
    uncertainty_target_mode: str = "confidence",
    temperature: float = 2.0,
    kd_mode: str = "uncertainty",
    kd_weight_clip_min: float = 0.05,
    kd_weight_clip_max: float = 1.0,
    normalize_kd_weights: bool = True,
    contrast_temperature: float = 0.2,
):
    """Joint known-class training and unlabeled discovery-pool consistency."""
    student.train()
    if teacher is not None:
        teacher.eval()
    meters = {name: AverageMeter() for name in ["ce", "kd", "feat_kd", "unc", "supcon", "proto", "consistency", "contrast", "cluster"]}
    known_iter = iter(known_loader)
    cluster_ids_t = None
    centroids_t = None
    if cluster_ids is not None:
        cluster_ids_t = torch.as_tensor(cluster_ids, dtype=torch.long, device=device)
    if cluster_centroids is not None and len(cluster_centroids) > 0:
        centroids_t = torch.as_tensor(cluster_centroids, dtype=torch.float32, device=device)

    for batch in tqdm(discovery_loader, desc="discovery-train", leave=False):
        view1, view2, _mapped, _raw, _known, real_index = batch
        view1 = view1.to(device)
        view2 = view2.to(device)
        try:
            known_batch = next(known_iter)
        except StopIteration:
            known_iter = iter(known_loader)
            known_batch = next(known_iter)
        images, labels, *_ = known_batch
        images = images.to(device)
        labels = labels.to(device)

        s_out = student(images)
        loss_ce = classification_loss(s_out["logits"], labels)
        loss_unc = uncertainty_alignment_loss(
            s_out["uncertainty"], s_out["logits"], labels, target_mode=uncertainty_target_mode
        )
        loss_supcon = supervised_contrastive_loss(s_out["proj"], labels)
        loss_proto = prototype_alignment_loss(s_out["features"], labels, student.classifier.weight)
        loss_kd = s_out["logits"].new_tensor(0.0)
        loss_feat_kd = s_out["logits"].new_tensor(0.0)
        if teacher is not None:
            with torch.no_grad():
                t_out = teacher(images)
            loss_kd = distillation_loss(
                s_out["logits"],
                t_out["logits"],
                teacher_uncertainty=t_out["uncertainty"],
                temperature=temperature,
                uncertainty_weighted=(kd_mode == "uncertainty"),
                weight_clip_min=kd_weight_clip_min,
                weight_clip_max=kd_weight_clip_max,
                normalize_weights=normalize_kd_weights,
            )
            loss_feat_kd = feature_distillation_loss(
                s_out["proj"],
                t_out["proj"],
                teacher_uncertainty=t_out["uncertainty"] if kd_mode == "uncertainty" else None,
                weight_clip_min=kd_weight_clip_min,
                weight_clip_max=kd_weight_clip_max,
                normalize_weights=normalize_kd_weights,
            )

        out_a = student(view1)
        out_b = student(view2)
        loss_consistency = view_consistency_loss(out_a["proj"], out_b["proj"])
        loss_contrast = nt_xent_loss(out_a["proj"], out_b["proj"], temperature=contrast_temperature)
        loss_cluster = s_out["logits"].new_tensor(0.0)
        if cluster_ids_t is not None and centroids_t is not None and centroids_t.numel() > 0:
            batch_ids = cluster_ids_t[real_index.to(device)]
            proto = torch.nn.functional.normalize(centroids_t, dim=-1)
            logits_a = torch.matmul(torch.nn.functional.normalize(out_a["proj"], dim=-1), proto.T)
            logits_b = torch.matmul(torch.nn.functional.normalize(out_b["proj"], dim=-1), proto.T)
            loss_cluster = pseudo_label_consistency_loss(logits_a, logits_b, batch_ids)
            loss_cluster = loss_cluster + cluster_prototype_consistency_loss(
                out_a["proj"], out_b["proj"], batch_ids, centroids_t
            )

        loss = (
            loss_ce
            + alpha_unc * loss_unc
            + alpha_kd * loss_kd
            + alpha_feat_kd * loss_feat_kd
            + alpha_supcon * loss_supcon
            + alpha_proto * loss_proto
            + alpha_consistency * loss_consistency
            + alpha_contrast * loss_contrast
            + alpha_cluster * loss_cluster
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = images.size(0)
        meters["ce"].update(loss_ce.item(), n)
        meters["kd"].update(loss_kd.item(), n)
        meters["feat_kd"].update(loss_feat_kd.item(), n)
        meters["unc"].update(loss_unc.item(), n)
        meters["supcon"].update(loss_supcon.item(), n)
        meters["proto"].update(loss_proto.item(), n)
        meters["consistency"].update(loss_consistency.item(), view1.size(0))
        meters["contrast"].update(loss_contrast.item(), view1.size(0))
        meters["cluster"].update(loss_cluster.item(), view1.size(0))
    return {key: meter.avg for key, meter in meters.items()}


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
def collect_simple_outputs(model, loader, device):
    """One-pass logits and uncertainty without MC Dropout."""
    model.eval()
    logits = []
    uncertainty = []
    labels = []
    for batch in tqdm(loader, desc="simple-extract", leave=False):
        images, y, *_ = batch
        out = model(images.to(device))
        logits.append(out["logits"].cpu())
        uncertainty.append(out["uncertainty"].cpu())
        labels.append(y if torch.is_tensor(y) else torch.as_tensor(y))
    return {
        "logits": torch.cat(logits).numpy(),
        "uncertainty": torch.cat(uncertainty).numpy(),
        "labels": torch.cat(labels).numpy(),
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
def extract_outputs(model, loader, device, mc_samples: int = 8):
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
    all_labels = []
    all_raw = []
    all_known = []
    for batch in tqdm(loader, desc="extract", leave=False):
        images, labels, raw_labels, is_known, _ = batch
        images = images.to(device)
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
        all_labels.append(labels)
        all_raw.append(raw_labels)
        all_known.append(is_known)
    return {
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


def compute_prototype_distance(features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    feat = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    proto = prototypes / np.clip(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8, None)
    sim = feat @ proto.T
    return 1.0 - sim.max(axis=1)


def collect_diagonal_gaussian_stats(model, loader, device, num_classes: int):
    """Collect class means and diagonal covariance for stable Mahalanobis OOD scoring."""
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
    return {"means": np.asarray(means), "variances": np.asarray(variances)}


def compute_mahalanobis_distance(features: np.ndarray, gaussian_stats: Dict[str, np.ndarray]) -> np.ndarray:
    feat = np.asarray(features)[:, None, :]
    means = np.asarray(gaussian_stats["means"])[None, :, :]
    variances = np.asarray(gaussian_stats["variances"])[None, :, :]
    distances = ((feat - means) ** 2 / np.clip(variances, 1e-6, None)).mean(axis=-1)
    return distances.min(axis=1)


def _zscore(values: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - stats["mean"]) / stats["std"]


def compute_open_score(
    outputs: Dict[str, np.ndarray],
    prototypes: np.ndarray | None = None,
    weights=(0.5, 0.3, 0.2),
    score_mode: str = "full",
    normalization: Dict[str, Dict[str, float]] | None = None,
    gaussian_stats: Dict[str, np.ndarray] | None = None,
    temperature: float = 1.0,
):
    entropy = outputs["entropy"]
    epistemic = outputs["epistemic"]
    aleatoric = outputs["aleatoric"]
    proto_dist = None
    if prototypes is not None:
        proto_dist = compute_prototype_distance(outputs["features"], prototypes)
    mahalanobis = None
    if gaussian_stats is not None:
        mahalanobis = compute_mahalanobis_distance(outputs["features"], gaussian_stats)
    temperature = max(float(temperature), 1e-3)

    if score_mode == "full":
        score = weights[0] * entropy + weights[1] * epistemic + weights[2] * aleatoric
        if proto_dist is not None:
            score = score + proto_dist
    elif score_mode == "max_softmax":
        score = 1.0 - outputs["probs"].max(axis=1)
    elif score_mode == "energy":
        logits = np.asarray(outputs["logits"], dtype=float)
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
    elif score_mode == "mahalanobis":
        if mahalanobis is None:
            raise ValueError("mahalanobis score requires gaussian statistics")
        score = mahalanobis
    elif score_mode == "entropy_mahalanobis":
        if mahalanobis is None:
            raise ValueError("entropy_mahalanobis score requires gaussian statistics")
        score = entropy + mahalanobis
    elif score_mode == "normalized_entropy_mahalanobis":
        if mahalanobis is None or normalization is None:
            raise ValueError("normalized_entropy_mahalanobis requires statistics")
        score = (
            (entropy - normalization["entropy"]["mean"]) / normalization["entropy"]["std"]
            + (mahalanobis - normalization["mahalanobis"]["mean"])
            / normalization["mahalanobis"]["std"]
        )
    elif score_mode == "zscore_fusion":
        if normalization is None:
            raise ValueError("zscore_fusion requires known-validation normalization")
        parts = [_zscore(entropy, normalization["entropy"])]
        if proto_dist is not None and "proto_dist" in normalization:
            parts.append(_zscore(proto_dist, normalization["proto_dist"]))
        if mahalanobis is not None and "mahalanobis" in normalization:
            parts.append(_zscore(mahalanobis, normalization["mahalanobis"]))
        score = np.mean(np.stack(parts, axis=0), axis=0)
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    return score, proto_dist


def fit_score_normalization(
    outputs_known: Dict[str, np.ndarray],
    prototypes: np.ndarray | None,
    gaussian_stats: Dict[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Fit score statistics using known validation samples only."""
    if prototypes is None:
        raise ValueError("Score normalization requires prototypes")
    _, proto_dist = compute_open_score(
        outputs_known,
        prototypes=prototypes,
        score_mode="proto_only",
    )
    entropy = np.asarray(outputs_known["entropy"], dtype=float)
    proto_dist = np.asarray(proto_dist, dtype=float)

    def stats(values):
        std = float(np.std(values))
        return {"mean": float(np.mean(values)), "std": max(std, 1e-6)}

    result = {"entropy": stats(entropy), "proto_dist": stats(proto_dist)}
    if gaussian_stats is not None:
        result["mahalanobis"] = stats(
            compute_mahalanobis_distance(outputs_known["features"], gaussian_stats)
        )
    return result


def calibrate_threshold(scores_known: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(scores_known, percentile))


def estimate_num_clusters(
    features: np.ndarray,
    max_clusters: int | None = None,
    min_clusters: int = 2,
    criterion: str = "combined",
):
    """Estimate K from internal cluster validity scores only.

    The search range never uses the true unknown class count. ``max_clusters``
    is an optional upper bound from sample size or a user-provided search
    window, not the oracle novel-class count.
    """
    n = len(features)
    if n < 4:
        selected = max(1, n)
        return selected, {"selected_k": selected, "reason": "too_few_samples"}
    default_upper = max(min_clusters, int(np.sqrt(n)))
    if max_clusters is None:
        upper = min(n - 1, max(default_upper, 8), 30)
    else:
        upper = min(int(max_clusters), n - 1, 30)
    if upper < min_clusters:
        return 1, {"selected_k": 1, "reason": "upper_lt_min"}
    values = _l2_normalize(np.asarray(features))
    if n > 2000:
        rng = np.random.default_rng(42)
        values = values[rng.choice(n, 2000, replace=False)]
    records = []
    for k in range(min_clusters, upper + 1):
        try:
            labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(values)
            if len(np.unique(labels)) < 2:
                continue
            sil = float(silhouette_score(values, labels))
            ch = float(calinski_harabasz_score(values, labels))
            db = float(davies_bouldin_score(values, labels))
        except ValueError:
            continue
        records.append({"k": int(k), "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
    if not records:
        return min_clusters, {"selected_k": min_clusters, "reason": "no_valid_k"}

    def _z(key, invert=False):
        arr = np.asarray([row[key] for row in records], dtype=float)
        std = float(arr.std()) if arr.size > 1 else 1.0
        std = max(std, 1e-6)
        values_z = (arr - float(arr.mean())) / std
        return -values_z if invert else values_z

    if criterion == "silhouette":
        scores = _z("silhouette")
    elif criterion == "calinski_harabasz":
        scores = _z("calinski_harabasz")
    elif criterion == "davies_bouldin":
        scores = _z("davies_bouldin", invert=True)
    else:
        scores = _z("silhouette") + _z("calinski_harabasz") + _z("davies_bouldin", invert=True)
    best_idx = int(np.argmax(scores))
    selected = int(records[best_idx]["k"])
    return selected, {
        "selected_k": selected,
        "criterion": criterion,
        "candidates": records,
        "combined_scores": [float(x) for x in scores],
    }


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
    """Fit a scalar temperature on known validation logits."""
    known = np.asarray(labels) >= 0
    if known.sum() == 0:
        return 1.0
    logit_t = torch.as_tensor(logits[known], dtype=torch.float32)
    label_t = torch.as_tensor(labels[known], dtype=torch.long)
    log_t = torch.nn.Parameter(torch.zeros(()))
    optim = torch.optim.LBFGS([log_t], lr=0.25, max_iter=max_iter)

    def closure():
        optim.zero_grad()
        temperature = log_t.exp().clamp(min=0.05, max=20.0)
        loss = torch.nn.functional.cross_entropy(logit_t / temperature, label_t)
        loss.backward()
        return loss

    try:
        optim.step(closure)
    except RuntimeError:
        return 1.0
    return float(log_t.exp().clamp(min=0.05, max=20.0).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    temperature = max(float(temperature), 1e-3)
    shifted = logits / temperature
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)


def teacher_uncertainty_diagnostics(
    teacher_uncertainty: np.ndarray,
    teacher_logits: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Check whether teacher uncertainty tracks classification error."""
    labels = np.asarray(labels)
    known = labels >= 0
    if known.sum() == 0:
        return {"n": 0}
    uncertainty = np.asarray(teacher_uncertainty, dtype=float)[known]
    logits = np.asarray(teacher_logits)[known]
    pred = logits.argmax(axis=1)
    error = (pred != labels[known]).astype(float)
    confidence = apply_temperature(logits, 1.0).max(axis=1)
    if uncertainty.std() < 1e-8 or error.std() < 1e-8:
        corr_error = float("nan")
    else:
        corr_error = float(np.corrcoef(uncertainty, error)[0, 1])
    if uncertainty.std() < 1e-8 or confidence.std() < 1e-8:
        corr_conf = float("nan")
    else:
        corr_conf = float(np.corrcoef(uncertainty, confidence)[0, 1])
    return {
        "n": int(known.sum()),
        "uncertainty_mean": float(uncertainty.mean()),
        "uncertainty_std": float(uncertainty.std()),
        "error_rate": float(error.mean()),
        "corr_with_error": corr_error,
        "corr_with_confidence": corr_conf,
        "ece": expected_calibration_error(confidence, 1.0 - error),
    }


def cluster_unknown_samples(features: np.ndarray, num_clusters: int):
    features = _l2_normalize(features)
    num_clusters = int(max(1, min(num_clusters, len(features))))
    if num_clusters == 1:
        return np.zeros(len(features), dtype=np.int64)
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    return km.fit_predict(features)


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def measure_inference_time(model, loader, device, max_batches: int = 10) -> Dict[str, float]:
    model.eval()
    times = []
    n_images = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            images = batch[0].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if start is not None:
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(images)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / 1000.0)
            else:
                import time

                t0 = time.perf_counter()
                model(images)
                times.append(time.perf_counter() - t0)
            n_images += images.size(0)
    total = float(sum(times)) if times else float("nan")
    return {
        "batches": len(times),
        "images": int(n_images),
        "seconds": total,
        "ms_per_image": float(1000.0 * total / max(n_images, 1)) if times else float("nan"),
    }


def apply_temperature_to_outputs(outputs: Dict[str, np.ndarray], temperature: float) -> Dict[str, np.ndarray]:
    calibrated = dict(outputs)
    probs = apply_temperature(outputs["logits"], temperature)
    calibrated["probs"] = probs
    calibrated["entropy"] = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=1)
    calibrated["temperature"] = np.asarray(temperature)
    return calibrated


SCORE_COMPARE_MODES = [
    "max_softmax",
    "energy",
    "entropy_only",
    "proto_only",
    "mahalanobis",
    "entropy_proto",
    "entropy_mahalanobis",
    "normalized_entropy_proto",
    "normalized_entropy_mahalanobis",
    "zscore_fusion",
]


def compare_score_modes(
    outputs_val,
    outputs_test,
    prototypes,
    gaussian_stats,
    normalization,
    temperature: float = 1.0,
):
    """Evaluate each score on the test set after known-val thresholding.

    This uses test labels only for reporting, never for selecting a threshold.
    """
    rows = []
    for score_mode in SCORE_COMPARE_MODES:
        try:
            scores_val, _ = compute_open_score(
                outputs_val,
                prototypes=prototypes,
                score_mode=score_mode,
                normalization=normalization,
                gaussian_stats=gaussian_stats,
                temperature=temperature,
            )
            scores_test, _ = compute_open_score(
                outputs_test,
                prototypes=prototypes,
                score_mode=score_mode,
                normalization=normalization,
                gaussian_stats=gaussian_stats,
                temperature=temperature,
            )
        except ValueError:
            continue
        threshold = calibrate_threshold(scores_val, percentile=95.0)
        known_mask = np.asarray(outputs_test["is_known"]).astype(bool)
        open_labels = (~known_mask).astype(int)
        pred_known = scores_test <= threshold
        pred_class = outputs_test["logits"].argmax(axis=1)
        rows.append(
            {
                "score_mode": score_mode,
                "auroc": float(compute_auroc(open_labels, scores_test)),
                "aupr": float(compute_aupr(open_labels, scores_test)),
                "fpr95": float(compute_fpr95(open_labels, scores_test)),
                "oscr": float(compute_oscr(known_mask, scores_test, pred_class, outputs_test["labels"])),
                "unknown_reject_rate": float(((~known_mask) & (~pred_known)).sum() / max(int((~known_mask).sum()), 1)),
            }
        )
    rows.sort(key=lambda x: (x["auroc"] if x["auroc"] == x["auroc"] else -1.0), reverse=True)
    return rows


def run_discovery(
    outputs: Dict[str, np.ndarray],
    threshold: float,
    num_novel: int,
    prototypes: np.ndarray | None = None,
    score_mode: str = "full",
    normalization: Dict[str, Dict[str, float]] | None = None,
    gaussian_stats: Dict[str, np.ndarray] | None = None,
    cluster_k: int | str = "oracle",
    temperature: float = 1.0,
    cluster_confidence_percentile: float = 0.0,
    k_criterion: str = "combined",
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
        temperature=temperature,
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
    auroc = float(compute_auroc(open_labels, score))
    aupr = float(compute_aupr(open_labels, score))
    fpr95 = float(compute_fpr95(open_labels, score))
    open_confusion = open_set_confusion(known_mask, pred_known)
    pred_class = outputs["logits"].argmax(axis=1)
    true_labels = outputs["labels"]
    oscr = compute_oscr(known_mask, score, pred_class, true_labels)
    known_class_correct = int(np.sum(known_mask & pred_known & (pred_class == true_labels)))
    known_class_wrong = int(np.sum(known_mask & pred_known & (pred_class != true_labels)))
    known_total = int(known_mask.sum())
    known_conf = np.asarray(outputs["probs"]).max(axis=1)[known_mask]
    known_correctness = (pred_class[known_mask] == true_labels[known_mask]).astype(float)
    result = {
        "auroc": auroc,
        "aupr": aupr,
        "fpr95": fpr95,
        "oscr": oscr,
        "known_ece": expected_calibration_error(known_conf, known_correctness),
        "known_ratio": float(pred_known.mean()),
        "known_class_correct": known_class_correct,
        "known_class_wrong": known_class_wrong,
        "known_class_accuracy_after_accept": float(known_class_correct / max(known_class_correct + known_class_wrong, 1)),
        "known_class_accuracy_all_known": float(known_class_correct / max(known_total, 1)),
        "true_num_novel": int(num_novel),
    }
    result.update(open_confusion)
    novel_mask = ~pred_known
    if cluster_confidence_percentile > 0 and novel_mask.any():
        rejected_scores = score[novel_mask]
        keep_thr = float(np.percentile(rejected_scores, cluster_confidence_percentile))
        novel_mask = novel_mask & (score >= keep_thr)
    selected_cluster_k = None
    k_search = None
    novel_pred = None
    novel_true = None
    cluster_features = _l2_normalize(np.asarray(outputs.get("projections", outputs["features"])))
    if novel_mask.sum() > 1:
        novel_features = cluster_features[novel_mask]
        novel_true = outputs["raw_labels"][novel_mask]
        if cluster_k == "oracle":
            selected_cluster_k = min(max(int(num_novel), 1), len(novel_features))
            k_search = {"mode": "oracle", "true_num_novel": int(num_novel)}
        else:
            selected_cluster_k, k_search = estimate_num_clusters(
                novel_features,
                max_clusters=None,
                criterion=k_criterion,
            )
            k_search["mode"] = "auto"
            k_search["true_num_novel"] = int(num_novel)
        novel_pred = cluster_unknown_samples(novel_features, num_clusters=selected_cluster_k)
        if len(np.unique(novel_pred)) > 0:
            result.update({f"cluster_{k}": v for k, v in clustering_report(novel_true, novel_pred).items()})
        result["estimated_k"] = int(selected_cluster_k)
        result["true_k"] = int(num_novel)
        result["k_abs_error"] = abs(int(selected_cluster_k) - int(num_novel)) if cluster_k != "oracle" else 0
        result["cluster_k_mode"] = "oracle" if cluster_k == "oracle" else "auto"
    result.update(detection_cluster_split(known_mask, pred_known, novel_true, novel_pred))
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
        "pred_known": pred_known,
        "true_known": known_mask,
        "pred_class": pred_class,
        "true_label": true_labels,
        "raw_labels": outputs["raw_labels"],
        "pred_cluster": None,
        "cluster_k": selected_cluster_k,
        "cluster_k_search": k_search,
    }
    if novel_pred is not None:
        pred_cluster = np.full(len(score), -1, dtype=np.int64)
        pred_cluster[novel_mask] = novel_pred
        detail["pred_cluster"] = pred_cluster
    return result, score, pred_known, detail
