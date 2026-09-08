from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import (
    classification_loss,
    distillation_loss,
    feature_distillation_loss,
    pseudo_unknown_loss,
    prototype_alignment_loss,
    supervised_contrastive_loss,
    uncertainty_alignment_loss,
)
from .metrics import clustering_report, compute_auroc, compute_fpr95, open_set_confusion
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
        )
        loss_feat_kd = feature_distillation_loss(
            s_out["proj"], t_out["proj"], teacher_uncertainty=t_out["uncertainty"]
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
    if gaussian_stats is not None:
        mahalanobis = compute_mahalanobis_distance(outputs["features"], gaussian_stats)

    if score_mode == "full":
        score = weights[0] * entropy + weights[1] * epistemic + weights[2] * aleatoric
        if proto_dist is not None:
            score = score + proto_dist
    elif score_mode == "max_softmax":
        score = 1.0 - outputs["probs"].max(axis=1)
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


def estimate_num_clusters(features: np.ndarray, max_clusters: int):
    """Estimate K without using unknown labels; used as an explicit baseline."""
    n = len(features)
    if n < 4:
        return max(1, n)
    upper = min(int(max_clusters), n - 1, 20)
    if upper < 2:
        return 1
    values = np.asarray(features)
    if n > 2000:
        rng = np.random.default_rng(42)
        values = values[rng.choice(n, 2000, replace=False)]
    best_k, best_score = 2, -1.0
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(values)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(values, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_unknown_samples(features: np.ndarray, num_clusters: int):
    num_clusters = int(max(1, min(num_clusters, len(features))))
    if num_clusters == 1:
        return np.zeros(len(features), dtype=np.int64)
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    return km.fit_predict(features)


def run_discovery(
    outputs: Dict[str, np.ndarray],
    threshold: float,
    num_novel: int,
    prototypes: np.ndarray | None = None,
    score_mode: str = "full",
    normalization: Dict[str, Dict[str, float]] | None = None,
    gaussian_stats: Dict[str, np.ndarray] | None = None,
    cluster_k: int | str = "oracle",
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
    fpr95 = compute_fpr95(open_labels, score)
    open_confusion = open_set_confusion(known_mask, pred_known)
    pred_class = outputs["logits"].argmax(axis=1)
    true_labels = outputs["labels"]
    known_class_correct = int(np.sum(known_mask & pred_known & (pred_class == true_labels)))
    known_class_wrong = int(np.sum(known_mask & pred_known & (pred_class != true_labels)))
    known_total = int(known_mask.sum())
    result = {
        "auroc": auroc,
        "fpr95": fpr95,
        "known_ratio": float(pred_known.mean()),
        "known_class_correct": known_class_correct,
        "known_class_wrong": known_class_wrong,
        "known_class_accuracy_after_accept": float(known_class_correct / max(known_class_correct + known_class_wrong, 1)),
        "known_class_accuracy_all_known": float(known_class_correct / max(known_total, 1)),
    }
    result.update(open_confusion)
    novel_mask = ~pred_known
    selected_cluster_k = None
    if novel_mask.sum() > 1 and num_novel > 0:
        novel_features = outputs.get("projections", outputs["features"])[novel_mask]
        novel_true = outputs["raw_labels"][novel_mask]
        selected_cluster_k = (
            min(num_novel, len(novel_features))
            if cluster_k == "oracle"
            else estimate_num_clusters(novel_features, max_clusters=num_novel)
        )
        novel_pred = cluster_unknown_samples(novel_features, num_clusters=selected_cluster_k)
        if len(np.unique(novel_pred)) > 0:
            result.update({f"cluster_{k}": v for k, v in clustering_report(novel_true, novel_pred).items()})
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
    }
    if novel_mask.sum() > 1 and num_novel > 0:
        pred_cluster = np.full(len(score), -1, dtype=np.int64)
        cluster_features = outputs.get("projections", outputs["features"])
        pred_cluster[novel_mask] = cluster_unknown_samples(
            cluster_features[novel_mask], num_clusters=selected_cluster_k
        )
        detail["pred_cluster"] = pred_cluster
    return result, score, pred_known, detail
