from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NovelPrototypeHead(nn.Module):
    """A lightweight learnable prototype head for unlabeled novel classes."""

    def __init__(self, feature_dim: int, num_novel: int, temperature: float = 0.2):
        super().__init__()
        if feature_dim <= 0 or num_novel <= 1:
            raise ValueError("feature_dim must be positive and num_novel must exceed one.")
        self.prototypes = nn.Parameter(torch.randn(num_novel, feature_dim))
        nn.init.normal_(self.prototypes, std=0.02)
        self.temperature = max(float(temperature), 1e-6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return features @ prototypes.T / self.temperature


@torch.no_grad()
def balanced_assignments(
    logits: torch.Tensor,
    temperature: float = 1.0,
    iterations: int = 3,
) -> torch.Tensor:
    """Approximate UNO-style balanced assignments with Sinkhorn scaling."""
    if logits.numel() == 0:
        return logits
    batch_size, num_classes = logits.shape
    q = torch.exp((logits / max(float(temperature), 1e-6)).T - logits.max(dim=1).values.detach())
    q = q / q.sum().clamp_min(1e-8)
    for _ in range(max(int(iterations), 1)):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    q = q * batch_size
    assignments = q.T.clamp_min(1e-8)
    return assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-8)


def balanced_assignment_loss(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Penalize collapse of the batch-average novel assignment distribution."""
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    probs = logits.softmax(dim=-1).mean(dim=0)
    num_classes = logits.size(-1)
    uniform = logits.new_tensor(1.0 / num_classes)
    return (probs * (probs.clamp_min(1e-8).log() - uniform.log())).sum()


def novel_consistency_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    confidence_threshold: float = 0.6,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Use one augmented view as a balanced pseudo-label teacher."""
    if first_logits.numel() == 0 or second_logits.numel() == 0:
        return first_logits.new_tensor(0.0)
    first_target = balanced_assignments(first_logits.detach(), temperature=temperature)
    second_target = balanced_assignments(second_logits.detach(), temperature=temperature)
    # Measure confidence relative to a uniform novel assignment. This keeps
    # the threshold meaningful when the number of novel classes is large.
    num_novel = first_logits.size(-1)
    first_prob = first_logits.softmax(dim=-1).max(dim=-1).values * num_novel
    second_prob = second_logits.softmax(dim=-1).max(dim=-1).values * num_novel
    first_mask = first_prob >= float(confidence_threshold)
    second_mask = second_prob >= float(confidence_threshold)
    first_loss = -(first_target * F.log_softmax(second_logits, dim=-1)).sum(dim=-1)
    second_loss = -(second_target * F.log_softmax(first_logits, dim=-1)).sum(dim=-1)
    losses = []
    if first_mask.any():
        losses.append(first_loss[first_mask].mean())
    if second_mask.any():
        losses.append(second_loss[second_mask].mean())
    if not losses:
        return first_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def neighbor_consistency_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Match novel predictions of nearby samples, following SCAN-style consistency."""
    if features.size(0) <= 1 or logits.numel() == 0 or k <= 0:
        return logits.new_tensor(0.0)
    effective_k = min(int(k), features.size(0) - 1)
    normalized = F.normalize(features, dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(float("-inf"))
    indices = similarity.topk(effective_k, dim=1, largest=True).indices
    probs = logits.softmax(dim=-1)
    neighbor_probs = probs[indices].detach().mean(dim=1)
    return F.kl_div(
        probs.clamp_min(1e-8).log(),
        neighbor_probs,
        reduction="batchmean",
    )


def joint_discovery_loss(
    first_features: torch.Tensor,
    second_features: torch.Tensor,
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    confidence_threshold: float = 0.6,
    assignment_temperature: float = 1.0,
    alpha_consistency: float = 1.0,
    alpha_balance: float = 0.1,
    alpha_neighbor: float = 0.1,
    neighbor_k: int = 5,
) -> dict[str, torch.Tensor]:
    """Combine novel pseudo-label, balance, and neighborhood objectives."""
    consistency = novel_consistency_loss(
        first_logits,
        second_logits,
        confidence_threshold=confidence_threshold,
        temperature=assignment_temperature,
    )
    balance = 0.5 * (
        balanced_assignment_loss(first_logits)
        + balanced_assignment_loss(second_logits)
    )
    neighbor = 0.5 * (
        neighbor_consistency_loss(first_features, first_logits, k=neighbor_k)
        + neighbor_consistency_loss(second_features, second_logits, k=neighbor_k)
    )
    total = (
        alpha_consistency * consistency
        + alpha_balance * balance
        + alpha_neighbor * neighbor
    )
    return {
        "total": total,
        "consistency": consistency,
        "balance": balance,
        "neighbor": neighbor,
    }
