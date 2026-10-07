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


def known_residual_weights(
    known_logits: torch.Tensor,
    temperature: float = 1.0,
    floor: float = 0.0,
) -> torch.Tensor:
    """Estimate how much an unlabeled sample remains unexplained by known classes.

    The result is intended to be used as a detached sample weight by the
    mixed-pool objective, not as a hard unknown label. High-confidence known
    predictions contribute little to novel-prototype learning, while ambiguous
    samples can still contribute without forcing every sample into a novel
    class.
    """
    if known_logits.numel() == 0:
        return known_logits.new_empty((known_logits.size(0),))
    probs = (known_logits / max(float(temperature), 1e-6)).softmax(dim=-1)
    residual = 1.0 - probs.max(dim=-1).values
    return residual.clamp(min=float(floor), max=1.0)


def novel_mass_weights(
    known_logits: torch.Tensor,
    novel_logits: torch.Tensor,
    known_temperature: float = 1.0,
    floor: float = 0.0,
) -> torch.Tensor:
    """Return the probability mass assigned to the novel subspace.

    Unlike ``known_residual_weights``, this compares known and novel evidence
    in one calibrated logit space. It is still only a soft sample weight; it
    does not assign an unlabeled sample a hard novel label.
    """
    if known_logits.size(0) != novel_logits.size(0):
        raise ValueError("known and novel logits must have the same batch size")
    unified = combine_known_novel_logits(
        known_logits, novel_logits, known_temperature=known_temperature
    )
    probs = unified.softmax(dim=-1)
    mass = probs[:, known_logits.size(-1) :].sum(dim=-1)
    return mass.clamp(min=float(floor), max=1.0)


def bounded_novel_mass_weights(
    novel_mass: torch.Tensor,
    agreement: torch.Tensor | None = None,
    floor: float = 0.0,
) -> torch.Tensor:
    """Keep a nonzero learning path while preserving relative novel evidence.

    The floor is applied after optional two-view agreement. This avoids
    discarding every mixed-pool sample when the current novel head is still
    poorly initialized, while retaining larger weights for samples with
    stronger novel evidence.
    """
    if novel_mass.ndim != 1:
        raise ValueError("novel_mass must be a one-dimensional tensor")
    floor = float(floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor must be in [0, 1)")
    raw = novel_mass.clamp(0.0, 1.0)
    if agreement is not None:
        if agreement.shape != raw.shape:
            raise ValueError("agreement must match novel_mass shape")
        raw = raw * agreement.detach().to(raw).clamp(0.0, 1.0)
    return floor + (1.0 - floor) * raw


def neighbor_novel_support_weights(
    features: torch.Tensor,
    novel_mass: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Estimate local support for novel evidence from feature-space neighbors.

    A sample's weight is its own novel mass multiplied by the mean novel mass
    of its nearest non-self neighbors. Isolated high-mass outliers are thus
    downweighted, while locally coherent candidate groups retain weight.
    Inputs used to form the weights are detached so the support selection does
    not create an unintended gradient path through nearest-neighbor indices.
    This operates within the current mixed-pool batch, not a global memory bank.
    """
    if features.ndim != 2 or novel_mass.ndim != 1:
        raise ValueError("features must be [N,D] and novel_mass must be [N]")
    if features.size(0) != novel_mass.numel():
        raise ValueError("features and novel_mass must contain the same samples")
    n = features.size(0)
    if n == 0:
        return novel_mass
    mass = novel_mass.detach().to(features).clamp(0.0, 1.0)
    if n == 1 or k <= 0:
        return mass
    normalized = F.normalize(features.detach(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(float("-inf"))
    neighbor_indices = similarity.topk(min(int(k), n - 1), dim=-1).indices
    support = mass[neighbor_indices].mean(dim=-1)
    return mass * support


def combine_known_novel_logits(
    known_logits: torch.Tensor | None,
    novel_logits: torch.Tensor,
    known_temperature: float = 1.0,
) -> torch.Tensor:
    """Build a unified known-plus-novel classification space."""
    if known_logits is None:
        return novel_logits
    if known_logits.size(0) != novel_logits.size(0):
        raise ValueError("known and novel logits must have the same batch size")
    temperature = max(float(known_temperature), 1e-6)
    return torch.cat([known_logits / temperature, novel_logits], dim=-1)


@torch.no_grad()
def balanced_assignments(
    logits: torch.Tensor,
    temperature: float = 1.0,
    iterations: int = 3,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate balanced assignments, optionally with weighted sample mass.

    Without ``sample_weights`` this preserves the original equal-mass Sinkhorn
    behavior. With weights, each sample receives a column marginal proportional
    to its detached weight. Zero-weight samples are excluded from transport and
    cannot change the balanced pseudo-labels of active samples.
    """
    if logits.numel() == 0:
        return logits
    batch_size, num_classes = logits.shape
    if sample_weights is not None:
        weights = sample_weights.reshape(-1).detach().to(logits).clamp_min(0.0)
        if weights.numel() != batch_size:
            raise ValueError("sample_weights must match the logits batch size")
        active = weights > 1e-12
        if not active.any():
            return logits.softmax(dim=-1)
        active_logits = logits[active]
        active_weights = weights[active]
        scaled_logits = active_logits / max(float(temperature), 1e-6)
        q = torch.exp(
            scaled_logits.T - scaled_logits.max(dim=1).values.detach()
        )
        q = q / q.sum().clamp_min(1e-8)
        sample_marginal = active_weights / active_weights.sum().clamp_min(1e-8)
        class_marginal = q.new_full((num_classes,), 1.0 / num_classes)
        for _ in range(max(int(iterations), 1)):
            q = q * (
                class_marginal / q.sum(dim=1).clamp_min(1e-8)
            ).unsqueeze(1)
            q = q * (
                sample_marginal / q.sum(dim=0).clamp_min(1e-8)
            ).unsqueeze(0)
        active_assignments = q.T / sample_marginal.unsqueeze(1).clamp_min(1e-8)
        active_assignments = active_assignments.clamp_min(1e-8)
        active_assignments = active_assignments / active_assignments.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        # Inactive rows are ignored by every weighted objective. A well-formed
        # probability row is still returned for API consistency.
        assignments = logits.softmax(dim=-1)
        assignments[active] = active_assignments
        return assignments

    scaled_logits = logits / max(float(temperature), 1e-6)
    q = torch.exp(
        scaled_logits.T - scaled_logits.max(dim=1).values.detach()
    )
    q = q / q.sum().clamp_min(1e-8)
    for _ in range(max(int(iterations), 1)):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    q = q * batch_size
    assignments = q.T.clamp_min(1e-8)
    return assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def global_balanced_assignments(
    current_logits: torch.Tensor,
    memory_logits: torch.Tensor | None = None,
    temperature: float = 1.0,
    iterations: int = 3,
    current_weights: torch.Tensor | None = None,
    memory_weight: float = 1.0,
) -> torch.Tensor:
    """Assign current samples using a detached cross-batch assignment pool.

    Batch-only Sinkhorn sees too few examples to reliably populate all novel
    prototypes.  This helper balances the current rows together with a FIFO
    memory bank, then returns only the current rows as pseudo-label targets.
    The memory rows are detached and never receive gradients.
    """
    if current_logits.ndim != 2:
        raise ValueError("current_logits must be two-dimensional")
    if current_logits.numel() == 0:
        return current_logits
    if memory_logits is None or memory_logits.numel() == 0:
        return balanced_assignments(
            current_logits,
            temperature=temperature,
            iterations=iterations,
            sample_weights=current_weights,
        )
    if memory_logits.ndim != 2:
        raise ValueError("memory_logits must be two-dimensional")
    if memory_logits.size(1) != current_logits.size(1):
        raise ValueError("memory and current logits must have the same class dimension")
    memory_weight = float(memory_weight)
    if memory_weight <= 0.0:
        raise ValueError("memory_weight must be positive")
    current = current_logits.detach()
    memory = memory_logits.detach().to(current)
    if current_weights is None:
        current_mass = current.new_ones(current.size(0))
    else:
        current_mass = current_weights.reshape(-1).detach().to(current).clamp_min(0.0)
        if current_mass.numel() != current.size(0):
            raise ValueError("current_weights must match current_logits rows")
    memory_mass = current.new_full((memory.size(0),), memory_weight)
    combined = torch.cat([current, memory], dim=0)
    assignments = balanced_assignments(
        combined,
        temperature=temperature,
        iterations=iterations,
        sample_weights=torch.cat([current_mass, memory_mass], dim=0),
    )
    return assignments[: current.size(0)]


def balanced_assignment_loss(
    logits: torch.Tensor,
    temperature: float = 1.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize collapse of the batch-average novel assignment distribution."""
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    probs = logits.softmax(dim=-1)
    if sample_weights is not None:
        weights = sample_weights.detach().to(logits).clamp_min(0.0)
        probs = (probs * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(1e-8)
    else:
        probs = probs.mean(dim=0)
    num_classes = logits.size(-1)
    uniform = logits.new_tensor(1.0 / num_classes)
    return (probs * (probs.clamp_min(1e-8).log() - uniform.log())).sum()


def information_maximization_loss(
    logits: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Minimize sample entropy while maximizing batch-marginal entropy."""
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    probs = logits.softmax(dim=-1).clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    if sample_weights is not None:
        weights = sample_weights.detach().to(logits).clamp_min(0.0)
        sample_entropy = (entropy * weights).sum() / weights.sum().clamp_min(1e-8)
        marginal = (probs * weights.unsqueeze(-1)).sum(dim=0)
        marginal = marginal / weights.sum().clamp_min(1e-8)
    else:
        sample_entropy = entropy.mean()
        marginal = probs.mean(dim=0)
    marginal = marginal.clamp_min(1e-8)
    marginal_entropy = -(marginal * marginal.log()).sum()
    return sample_entropy - marginal_entropy


def novel_consistency_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    confidence_threshold: float = 0.6,
    temperature: float = 1.0,
    sample_weights: torch.Tensor | None = None,
    weighted_assignments: bool = True,
    first_assignments: torch.Tensor | None = None,
    second_assignments: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use one augmented view as a balanced pseudo-label teacher."""
    if first_logits.numel() == 0 or second_logits.numel() == 0:
        return first_logits.new_tensor(0.0)
    assignment_weights = sample_weights if weighted_assignments else None
    first_target = first_assignments
    if first_target is None:
        first_target = balanced_assignments(
            first_logits.detach(), temperature=temperature, sample_weights=assignment_weights
        )
    second_target = second_assignments
    if second_target is None:
        second_target = balanced_assignments(
            second_logits.detach(), temperature=temperature, sample_weights=assignment_weights
        )
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
        values = first_loss[first_mask]
        if sample_weights is None:
            losses.append(values.mean())
        else:
            weights = sample_weights[first_mask].detach().to(values).clamp_min(0.0)
            losses.append((values * weights).sum() / weights.sum().clamp_min(1e-8))
    if second_mask.any():
        values = second_loss[second_mask]
        if sample_weights is None:
            losses.append(values.mean())
        else:
            weights = sample_weights[second_mask].detach().to(values).clamp_min(0.0)
            losses.append((values * weights).sum() / weights.sum().clamp_min(1e-8))
    if not losses:
        return first_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def prototype_pseudo_label_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    confidence_threshold: float = 1.0,
    assignment_temperature: float = 1.0,
    sample_weights: torch.Tensor | None = None,
    weighted_assignments: bool = True,
    first_assignments: torch.Tensor | None = None,
    second_assignments: torch.Tensor | None = None,
    target_mode: str = "hard",
) -> torch.Tensor:
    """Cross-view hard pseudo-label loss for prototype discovery.

    Sinkhorn assignments are used only as detached, approximately balanced
    targets.  Compared with a soft KL target, the hard target supplies a
    stronger class-specific gradient once KMeans or another initialization has
    broken prototype symmetry.  The two directions are averaged, and samples
    below the relative confidence threshold are ignored.
    """
    if first_logits.numel() == 0 or second_logits.numel() == 0:
        return first_logits.new_tensor(0.0)
    assignment_weights = sample_weights if weighted_assignments else None
    if first_assignments is None:
        first_assignments = balanced_assignments(
            first_logits.detach(),
            temperature=assignment_temperature,
            sample_weights=assignment_weights,
        )
    if second_assignments is None:
        second_assignments = balanced_assignments(
            second_logits.detach(),
            temperature=assignment_temperature,
            sample_weights=assignment_weights,
        )
    if target_mode not in {"hard", "soft"}:
        raise ValueError("target_mode must be 'hard' or 'soft'")
    if target_mode == "hard":
        first_target = first_assignments.argmax(dim=-1)
        second_target = second_assignments.argmax(dim=-1)
    else:
        first_target = first_assignments
        second_target = second_assignments
    first_conf = (
        first_logits.detach().softmax(dim=-1).max(dim=-1).values
        * first_logits.size(-1)
    )
    second_conf = (
        second_logits.detach().softmax(dim=-1).max(dim=-1).values
        * second_logits.size(-1)
    )

    def direction_loss(logits, targets, confidence):
        mask = confidence >= float(confidence_threshold)
        if not mask.any():
            return logits.new_tensor(0.0)
        if target_mode == "hard":
            values = F.cross_entropy(logits[mask], targets[mask], reduction="none")
        else:
            values = -(
                targets[mask].detach()
                * F.log_softmax(logits[mask], dim=-1)
            ).sum(dim=-1)
        if sample_weights is None:
            return values.mean()
        weights = sample_weights[mask].detach().to(values).clamp_min(0.0)
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    return 0.5 * (
        direction_loss(second_logits, first_target, first_conf)
        + direction_loss(first_logits, second_target, second_conf)
    )


def neighbor_consistency_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    k: int = 5,
    sample_weights: torch.Tensor | None = None,
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
    per_sample = F.kl_div(
        probs.clamp_min(1e-8).log(),
        neighbor_probs,
        reduction="none",
    ).sum(dim=-1)
    if sample_weights is not None:
        weights = sample_weights.detach().to(per_sample).clamp_min(0.0)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    return per_sample.mean()


def memory_neighbor_consistency_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    memory_features: torch.Tensor | None,
    memory_logits: torch.Tensor | None,
    k: int = 5,
    temperature: float = 0.2,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match current assignments to detached neighbors from a global queue."""
    if (
        features.ndim != 2
        or logits.ndim != 2
        or features.size(0) == 0
        or memory_features is None
        or memory_logits is None
        or memory_features.numel() == 0
        or memory_logits.numel() == 0
        or k <= 0
    ):
        return logits.new_tensor(0.0)
    if memory_features.ndim != 2 or memory_logits.ndim != 2:
        raise ValueError("memory features and logits must be two-dimensional")
    if memory_features.size(0) != memory_logits.size(0):
        raise ValueError("memory features and logits must have the same size")
    if memory_features.size(1) != features.size(1) or memory_logits.size(1) != logits.size(1):
        raise ValueError("memory feature/logit dimensions must match the query")
    query = F.normalize(features.detach(), dim=-1)
    bank = F.normalize(memory_features.detach().to(query), dim=-1)
    neighbors = (query @ bank.T).topk(min(int(k), bank.size(0)), dim=-1).indices
    target = F.softmax(
        memory_logits.detach().to(logits)[neighbors] / max(float(temperature), 1e-6),
        dim=-1,
    ).mean(dim=1)
    predicted = F.log_softmax(logits / max(float(temperature), 1e-6), dim=-1)
    per_sample = F.kl_div(predicted, target, reduction="none").sum(dim=-1)
    if sample_weights is None:
        return per_sample.mean()
    weights = sample_weights.detach().to(per_sample).reshape(-1).clamp_min(0.0)
    if weights.numel() != per_sample.numel():
        raise ValueError("sample_weights must match the query batch")
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def joint_discovery_loss(
    first_features: torch.Tensor,
    second_features: torch.Tensor,
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    confidence_threshold: float = 0.6,
    assignment_temperature: float = 1.0,
    alpha_consistency: float = 1.0,
    alpha_balance: float = 0.1,
    alpha_information: float = 0.1,
    alpha_neighbor: float = 0.1,
    alpha_pseudo: float = 0.0,
    neighbor_k: int = 5,
    first_known_logits: torch.Tensor | None = None,
    second_known_logits: torch.Tensor | None = None,
    known_temperature: float = 1.0,
    sample_weights: torch.Tensor | None = None,
    weighted_assignments: bool = True,
    memory_features: torch.Tensor | None = None,
    memory_logits: torch.Tensor | None = None,
    alpha_memory_neighbor: float = 0.0,
    memory_neighbor_k: int = 5,
    memory_temperature: float = 0.2,
    global_assignment: bool = False,
    global_memory_logits: torch.Tensor | None = None,
    global_memory_weight: float = 1.0,
    global_assignment_iterations: int = 3,
    pseudo_target_mode: str = "hard",
    first_target_logits: torch.Tensor | None = None,
    second_target_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combine novel or unified-space pseudo-label objectives."""
    first_joint_logits = combine_known_novel_logits(
        first_known_logits, first_logits, known_temperature=known_temperature
    )
    second_joint_logits = combine_known_novel_logits(
        second_known_logits, second_logits, known_temperature=known_temperature
    )

    first_assignments = None
    second_assignments = None
    assignment_first_logits = (
        first_joint_logits if first_target_logits is None else first_target_logits.detach()
    )
    assignment_second_logits = (
        second_joint_logits if second_target_logits is None else second_target_logits.detach()
    )
    if global_assignment:
        first_assignments = global_balanced_assignments(
            assignment_first_logits,
            memory_logits=global_memory_logits,
            temperature=assignment_temperature,
            iterations=global_assignment_iterations,
            current_weights=sample_weights,
            memory_weight=global_memory_weight,
        )
        second_assignments = global_balanced_assignments(
            assignment_second_logits,
            memory_logits=global_memory_logits,
            temperature=assignment_temperature,
            iterations=global_assignment_iterations,
            current_weights=sample_weights,
            memory_weight=global_memory_weight,
        )
    if first_target_logits is not None:
        first_assignments = balanced_assignments(
            assignment_first_logits,
            temperature=assignment_temperature,
            iterations=global_assignment_iterations,
            sample_weights=sample_weights if weighted_assignments else None,
        )
    if second_target_logits is not None:
        second_assignments = balanced_assignments(
            assignment_second_logits,
            temperature=assignment_temperature,
            iterations=global_assignment_iterations,
            sample_weights=sample_weights if weighted_assignments else None,
        )

    def assignment_diagnostics(logits: torch.Tensor):
        probs = logits.detach().softmax(dim=-1).clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        max_probability = probs.max(dim=-1).values.mean()
        active_classes = logits.argmax(dim=-1).unique().numel()
        return entropy, max_probability, logits.new_tensor(float(active_classes))

    first_entropy, first_max_probability, first_active_classes = assignment_diagnostics(
        first_joint_logits
    )
    second_entropy, second_max_probability, second_active_classes = assignment_diagnostics(
        second_joint_logits
    )
    consistency = novel_consistency_loss(
        first_joint_logits,
        second_joint_logits,
        confidence_threshold=confidence_threshold,
        temperature=assignment_temperature,
        sample_weights=sample_weights,
        weighted_assignments=weighted_assignments,
        first_assignments=first_assignments,
        second_assignments=second_assignments,
    )
    balance = 0.5 * (
        balanced_assignment_loss(first_joint_logits, sample_weights=sample_weights)
        + balanced_assignment_loss(second_joint_logits, sample_weights=sample_weights)
    )
    information = 0.5 * (
        information_maximization_loss(first_joint_logits, sample_weights=sample_weights)
        + information_maximization_loss(second_joint_logits, sample_weights=sample_weights)
    )
    neighbor = 0.5 * (
        neighbor_consistency_loss(
            first_features, first_joint_logits, k=neighbor_k, sample_weights=sample_weights
        )
        + neighbor_consistency_loss(
            second_features, second_joint_logits, k=neighbor_k, sample_weights=sample_weights
        )
    )
    memory_neighbor = 0.5 * (
        memory_neighbor_consistency_loss(
            first_features, first_joint_logits, memory_features, memory_logits,
            k=memory_neighbor_k, temperature=memory_temperature,
            sample_weights=sample_weights,
        )
        + memory_neighbor_consistency_loss(
            second_features, second_joint_logits, memory_features, memory_logits,
            k=memory_neighbor_k, temperature=memory_temperature,
            sample_weights=sample_weights,
        )
    )
    pseudo = prototype_pseudo_label_loss(
        first_joint_logits,
        second_joint_logits,
        confidence_threshold=confidence_threshold,
        assignment_temperature=assignment_temperature,
        sample_weights=sample_weights,
        weighted_assignments=weighted_assignments,
        first_assignments=first_assignments,
        second_assignments=second_assignments,
        target_mode=pseudo_target_mode,
    )
    total = (
        alpha_consistency * consistency
        + alpha_balance * balance
        + alpha_information * information
        + alpha_neighbor * neighbor
        + float(alpha_memory_neighbor) * memory_neighbor
        + alpha_pseudo * pseudo
    )
    return {
        "total": total,
        "consistency": consistency,
        "balance": balance,
        "information": information,
        "neighbor": neighbor,
        "memory_neighbor": memory_neighbor,
        "pseudo": pseudo,
        "assignment_entropy": 0.5 * (first_entropy + second_entropy),
        "assignment_max_probability": 0.5 * (
            first_max_probability + second_max_probability
        ),
        "assignment_active_classes": 0.5 * (
            first_active_classes + second_active_classes
        ),
        "global_assignment_entropy": (
            0.5
            * (
                -(first_assignments * first_assignments.clamp_min(1e-8).log()).sum(dim=-1).mean()
                + -(second_assignments * second_assignments.clamp_min(1e-8).log()).sum(dim=-1).mean()
            )
            if first_assignments is not None
            else first_joint_logits.new_tensor(0.0)
        ),
        "global_assignment_max_probability": (
            0.5
            * (first_assignments.max(dim=-1).values.mean() + second_assignments.max(dim=-1).values.mean())
            if first_assignments is not None
            else first_joint_logits.new_tensor(0.0)
        ),
        "global_assignment_active_classes": (
            first_joint_logits.new_tensor(
                0.5
                * (
                    first_assignments.argmax(dim=-1).unique().numel()
                    + second_assignments.argmax(dim=-1).unique().numel()
                )
            )
            if first_assignments is not None
            else first_joint_logits.new_tensor(0.0)
        ),
    }
    if first_target_logits is not None:
        first_assignments = balanced_assignments(
            assignment_first_logits,
            temperature=assignment_temperature,
            iterations=global_assignment_iterations,
            sample_weights=sample_weights if weighted_assignments else None,
        )
    if second_target_logits is not None:
        second_assignments = balanced_assignments(
            assignment_second_logits,
            temperature=assignment_temperature,
            iterations=global_assignment_iterations,
            sample_weights=sample_weights if weighted_assignments else None,
        )
