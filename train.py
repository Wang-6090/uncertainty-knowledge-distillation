from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

# Keep downloaded torchvision weights inside the project by default.
os.environ.setdefault("TORCH_HOME", str(Path.cwd() / ".torch_cache"))

from novel_discovery.data import (
    TwoViewDataset,
    build_data_bundle,
    build_outlier_dataset,
    build_transforms,
)
from novel_discovery.joint_discovery import NovelPrototypeHead, combine_known_novel_logits
from novel_discovery.metrics import compute_auroc
from novel_discovery.models import build_model
from novel_discovery.pipeline import (
    build_loader,
    calibrate_threshold,
    calibrate_class_thresholds,
    calibrate_open_threshold,
    calibrate_coverage_threshold,
    collect_diagonal_gaussian_stats,
    collect_prototypes,
    collect_classwise_knn_support,
    fit_score_normalization,
    evaluate_classification,
    extract_outputs,
    compute_open_score,
    compute_predicted_classwise_mahalanobis,
    select_discovery_candidates,
    apply_temperature,
    calibration_diagnostics,
    fit_temperature,
    run_discovery,
    collect_activation_clip_value,
    fit_feature_rejector,
    fit_virtual_outlier_rejector,
    fit_known_support_rejector,
    attach_known_support_score,
    fit_pu_feature_rejector,
    fit_nnpu_feature_rejector,
    attach_feature_rejector_score,
    collect_knn_feature_bank,
    attach_knn_distances,
    purify_candidate_mask,
    evaluate_candidate_purification,
    collect_vim_stats,
    attach_vim_residual,
    train_one_epoch_student,
    train_one_epoch_teacher,
    uncertainty_error_diagnostics,
)
from novel_discovery.utils import ensure_dir, save_json, set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagefolder", "toy", "fake"])
        p.add_argument("--data-root", default="./data")
        p.add_argument("--split-path", default="./splits.json")
        p.add_argument("--num-known", type=int, default=60)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--image-size", type=int, default=224)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--backbone", default="resnet18")
        p.add_argument("--teacher-backbone", default=None)
        p.add_argument("--student-backbone", default=None)
        p.add_argument("--pretrained", action="store_true")
        p.add_argument(
            "--encoder-lr-scale",
            type=float,
            default=1.0,
            help=(
                "Multiply the learning rate of the student encoder only. "
                "Values below 1 preserve pretrained features during fine-tuning; "
                "default 1.0 keeps the historical behavior."
            ),
        )
        p.add_argument("--proj-dim", type=int, default=128)
        p.add_argument("--dropout", type=float, default=0.2)
        p.add_argument(
            "--cifar-stem",
            action="store_true",
            help=(
                "Use a CIFAR-style 3x3/stride-1 ResNet stem and remove the "
                "initial max-pool. Disabled by default for historical compatibility."
            ),
        )
        p.add_argument(
            "--freeze-bn-stats",
            action="store_true",
            help=(
                "Keep BatchNorm running statistics fixed during training while "
                "still training BatchNorm affine parameters. Disabled by default."
            ),
        )
        p.add_argument("--download", action="store_true")
        p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
        p.add_argument("--work-dir", default="./runs")
        p.add_argument("--limit-train", type=int, default=0)
        p.add_argument("--limit-val", type=int, default=0)
        p.add_argument("--limit-test", type=int, default=0)
        p.add_argument("--limit-discovery", type=int, default=0)
        p.add_argument(
            "--calibration-ratio",
            type=float,
            default=0.0,
            help=(
                "Reserve this fraction of the limited known validation set for "
                "threshold calibration; the remainder is used for checkpoint selection. "
                "Use the same value in training and discover commands."
            ),
        )
        p.add_argument(
            "--calibration-overlap-control",
            action="store_true",
            help=(
                "Diagnostic only: keep the calibration subset the same size, "
                "but sample it from the model-selection subset to measure the "
                "effect of validation overlap. Do not use as a final protocol."
            ),
        )
        p.add_argument("--discovery-pool-mode", choices=["unknown", "mixed"], default="unknown")
        p.add_argument(
            "--rejector-pool-selection",
            choices=["all", "high_risk"],
            default="all",
            help="Use all discovery samples or only high-risk pseudo-unknowns to fit feature_rejector.",
        )
        p.add_argument("--rejector-candidate-ratio", type=float, default=0.25)
        p.add_argument(
            "--rejector-training",
            choices=["hard", "soft_pu", "nnpu"],
            default="hard",
            help="Hard binary rejector, heuristic soft-PU, or non-negative PU risk on the discovery pool.",
        )
        p.add_argument("--rejector-pu-iterations", type=int, default=4)
        p.add_argument(
            "--rejector-known-prior",
            type=float,
            default=0.2,
            help="Estimated known fraction in a mixed unlabeled pool for nnPU training.",
        )
        p.add_argument(
            "--rejector-feature-mode",
            choices=[
                "embedding",
                "augmented",
                "support_augmented",
                "uncertainty_augmented",
                "support_uncertainty_augmented",
            ],
            default="embedding",
            help=(
                "Use frozen embeddings alone, append classifier summaries, or also "
                "append MC-dropout epistemic/entropy signals."
            ),
        )
        p.add_argument(
            "--rejector-mc-samples",
            type=int,
            default=1,
            help="MC-dropout passes used to build rejector training features; 1 disables MC features.",
        )
        p.add_argument(
            "--rejector-model",
            choices=["logistic", "mlp"],
            default="logistic",
            help="Linear or small nonlinear classifier for the frozen-feature rejector.",
        )
        p.add_argument(
            "--rejector-strict-mixed",
            action="store_true",
            help=(
                "For mixed discovery, train the rejector's known side from the "
                "reserved open-validation known subset, requiring --open-val-ratio > 0."
            ),
        )
        p.add_argument(
            "--rejector-candidate-mode",
            choices=["entropy", "max_softmax", "energy", "entropy_uncertainty", "consensus"],
            default="entropy_uncertainty",
        )
        p.add_argument(
            "--known-split-mode",
            choices=["random", "stratified"],
            default="random",
            help="How to split known training samples into train/validation; random preserves the historical baseline.",
        )
        p.add_argument(
            "--mixed-known-pool-ratio",
            type=float,
            default=0.2,
            help=(
                "Fraction of the known training partition reserved as disjoint "
                "unlabeled samples when --discovery-pool-mode=mixed."
            ),
        )

    p = sub.add_parser("train_teacher")
    add_common(p)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha-unc", type=float, default=0.1)
    p.add_argument("--alpha-proto", type=float, default=0.0)
    p.add_argument(
        "--alpha-proto-repulsion", type=float, default=0.0,
        help="Weight a cosine-margin regularizer that separates class prototypes.",
    )
    p.add_argument("--proto-repulsion-margin", type=float, default=0.0)
    p.add_argument("--alpha-proxy", type=float, default=0.0)
    p.add_argument("--proxy-temperature", type=float, default=0.1)
    p.add_argument("--alpha-proxy-anchor", type=float, default=0.0)
    p.add_argument("--proxy-anchor-alpha", type=float, default=32.0)
    p.add_argument("--proxy-anchor-margin", type=float, default=0.1)
    p.add_argument("--alpha-pseudo", type=float, default=0.0)
    p.add_argument("--alpha-energy", type=float, default=0.0)
    p.add_argument(
        "--alpha-angular", type=float, default=0.0,
        help="Weight an ArcFace-style additive angular-margin loss for known classes.",
    )
    p.add_argument("--angular-margin", type=float, default=0.2)
    p.add_argument("--angular-scale", type=float, default=16.0)
    p.add_argument("--energy-margin", type=float, default=1.0)
    p.add_argument("--energy-temperature", type=float, default=1.0)
    # Keep the validated baseline as the default; ``strong`` remains an
    # explicit experimental option and is documented by its own run.
    p.add_argument("--pseudo-mode", choices=["legacy", "strong"], default="legacy")
    p.add_argument("--pseudo-feature-noise", type=float, default=0.05)
    p.add_argument("--uncertainty-target-mode", choices=["confidence", "classification_error", "margin"], default="confidence")
    p.add_argument("--teacher-ckpt", default="./runs/teacher.pt")

    p = sub.add_parser("train_student")
    add_common(p)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha-unc", type=float, default=0.1)
    p.add_argument("--alpha-kd", type=float, default=1.0)
    p.add_argument("--alpha-feat-kd", type=float, default=0.0)
    p.add_argument("--kd-mode", choices=["standard", "uncertainty"], default="uncertainty")
    p.add_argument("--alpha-supcon", type=float, default=0.1)
    p.add_argument(
        "--alpha-raw-supcon",
        type=float,
        default=0.0,
        help="Additional supervised contrastive loss directly on backbone features.",
    )
    p.add_argument("--alpha-proto", type=float, default=0.0)
    p.add_argument(
        "--alpha-proto-repulsion", type=float, default=0.0,
        help="Weight a cosine-margin regularizer that separates class prototypes.",
    )
    p.add_argument("--proto-repulsion-margin", type=float, default=0.0)
    p.add_argument("--alpha-proxy", type=float, default=0.0)
    p.add_argument("--proxy-temperature", type=float, default=0.1)
    p.add_argument("--alpha-proxy-anchor", type=float, default=0.0)
    p.add_argument("--proxy-anchor-alpha", type=float, default=32.0)
    p.add_argument("--proxy-anchor-margin", type=float, default=0.1)
    p.add_argument(
        "--reciprocal-points", type=int, default=0,
        help="Number of learnable ARPL-inspired reciprocal points (0 disables them).",
    )
    p.add_argument("--alpha-reciprocal", type=float, default=0.0)
    p.add_argument("--reciprocal-margin", type=float, default=0.2)
    p.add_argument("--alpha-pseudo", type=float, default=0.0)
    p.add_argument("--alpha-energy", type=float, default=0.0)
    p.add_argument(
        "--alpha-vos", type=float, default=0.0,
        help="Weight VOS-inspired virtual-outlier Energy/uniform separation during student training.",
    )
    p.add_argument(
        "--vos-mode", choices=["gaussian", "batch_center"], default="gaussian",
        help="Virtual-outlier generator: class-conditional Gaussian tails or online batch centers.",
    )
    p.add_argument(
        "--vos-tail-scale", type=float, default=1.5,
        help="Radial scale of the synthetic low-density feature direction.",
    )
    p.add_argument(
        "--vos-noise-scale", type=float, default=0.25,
        help="Orthogonal noise scale used for virtual outlier synthesis.",
    )
    p.add_argument(
        "--vos-uniform-weight", type=float, default=0.25,
        help="Relative uniform-logit weight inside the VOS loss.",
    )
    p.add_argument(
        "--alpha-angular", type=float, default=0.0,
        help="Weight an ArcFace-style additive angular-margin loss for known classes.",
    )
    p.add_argument("--angular-margin", type=float, default=0.2)
    p.add_argument("--angular-scale", type=float, default=16.0)
    p.add_argument("--energy-margin", type=float, default=1.0)
    p.add_argument("--energy-temperature", type=float, default=1.0)
    p.add_argument("--pseudo-mode", choices=["legacy", "strong"], default="legacy")
    p.add_argument("--pseudo-feature-noise", type=float, default=0.05)
    p.add_argument("--uncertainty-target-mode", choices=["confidence", "classification_error", "margin"], default="confidence")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--uncertainty-weight-mode", choices=["raw", "mean_normalized"], default="raw")
    p.add_argument(
        "--kd-uncertainty-source",
        choices=["head", "mc_epistemic", "mc_predictive_entropy"],
        default="head",
        help=(
            "Uncertainty used to weight KD: the learned teacher head, or "
            "MC-Dropout epistemic/predictive entropy."
        ),
    )
    p.add_argument(
        "--kd-mc-samples",
        type=int,
        default=4,
        help="Number of stochastic teacher passes for MC uncertainty KD.",
    )
    p.add_argument("--uncertainty-weight-min", type=float, default=None)
    p.add_argument("--uncertainty-weight-max", type=float, default=None)
    p.add_argument("--teacher-ckpt", default="./runs/teacher.pt")
    p.add_argument("--student-ckpt", default="./runs/student.pt")
    p.add_argument(
        "--discovery-pool",
        action="store_true",
        help="Use unlabeled novel-class training images for two-view consistency learning.",
    )
    p.add_argument("--alpha-discovery", type=float, default=0.0)
    p.add_argument("--alpha-discovery-unknown", type=float, default=0.0)
    p.add_argument(
        "--alpha-discovery-energy",
        type=float,
        default=0.0,
        help="Apply energy-margin separation between known samples and a pure unknown discovery pool.",
    )
    p.add_argument(
        "--alpha-discovery-uniform",
        type=float,
        default=0.0,
        help="Apply Outlier Exposure-style uniform known-class logits to a pure unknown discovery pool.",
    )
    p.add_argument(
        "--discovery-uniform-warmup-epochs",
        type=int,
        default=0,
        help="Keep uniform-logit Outlier Exposure disabled for the first N student epochs.",
    )
    p.add_argument(
        "--discovery-uniform-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp uniform-logit Outlier Exposure after its warmup period.",
    )
    p.add_argument(
        "--alpha-discovery-feature-margin",
        type=float,
        default=0.0,
        help="Push pure-unknown discovery features away from the nearest known classifier prototype.",
    )
    p.add_argument(
        "--discovery-feature-margin",
        type=float,
        default=0.2,
        help="Maximum cosine similarity allowed between a discovery feature and its nearest known prototype.",
    )
    p.add_argument(
        "--alpha-discovery-feature-separation",
        type=float,
        default=0.0,
        help="Push pure-unknown features away from the current known feature batch.",
    )
    p.add_argument(
        "--discovery-feature-separation-margin",
        type=float,
        default=0.0,
        help="Maximum smooth similarity allowed between unknown and observed known features.",
    )
    p.add_argument(
        "--discovery-feature-separation-temperature",
        type=float,
        default=0.1,
        help="Temperature for the smooth maximum over known-feature similarities.",
    )
    p.add_argument(
        "--discovery-feature-candidate-gating",
        action="store_true",
        help=(
            "In mixed discovery pools, apply feature margin/separation/boundary "
            "losses only to high-risk candidates selected by discovery-select-ratio/mode."
        ),
    )
    p.add_argument(
        "--alpha-discovery-boundary",
        type=float,
        default=0.0,
        help="Weight a hybrid known-prototype/local-centroid boundary loss for pure-unknown discovery features.",
    )
    p.add_argument(
        "--discovery-boundary-margin",
        type=float,
        default=0.2,
        help="Maximum hybrid cosine similarity allowed for pure-unknown features.",
    )
    p.add_argument(
        "--discovery-boundary-prototype-weight",
        type=float,
        default=0.5,
        help="Weight of global classifier prototypes versus current known-batch centroids.",
    )
    p.add_argument(
        "--discovery-boundary-temperature",
        type=float,
        default=0.1,
        help="Temperature for the local-centroid smooth maximum.",
    )
    p.add_argument(
        "--alpha-discovery-knn-boundary",
        type=float,
        default=0.0,
        help="Weight a class-conditional kNN support-boundary loss for a pure-unknown discovery pool.",
    )
    p.add_argument(
        "--discovery-knn-k",
        type=int,
        default=5,
        help="Number of same-class neighbors used in support distances.",
    )
    p.add_argument(
        "--discovery-knn-quantile",
        type=float,
        default=0.95,
        help="Known-training leave-one-out distance quantile defining each class support radius.",
    )
    p.add_argument(
        "--discovery-knn-margin",
        type=float,
        default=0.02,
        help="Extra cosine-distance margin beyond each known-class support radius.",
    )
    p.add_argument(
        "--discovery-knn-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Number of initial student epochs with the KNN support-boundary "
            "loss disabled; useful when early features are unstable."
        ),
    )
    p.add_argument(
        "--discovery-knn-ramp-epochs",
        type=int,
        default=0,
        help=(
            "Linearly ramp the KNN support-boundary loss after warm-up "
            "instead of applying its full weight immediately."
        ),
    )
    p.add_argument(
        "--alpha-discovery-uncertainty-separation",
        type=float,
        default=0.0,
        help="Train the uncertainty head to separate known samples from a pure-unknown discovery pool.",
    )
    p.add_argument(
        "--alpha-discovery-uncertainty-pu",
        type=float,
        default=0.0,
        help="Weight non-negative PU uncertainty training on known labels and a mixed unlabeled discovery pool.",
    )
    p.add_argument(
        "--discovery-uncertainty-known-prior",
        type=float,
        default=0.2,
        help="Assumed known proportion in the mixed discovery pool for PU uncertainty training.",
    )
    p.add_argument(
        "--discovery-uncertainty-loss",
        choices=["bce", "ranking"],
        default="bce",
        help="Objective for known/unknown uncertainty separation.",
    )
    p.add_argument(
        "--discovery-uncertainty-margin",
        type=float,
        default=0.1,
        help="Pairwise margin for ranking-based uncertainty separation.",
    )
    p.add_argument(
        "--uncertainty-separation-warmup-epochs",
        type=int,
        default=0,
        help="Keep uncertainty separation disabled for the first N student epochs.",
    )
    p.add_argument(
        "--uncertainty-separation-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp uncertainty separation after its warmup period.",
    )
    p.add_argument(
        "--alpha-discovery-selective-unknown",
        type=float,
        default=0.0,
        help="Apply unknown loss only to high-risk samples selected from an unlabeled discovery pool.",
    )
    p.add_argument(
        "--alpha-discovery-selective-energy",
        type=float,
        default=0.0,
        help="Apply energy separation only to high-risk samples selected from an unlabeled discovery pool.",
    )
    p.add_argument("--discovery-select-ratio", type=float, default=0.25)
    p.add_argument(
        "--discovery-select-mode",
        choices=["entropy", "max_softmax", "energy", "entropy_uncertainty", "consensus"],
        default="entropy_uncertainty",
    )
    p.add_argument(
        "--discovery-selective-warmup-epochs",
        type=int,
        default=0,
        help="Keep selective discovery unknown/energy losses disabled for the first N epochs.",
    )
    p.add_argument(
        "--discovery-selective-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp selective discovery unknown/energy loss weights after warmup.",
    )
    p.add_argument(
        "--discovery-selection-model",
        choices=["student", "ema"],
        default="student",
        help="Model used to select high-risk discovery samples; ema follows Mean Teacher-style smoothing.",
    )
    p.add_argument(
        "--discovery-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay for discovery-selection model when --discovery-selection-model ema is used.",
    )
    p.add_argument(
        "--discovery-neighbor-filter",
        action="store_true",
        help="Filter selected discovery candidates by kNN agreement in projection space.",
    )
    p.add_argument(
        "--discovery-neighbor-k",
        type=int,
        default=5,
        help="Number of nearest neighbors used by --discovery-neighbor-filter.",
    )
    p.add_argument(
        "--discovery-neighbor-min-votes",
        type=int,
        default=2,
        help="Minimum selected neighbors required to keep a discovery candidate.",
    )
    p.add_argument(
        "--discovery-soft-weighting",
        action="store_true",
        help="Use continuous candidate weights instead of equal-weight selective discovery loss.",
    )
    p.add_argument(
        "--discovery-neighbor-temperature",
        type=float,
        default=0.5,
        help="Temperature for smoothing kNN agreement in soft candidate weighting.",
    )
    p.add_argument(
        "--joint-discovery",
        action="store_true",
        help="Enable the prototype-based joint novel-class discovery objective.",
    )
    p.add_argument("--joint-num-novel", type=int, default=40)
    p.add_argument(
        "--joint-prototype-init",
        choices=["random", "kmeans", "kmeans_candidates"],
        default="random",
        help=(
            "Initialize novel prototypes randomly, from the full discovery pool, "
            "or from its high-risk candidate subset."
        ),
    )
    p.add_argument(
        "--joint-prototype-candidate-ratio",
        type=float,
        default=0.25,
        help="Top-risk fraction used by kmeans_candidates prototype initialization.",
    )
    p.add_argument(
        "--joint-prototype-warmup-epochs",
        type=int,
        default=0,
        help="Known-only warmup epochs before KMeans prototype initialization and joint loss.",
    )
    p.add_argument(
        "--joint-space",
        choices=["novel", "unified"],
        default="novel",
        help="Train only novel prototypes or a unified known-plus-novel class space.",
    )
    p.add_argument("--alpha-joint-discovery", type=float, default=1.0)
    p.add_argument(
        "--joint-head-temperature",
        type=float,
        default=0.2,
        help="Temperature for cosine novel-prototype logits; lower values sharpen assignments.",
    )
    p.add_argument(
        "--joint-confidence-threshold",
        type=float,
        default=1.1,
        help="Relative novel confidence: max softmax probability multiplied by num novel classes.",
    )
    p.add_argument("--joint-assignment-temperature", type=float, default=1.0)
    p.add_argument("--alpha-joint-consistency", type=float, default=1.0)
    p.add_argument("--alpha-joint-balance", type=float, default=0.1)
    p.add_argument("--alpha-joint-information", type=float, default=0.1)
    p.add_argument("--alpha-joint-neighbor", type=float, default=0.1)
    p.add_argument(
        "--alpha-joint-pseudo",
        type=float,
        default=0.0,
        help="Weight cross-view hard pseudo-label learning for novel prototypes.",
    )
    p.add_argument(
        "--alpha-joint-gate",
        type=float,
        default=0.0,
        help="Weight independent uncertainty ranking for selected mixed-pool candidates.",
    )
    p.add_argument("--joint-gate-margin", type=float, default=0.1)
    p.add_argument(
        "--alpha-joint-proto-repulsion",
        type=float,
        default=0.0,
        help="Separate learnable novel prototypes with an optional cosine-margin regularizer.",
    )
    p.add_argument("--joint-proto-repulsion-margin", type=float, default=0.0)
    p.add_argument("--joint-neighbor-k", type=int, default=5)
    p.add_argument("--alpha-joint-known-ce", type=float, default=0.1)
    p.add_argument("--joint-known-temperature", type=float, default=1.0)
    p.add_argument(
        "--joint-mixed-residual",
        action="store_true",
        help=(
            "On a mixed discovery pool, weight novel-prototype learning by "
            "1-max known-class probability instead of balancing known and "
            "novel logits in one Sinkhorn space."
        ),
    )
    p.add_argument(
        "--joint-residual-temperature",
        type=float,
        default=1.0,
        help="Temperature used to compute mixed-pool known residual weights.",
    )
    p.add_argument(
        "--joint-residual-floor",
        type=float,
        default=0.0,
        help="Minimum residual weight retained for mixed-pool novel learning.",
    )
    p.add_argument(
        "--joint-novel-mass",
        action="store_true",
        help=(
            "Use the unified known-plus-novel probability mass, with two-view "
            "agreement, to weight novel learning on a mixed discovery pool."
        ),
    )
    p.add_argument(
        "--joint-novel-neighbor-support",
        action="store_true",
        help=(
            "Multiply cross-view novel-mass weights by local kNN support in "
            "the mixed-pool feature batch. Requires --joint-novel-mass."
        ),
    )
    p.add_argument("--joint-novel-neighbor-k", type=int, default=5)
    p.add_argument(
        "--joint-weighted-sinkhorn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use detached novel-mass sample weights as Sinkhorn sample marginals; "
            "loss weighting remains active when disabled."
        ),
    )
    p.add_argument(
        "--joint-novel-ema-weights",
        action="store_true",
        help=(
            "Generate joint novel-mass and neighbor-support weights from the "
            "EMA selection model; requires --discovery-selection-model ema."
        ),
    )
    p.add_argument(
        "--alpha-joint-novel-margin",
        type=float,
        default=0.0,
        help="Weight soft feature repulsion for high-novel-mass mixed-pool samples.",
    )
    p.add_argument(
        "--joint-novel-margin",
        type=float,
        default=0.2,
        help="Maximum cosine similarity to a known prototype for soft novel repulsion.",
    )
    p.add_argument(
        "--joint-candidate-gating",
        action="store_true",
        help="Apply the joint novel objective only to high-risk unlabeled discovery candidates.",
    )
    p.add_argument("--joint-candidate-ratio", type=float, default=0.25)
    p.add_argument(
        "--joint-candidate-mode",
        choices=[
            "entropy",
            "max_softmax",
            "energy",
            "entropy_uncertainty",
            "consensus",
            "novel_mass",
            "prototype_distance",
            "distance_consensus",
        ],
        default="consensus",
    )
    p.add_argument("--joint-candidate-neighbor-filter", action="store_true")
    p.add_argument("--joint-candidate-neighbor-k", type=int, default=5)
    p.add_argument("--joint-candidate-min-votes", type=int, default=2)
    p.add_argument("--joint-candidate-soft-weighting", action="store_true")
    p.add_argument("--joint-candidate-weight-floor", type=float, default=0.05)
    p.add_argument(
        "--joint-mixed-known-consistency",
        action="store_true",
        help=(
            "On a mixed unlabeled pool, train low-risk, cross-view-consistent "
            "samples toward EMA known-class pseudo labels. Requires joint "
            "candidate gating and mixed discovery mode."
        ),
    )
    p.add_argument(
        "--alpha-joint-known-consistency",
        type=float,
        default=0.0,
        help="Weight the mixed-pool known-anchor pseudo-label consistency loss.",
    )
    p.add_argument(
        "--joint-known-confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum EMA confidence for a low-risk mixed-pool known anchor.",
    )
    p.add_argument(
        "--joint-known-target",
        choices=["ema", "teacher"],
        default="ema",
        help="Detached model used to assign labels for mixed-pool known anchors.",
    )
    p.add_argument(
        "--outlier-dataset",
        choices=["cifar10", "imagefolder"],
        default="",
        help="Optional unlabeled auxiliary dataset for Outlier Exposure.",
    )
    p.add_argument("--outlier-data-root", default="")
    p.add_argument("--outlier-download", action="store_true")
    p.add_argument("--outlier-batch-size", type=int, default=0)
    p.add_argument("--alpha-outlier-uniform", type=float, default=0.0)
    p.add_argument("--alpha-outlier-energy", type=float, default=0.0)
    p.add_argument("--alpha-outlier-uncertainty", type=float, default=0.0)
    p.add_argument(
        "--alpha-outlier-feature-margin", type=float, default=0.0,
        help="Push auxiliary outlier features away from the nearest known classifier prototype.",
    )
    p.add_argument("--outlier-feature-margin", type=float, default=0.2)
    p.add_argument("--discovery-batch-size", type=int, default=0)
    p.add_argument("--discovery-loss", choices=["consistency", "nt_xent"], default="nt_xent")
    p.add_argument("--discovery-temperature", type=float, default=0.2)

    p = sub.add_parser("discover")
    add_common(p)
    p.add_argument("--student-ckpt", default="./runs/student.pt")
    p.add_argument(
        "--novel-head-ckpt",
        default="",
        help="Optional novel_head.pt from joint discovery training; enables novel/novel_pca clustering features.",
    )
    p.add_argument(
        "--novel-known-temperature",
        type=float,
        default=None,
        help="Override the saved known-logit temperature when scoring a unified novel head.",
    )
    p.add_argument("--num-novel", type=int, default=40)
    p.add_argument("--cluster-k", choices=["oracle", "auto"], default="auto")
    p.add_argument(
        "--cluster-method",
        choices=["kmeans", "agglomerative", "spectral"],
        default="kmeans",
        help="Clustering algorithm for the rejected/unknown candidate pool.",
    )
    p.add_argument(
        "--cluster-feature",
        choices=[
            "projection",
            "projection_pca",
            "feature",
            "feature_pca",
            "novel",
            "novel_pca",
            "unified",
            "unified_pca",
        ],
        default="projection_pca",
        help=(
            "Representation used for clustering; projection_pca is the default "
            "because it performed best in the current CIFAR-100 comparison."
        ),
    )
    p.add_argument(
        "--cluster-selection",
        choices=["silhouette", "composite", "stability", "gmm_bic"],
        default="silhouette",
        help="K selection criterion: silhouette, composite metrics, stability, or diagonal-GMM BIC.",
    )
    p.add_argument(
        "--max-auto-clusters",
        type=int,
        default=50,
        help="Upper bound for unlabeled auto-K search; independent of the evaluation-only --num-novel.",
    )
    p.add_argument("--cluster-pca-dim", type=int, default=32)
    p.add_argument(
        "--cluster-normalize",
        action="store_true",
        help="L2-normalize clustering features (default remains raw features for compatibility).",
    )
    p.add_argument("--cluster-no-whiten", action="store_true")
    p.add_argument("--cluster-n-init", type=int, default=10)
    p.add_argument("--cluster-stability-repeats", type=int, default=5)
    p.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering and report only open-set detection metrics.",
    )
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    p.add_argument(
        "--threshold-policy",
        choices=["global", "class_conditional", "known_coverage", "open_balanced", "open_f1", "open_known_coverage"],
        default="global",
        help="Use a known-only threshold, matched known coverage, class thresholds, or an open-validation operating point.",
    )
    p.add_argument(
        "--target-known-coverage",
        type=float,
        default=0.95,
        help="Target known acceptance rate for --threshold-policy known_coverage.",
    )
    p.add_argument("--threshold-min-class-samples", type=int, default=5)
    p.add_argument(
        "--candidate-purify",
        choices=["none", "open_score", "entropy", "head_uncertainty", "uncertainty_consensus", "knn_support", "open_knn_support"],
        default="none",
        help="Purify rejected candidates only before clustering; detection metrics are unchanged.",
    )
    p.add_argument("--candidate-keep-ratio", type=float, default=1.0)
    p.add_argument(
        "--support-quantile",
        type=float,
        default=0.95,
        help="Per-class known-feature distance quantile for known_support detection.",
    )
    p.add_argument("--mc-samples", type=int, default=8)
    p.add_argument("--temperature-calibration", action="store_true")
    p.add_argument("--calibration-bins", type=int, default=15)
    p.add_argument(
        "--score-mode",
        default="entropy_proto",
        choices=[
            "auto",
            "full",
            "max_softmax",
            "max_logit",
            "logit_margin",
            "novel_msp",
            "novel_entropy",
            "unified_novel_mass",
            "classwise_unified_novel_mass",
            "normalized_unified_novel_mass_entropy",
            "odin_msp",
            "energy",
            "react_energy",
            "knn_distance",
            "normalized_entropy_knn",
            "predicted_class_knn_distance",
            "normalized_entropy_predicted_class_knn",
            "normalized_entropy_predicted_class_knn_global",
            "normalized_entropy_relative_predicted_class_knn",
            "normalized_entropy_knn_conflict",
            "normalized_entropy_mahalanobis_knn",
            "gaussian_nll",
            "openmax",
            "reciprocal",
            "normalized_entropy_gaussian_nll",
            "relative_mahalanobis",
            "normalized_entropy_relative_mahalanobis",
            "vim_residual",
            "head_uncertainty",
            "feature_rejector",
            "virtual_rejector",
            "known_support",
            "margin_uncertainty",
            "entropy_only",
            "proto_only",
            "entropy_proto",
            "normalized_full",
            "normalized_entropy_proto",
            "entropy_epistemic",
            "entropy_aleatoric",
            "expected_entropy",
            "head_uncertainty",
            "margin_uncertainty",
            "epistemic",
            "mahalanobis",
            "mahalanobis_diag",
            "mahalanobis_shared",
            "entropy_mahalanobis",
            "entropy_mahalanobis_diag",
            "entropy_mahalanobis_shared",
            "normalized_entropy_mahalanobis",
            "normalized_entropy_mahalanobis_diag",
            "normalized_entropy_mahalanobis_shared",
            "classwise_mahalanobis",
            "normalized_entropy_classwise_mahalanobis",
        ],
    )
    p.add_argument(
        "--open-val-ratio",
        type=float,
        default=0.0,
        help="Reserve train-split known and novel samples for score selection; never splits the final test set.",
    )
    p.add_argument("--auto-calibrate-score", action="store_true")
    p.add_argument(
        "--auto-score-fast",
        action="store_true",
        help="Use only lightweight scores during auto calibration; skip Mahalanobis candidates.",
    )
    p.add_argument(
        "--odin-epsilon",
        type=float,
        default=0.0,
        help="Enable ODIN-style input perturbation for odin_msp score; value is in normalized image units.",
    )
    p.add_argument(
        "--odin-temperature",
        type=float,
        default=1000.0,
        help="Temperature used by ODIN-style MSP scoring.",
    )
    p.add_argument(
        "--react-percentile",
        type=float,
        default=0.0,
        help="Enable ReAct feature clipping using this known-training activation percentile.",
    )
    p.add_argument("--knn-ood", action="store_true", help="Enable KNN distance OOD scoring from known training projections.")
    p.add_argument("--knn-k", type=int, default=10, help="Number of known training neighbors for KNN-OOD.")
    p.add_argument("--knn-feature", choices=["features", "proj"], default="features", help="Embedding used by KNN-OOD: backbone feature or projection head.")
    p.add_argument("--knn-bank-size", type=int, default=0, help="Optional maximum known training vectors in the KNN bank; 0 uses all.")
    p.add_argument("--knn-support-quantile", type=float, default=0.95, help="Known-training leave-one-out quantile used for classwise kNN support radii.")
    p.add_argument("--vim-ood", action="store_true", help="Enable VIM-style principal-subspace residual scoring.")
    p.add_argument("--vim-rank", type=int, default=64, help="Known feature principal-subspace rank for VIM-OOD.")

    p = sub.add_parser("inspect_data")
    add_common(p)
    p.add_argument("--sample-count", type=int, default=5)
    return parser.parse_args(argv)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


@torch.no_grad()
def attach_novel_outputs(outputs, novel_head, device, known_temperature: float = 1.0):
    """Add novel-head logits/probabilities for clustering evaluation."""
    features = torch.as_tensor(outputs["features"], dtype=torch.float32, device=device)
    novel_logits = novel_head(features)
    novel_probs = novel_logits.softmax(dim=-1)
    outputs = dict(outputs)
    outputs["novel_logits"] = novel_logits.cpu().numpy()
    outputs["novel_probs"] = novel_probs.cpu().numpy()
    outputs["novel_entropy"] = (
        -(novel_probs.clamp_min(1e-8) * novel_probs.clamp_min(1e-8).log()).sum(dim=-1)
    ).cpu().numpy()
    known_logits = torch.as_tensor(outputs["logits"], dtype=torch.float32, device=device)
    unified_logits = combine_known_novel_logits(
        known_logits,
        novel_logits,
        known_temperature=known_temperature,
    )
    unified_probs = unified_logits.softmax(dim=-1)
    known_count = known_logits.size(-1)
    outputs["unified_probs"] = unified_probs.cpu().numpy()
    outputs["unified_novel_mass"] = unified_probs[:, known_count:].sum(dim=-1).cpu().numpy()
    return outputs


@torch.no_grad()
def initialize_novel_head_kmeans(
    novel_head,
    student,
    discovery_loader,
    device,
    num_novel: int,
    seed: int,
    candidate_ratio: float = 0.25,
    candidates_only: bool = False,
):
    """Initialize novel prototypes from discovery features without labels.

    For a mixed discovery pool, the candidate mode keeps only the highest-risk
    samples according to the known classifier before fitting KMeans. This avoids
    using the entire unlabeled pool as if it were novel data.
    """
    if discovery_loader is None:
        raise ValueError("KMeans prototype initialization requires a discovery pool.")
    student.eval()
    features = []
    risks = []
    for first_view, _ in discovery_loader:
        outputs = student(first_view.to(device))
        features.append(torch.nn.functional.normalize(outputs["features"], dim=-1).cpu().numpy())
        if candidates_only:
            probs = outputs["logits"].softmax(dim=-1).clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1)
            risk = entropy + outputs["uncertainty"]
            risks.append(risk.cpu().numpy())
    if not features:
        raise ValueError("Discovery pool is empty; cannot initialize novel prototypes.")
    feature_array = np.concatenate(features, axis=0)
    if candidates_only:
        risk_array = np.concatenate(risks, axis=0)
        ratio = min(max(float(candidate_ratio), 1e-3), 1.0)
        selected_count = max(num_novel, int(np.ceil(len(feature_array) * ratio)))
        selected_count = min(selected_count, len(feature_array))
        selected = np.argsort(risk_array)[-selected_count:]
        feature_array = feature_array[selected]
    if len(feature_array) < num_novel:
        raise ValueError(
            f"Need at least {num_novel} discovery samples for KMeans initialization, "
            f"got {len(feature_array)}."
        )
    clustering = KMeans(
        n_clusters=num_novel,
        n_init=10,
        random_state=int(seed),
    )
    clustering.fit(feature_array)
    centers = torch.as_tensor(clustering.cluster_centers_, dtype=torch.float32, device=device)
    centers = torch.nn.functional.normalize(centers, dim=-1)
    novel_head.prototypes.copy_(centers)
    student.train()
    return {
        "samples": int(len(feature_array)),
        "inertia": float(clustering.inertia_),
        "candidate_mode": bool(candidates_only),
        "candidate_ratio": float(candidate_ratio) if candidates_only else None,
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available in this Python environment, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def save_checkpoint(model, path, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_command_config(run_dir, command: str, config: dict) -> None:
    """Keep the legacy config.json while preserving each command's arguments."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)
    save_json(run_dir / f"{command}_config.json", config)


def scheduled_weight(epoch: int, warmup_epochs: int = 0, ramp_epochs: int = 0) -> float:
    """Return a loss multiplier using epoch-indexed warmup and linear ramp."""
    warmup_epochs = max(int(warmup_epochs), 0)
    ramp_epochs = max(int(ramp_epochs), 0)
    epoch_number = int(epoch) + 1
    if epoch_number <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, (epoch_number - warmup_epochs) / ramp_epochs))


def clone_dataset_with_eval_transform(dataset, image_size: int):
    """Shallow-clone Subset wrappers and use deterministic transforms on base data."""
    cloned = copy.copy(dataset)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        cloned.dataset = clone_dataset_with_eval_transform(dataset.dataset, image_size)
        return cloned
    if hasattr(dataset, "base"):
        cloned.base = copy.copy(dataset.base)
        cloned.base.transform = build_transforms(image_size, train=False)
        return cloned
    raise TypeError(
        "kNN support-bank refresh requires a dataset wrapper exposing a base dataset"
    )


def _checkpoint_parts(path_arg: str):
    path = Path(path_arg)
    return tuple(part for part in path.parts if part not in {"."})


def _is_default_checkpoint(path_arg: str, default_name: str) -> bool:
    parts = _checkpoint_parts(path_arg)
    return parts in {(default_name,), ("runs", default_name)}


def resolve_output_checkpoint(path_arg: str, work_dir: str, default_name: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if _is_default_checkpoint(path_arg, default_name):
        return Path(work_dir) / path.name
    return path


def resolve_input_checkpoint(path_arg: str, work_dir: str, default_name: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute() or not _is_default_checkpoint(path_arg, default_name):
        if path.exists():
            return path
        available = sorted(str(p) for p in Path(work_dir).glob("*.pt"))
        raise FileNotFoundError(
            f"Could not find checkpoint at {path}. Available .pt files in work dir: {available}"
        )

    candidates = [Path(work_dir) / default_name, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = sorted(str(p) for p in Path(work_dir).glob("*.pt"))
    raise FileNotFoundError(
        f"Could not find checkpoint. Checked: {', '.join(str(c) for c in candidates)}. Available .pt files in work dir: {available}"
    )


def _select_score_mode(
    outputs_known,
    outputs_open,
    prototypes,
    score_modes,
    normalization=None,
    gaussian_stats=None,
):
    unknown_mask = np.asarray(outputs_open["is_known"], dtype=bool) == 0
    results = []
    for score_mode in score_modes:
        try:
            known_scores, _ = compute_open_score(
                outputs_known,
                prototypes=prototypes,
                score_mode=score_mode,
                normalization=normalization,
                gaussian_stats=gaussian_stats,
            )
            open_scores, _ = compute_open_score(
                outputs_open,
                prototypes=prototypes,
                score_mode=score_mode,
                normalization=normalization,
                gaussian_stats=gaussian_stats,
            )
        except ValueError as exc:
            results.append(
                {
                    "score_mode": score_mode,
                    "auroc": float("nan"),
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            continue
        unknown_scores = open_scores[unknown_mask]
        labels = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
        scores = np.concatenate([known_scores, unknown_scores])
        auroc = compute_auroc(labels, scores)
        results.append(
            {
                "score_mode": score_mode,
                "auroc": auroc,
                "status": "ok",
            }
        )

    valid = [item for item in results if item["status"] == "ok" and np.isfinite(item["auroc"])]
    if not valid:
        raise ValueError("No usable score mode was available for automatic calibration.")
    valid.sort(key=lambda x: x["auroc"], reverse=True)
    return valid[0], results


def _score_needs_shared_mahalanobis(score_mode: str, auto_calibrate: bool = False) -> bool:
    if auto_calibrate or score_mode == "auto":
        return True
    return score_mode.endswith("_shared") or score_mode in {
        "mahalanobis",
        "entropy_mahalanobis",
        "normalized_entropy_mahalanobis",
        "normalized_entropy_mahalanobis_knn",
        "relative_mahalanobis",
        "normalized_entropy_relative_mahalanobis",
        "classwise_mahalanobis",
        "normalized_entropy_classwise_mahalanobis",
        "gaussian_nll",
        "normalized_entropy_gaussian_nll",
        "relative_mahalanobis",
        "normalized_entropy_relative_mahalanobis",
    }


def _calibrate_discovery_threshold(
    args,
    scores_val,
    outputs_val,
    scores_open_val,
    outputs_open_val,
    num_classes,
):
    if args.threshold_policy in {"open_balanced", "open_f1", "open_known_coverage"}:
        if scores_open_val is None or outputs_open_val is None:
            raise ValueError(
                "open threshold policies require --open-val-ratio greater than 0"
            )
        objective = {
            "open_f1": "unknown_f1",
            "open_known_coverage": "known_coverage",
        }.get(args.threshold_policy, "balanced_accuracy")
        threshold, diagnostics = calibrate_open_threshold(
            scores_open_val,
            np.asarray(outputs_open_val["is_known"], dtype=bool),
            objective=objective,
            target_known_coverage=args.target_known_coverage,
        )
        return threshold, {
            "type": f"open_val_{objective}",
            "percentile": None,
            "min_class_samples": None,
            **diagnostics,
        }
    if args.threshold_policy == "known_coverage":
        threshold, diagnostics = calibrate_coverage_threshold(
            scores_val,
            target_known_coverage=args.target_known_coverage,
        )
        return threshold, {
            "type": "known_val_known_coverage",
            "percentile": None,
            "min_class_samples": None,
            **diagnostics,
        }
    if args.threshold_policy == "class_conditional":
        threshold = calibrate_class_thresholds(
            scores_val,
            outputs_val["logits"].argmax(axis=1),
            num_classes,
            percentile=args.threshold_percentile,
            min_samples=args.threshold_min_class_samples,
        )
    else:
        threshold = calibrate_threshold(scores_val, percentile=args.threshold_percentile)
    return threshold, {
        "type": f"known_val_{args.threshold_policy}",
        "percentile": args.threshold_percentile,
        "min_class_samples": args.threshold_min_class_samples,
    }


def fit_teacher(args):
    set_seed(args.seed)
    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=args.download,
        split_path=args.split_path,
        limit_train=args.limit_train or None,
        limit_val=args.limit_val or None,
        limit_test=args.limit_test or None,
        limit_discovery=args.limit_discovery or None,
        discovery_pool_mode=args.discovery_pool_mode,
        calibration_ratio=args.calibration_ratio,
        calibration_overlap_control=args.calibration_overlap_control,
        known_split_mode=args.known_split_mode,
        mixed_known_pool_ratio=args.mixed_known_pool_ratio,
        open_val_ratio=getattr(args, "open_val_ratio", 0.0),
    )
    device = resolve_device(args.device)
    print(f"device: {device}")
    train_loader = build_loader(bundle.train, args.batch_size, True, args.num_workers)
    val_loader = build_loader(bundle.val, args.batch_size, False, args.num_workers)
    model = build_model(
        len(bundle.known_classes),
        backbone=args.teacher_backbone or args.backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
        cifar_stem=args.cifar_stem,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_acc = -1.0
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(args.epochs):
        stats = train_one_epoch_teacher(
            model,
            train_loader,
            optim,
            device,
            alpha_unc=args.alpha_unc,
            alpha_proto=args.alpha_proto,
            alpha_proto_repulsion=args.alpha_proto_repulsion,
            proto_repulsion_margin=args.proto_repulsion_margin,
            alpha_proxy=args.alpha_proxy,
            alpha_pseudo=args.alpha_pseudo,
            alpha_energy=args.alpha_energy,
            pseudo_mode=args.pseudo_mode,
            pseudo_feature_noise=args.pseudo_feature_noise,
            uncertainty_target_mode=args.uncertainty_target_mode,
            energy_margin=args.energy_margin,
            energy_temperature=args.energy_temperature,
            proxy_temperature=args.proxy_temperature,
            alpha_angular=args.alpha_angular,
            angular_margin=args.angular_margin,
            angular_scale=args.angular_scale,
            alpha_proxy_anchor=args.alpha_proxy_anchor,
            proxy_anchor_alpha=args.proxy_anchor_alpha,
            proxy_anchor_margin=args.proxy_anchor_margin,
            freeze_bn_stats=args.freeze_bn_stats,
        )
        val_stats = evaluate_classification(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train": stats, "validation": val_stats})
        print(f"[teacher][{epoch+1}/{args.epochs}] {stats} {val_stats}")
        if val_stats["known_acc"] > best_acc:
            best_acc = val_stats["known_acc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
    if best_state is not None:
        model.load_state_dict(best_state)
    run_dir = ensure_dir(args.work_dir)
    save_command_config(
        run_dir,
        args.command,
        {**vars(args), "best_epoch": best_epoch, "best_val_known_acc": best_acc},
    )
    save_json(run_dir / "train_history.json", history)
    proto_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
    prototypes = collect_prototypes(model, proto_loader, device, len(bundle.known_classes)).cpu()
    gaussian_stats = collect_diagonal_gaussian_stats(
        model, proto_loader, device, len(bundle.known_classes)
    )
    gaussian_stats = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in gaussian_stats.items()}
    ckpt_path = resolve_output_checkpoint(args.teacher_ckpt, args.work_dir, "teacher.pt")
    save_checkpoint(
        model,
        ckpt_path,
        extra={
            "known_classes": list(bundle.known_classes),
            "prototypes": prototypes,
            "gaussian_stats": gaussian_stats,
        },
    )
    torch.save({"known_classes": list(bundle.known_classes), "novel_classes": list(bundle.novel_classes)}, run_dir / "split.pt")
    print(f"saved to {ckpt_path} (best_epoch={best_epoch}, best_val_known_acc={best_acc:.4f})")


def fit_student(args):
    if args.joint_mixed_known_consistency or args.alpha_joint_known_consistency > 0.0:
        if (
            not args.joint_discovery
            or not args.joint_candidate_gating
            or args.alpha_joint_discovery <= 0.0
        ):
            raise ValueError(
                "mixed known consistency requires --joint-discovery and "
                "--joint-candidate-gating"
            )
        if args.discovery_pool_mode != "mixed":
            raise ValueError(
                "mixed known consistency requires --discovery-pool-mode mixed"
            )
        if not args.discovery_pool:
            raise ValueError(
                "mixed known consistency requires --discovery-pool"
            )
        if args.discovery_selection_model != "ema":
            raise ValueError(
                "mixed known consistency requires --discovery-selection-model ema"
            )
        if args.alpha_joint_known_consistency <= 0.0:
            raise ValueError(
                "--joint-mixed-known-consistency requires "
                "--alpha-joint-known-consistency > 0"
            )
        if not 0.0 < args.joint_known_confidence_threshold <= 1.0:
            raise ValueError("--joint-known-confidence-threshold must be in (0, 1]")
    if args.joint_novel_ema_weights and (
        not args.joint_novel_mass or args.discovery_selection_model != "ema"
    ):
        raise ValueError(
            "--joint-novel-ema-weights requires --joint-novel-mass and "
            "--discovery-selection-model ema"
        )
    if args.alpha_reciprocal > 0.0 and args.reciprocal_points <= 0:
        raise ValueError("--alpha-reciprocal requires --reciprocal-points greater than zero")
    if args.alpha_reciprocal > 0.0 and (
        not args.discovery_pool or args.discovery_pool_mode != "unknown"
    ):
        raise ValueError(
            "reciprocal-point discovery loss requires --discovery-pool with a pure unknown pool"
        )
    set_seed(args.seed)
    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=args.download,
        split_path=args.split_path,
        limit_train=args.limit_train or None,
        limit_val=args.limit_val or None,
        limit_test=args.limit_test or None,
        limit_discovery=args.limit_discovery or None,
        discovery_pool_mode=args.discovery_pool_mode,
        calibration_ratio=args.calibration_ratio,
        known_split_mode=args.known_split_mode,
        mixed_known_pool_ratio=args.mixed_known_pool_ratio,
        open_val_ratio=getattr(args, "open_val_ratio", 0.0),
    )
    device = resolve_device(args.device)
    print(f"device: {device}")
    train_loader = build_loader(bundle.train, args.batch_size, True, args.num_workers)
    val_loader = build_loader(bundle.val, args.batch_size, False, args.num_workers)
    discovery_loader = None
    outlier_loader = None
    uses_outlier_exposure = (
        args.alpha_outlier_uniform > 0.0
        or args.alpha_outlier_energy > 0.0
        or args.alpha_outlier_uncertainty > 0.0
        or args.alpha_outlier_feature_margin > 0.0
    )
    if uses_outlier_exposure:
        if not args.outlier_dataset or not args.outlier_data_root:
            raise ValueError(
                "Outlier Exposure requires --outlier-dataset and --outlier-data-root."
            )
        outlier_dataset = build_outlier_dataset(
            args.outlier_dataset,
            args.outlier_data_root,
            args.image_size,
            download=args.outlier_download,
        )
        outlier_loader = build_loader(
            outlier_dataset,
            args.outlier_batch_size or args.batch_size,
            True,
            args.num_workers,
        )
    uses_discovery_regularizer = (
        args.alpha_discovery_unknown > 0.0
        or args.alpha_discovery_energy > 0.0
        or args.alpha_discovery_uniform > 0.0
        or args.alpha_discovery_feature_margin > 0.0
        or args.alpha_discovery_feature_separation > 0.0
        or args.alpha_discovery_boundary > 0.0
        or args.alpha_discovery_knn_boundary > 0.0
        or args.alpha_discovery_uncertainty_separation > 0.0
        or args.alpha_discovery_uncertainty_pu > 0.0
        or args.alpha_discovery_selective_unknown > 0.0
        or args.alpha_discovery_selective_energy > 0.0
        or args.joint_discovery
    )
    if uses_discovery_regularizer and not args.discovery_pool:
        raise ValueError("Discovery regularizers require --discovery-pool.")
    if args.alpha_discovery_uncertainty_pu > 0.0:
        if args.discovery_pool_mode != "mixed":
            raise ValueError(
                "PU uncertainty training requires --discovery-pool-mode mixed; "
                "it must not treat a pure unknown pool as unlabeled mixture."
            )
        if not 0.0 < args.discovery_uncertainty_known_prior < 1.0:
            raise ValueError("--discovery-uncertainty-known-prior must be in (0, 1).")
    if args.discovery_pool and bundle.discovery_pool is not None and len(bundle.discovery_pool) > 0:
        if args.discovery_pool_mode == "mixed" and (
            args.alpha_discovery_unknown > 0.0
            or args.alpha_discovery_energy > 0.0
            or args.alpha_discovery_uniform > 0.0
            or (
                (
                    args.alpha_discovery_feature_margin > 0.0
                    or args.alpha_discovery_feature_separation > 0.0
                    or args.alpha_discovery_boundary > 0.0
                )
                and not args.discovery_feature_candidate_gating
            )
            or (
                args.alpha_discovery_knn_boundary > 0.0
                and not args.discovery_feature_candidate_gating
            )
            or args.alpha_discovery_uncertainty_separation > 0.0
        ):
            raise ValueError(
                "full-pool discovery unknown/energy/feature-boundary losses require "
                "--discovery-pool-mode unknown; use --discovery-feature-candidate-gating "
                "for feature losses on a mixed pool."
            )
        discovery_loader = build_loader(
            TwoViewDataset(bundle.discovery_pool),
            args.discovery_batch_size or args.batch_size,
            True,
            args.num_workers,
        )
    teacher_backbone = args.teacher_backbone or args.backbone
    student_backbone = args.student_backbone or args.backbone
    teacher = build_model(
        len(bundle.known_classes),
        backbone=teacher_backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
        cifar_stem=args.cifar_stem,
    ).to(device)
    student = build_model(
        len(bundle.known_classes),
        backbone=student_backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
        cifar_stem=args.cifar_stem,
    ).to(device)
    knn_support_loader = None
    if args.alpha_discovery_knn_boundary > 0.0:
        if args.discovery_knn_k <= 0:
            raise ValueError("--discovery-knn-k must be positive")
        if not 0.5 <= args.discovery_knn_quantile < 1.0:
            raise ValueError("--discovery-knn-quantile must be in [0.5, 1.0)")
        if args.discovery_knn_margin < 0.0:
            raise ValueError("--discovery-knn-margin must be non-negative")
        deterministic_known_train = clone_dataset_with_eval_transform(
            bundle.train, args.image_size
        )
        knn_support_loader = build_loader(
            deterministic_known_train, args.batch_size, False, args.num_workers
        )
    novel_head = None
    if args.joint_discovery:
        novel_head = NovelPrototypeHead(
            student.encoder.out_dim,
            args.joint_num_novel,
            temperature=args.joint_head_temperature,
        ).to(device)
    discovery_selection_model = None
    if args.discovery_selection_model == "ema":
        discovery_selection_model = copy.deepcopy(student).to(device)
        discovery_selection_model.eval()
        for parameter in discovery_selection_model.parameters():
            parameter.requires_grad_(False)
    teacher_ckpt = resolve_input_checkpoint(args.teacher_ckpt, args.work_dir, "teacher.pt")
    load_checkpoint(teacher, teacher_ckpt, device)
    vos_gaussian_stats = None

    def refresh_vos_gaussian_stats():
        raw_stats = collect_diagonal_gaussian_stats(
            student,
            build_loader(bundle.train, args.batch_size, False, args.num_workers),
            device,
            len(bundle.known_classes),
            include_shared_covariance=False,
        )
        return {
            "means": torch.as_tensor(raw_stats["means"], dtype=torch.float32, device=device),
            "variances": torch.as_tensor(raw_stats["variances"], dtype=torch.float32, device=device),
        }

    if args.alpha_vos > 0.0 and args.vos_mode == "gaussian":
        vos_gaussian_stats = refresh_vos_gaussian_stats()
        print("initialized VOS class-conditional Gaussian statistics")
    if args.encoder_lr_scale < 0.0:
        raise ValueError("--encoder-lr-scale must be non-negative")
    encoder_parameters = list(student.encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter
        for parameter in student.parameters()
        if id(parameter) not in encoder_parameter_ids
    ]
    reciprocal_points = None
    if args.reciprocal_points > 0:
        reciprocal_points = torch.nn.Parameter(
            torch.randn(args.reciprocal_points, student.encoder.out_dim, device=device) * 0.02
        )
    optimizer_groups = [
        {
            "params": encoder_parameters,
            "lr": args.lr * args.encoder_lr_scale,
        },
        {
            "params": head_parameters,
            "lr": args.lr,
        },
    ]
    if novel_head is not None:
        optimizer_groups.append({"params": list(novel_head.parameters()), "lr": args.lr})
    if reciprocal_points is not None:
        optimizer_groups.append({"params": [reciprocal_points], "lr": args.lr})
    optim = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    best_acc = -1.0
    best_state = None
    best_novel_head_state = None
    best_reciprocal_state = None
    best_epoch = 0
    history = []
    prototype_init_stats = None
    prototype_warmup = max(int(args.joint_prototype_warmup_epochs), 0)
    if (
        novel_head is not None
        and args.joint_prototype_init in {"kmeans", "kmeans_candidates"}
        and prototype_warmup == 0
    ):
        prototype_init_stats = initialize_novel_head_kmeans(
            novel_head,
            student,
            discovery_loader,
            device,
            args.joint_num_novel,
            args.seed,
            candidate_ratio=args.joint_prototype_candidate_ratio,
            candidates_only=args.joint_prototype_init == "kmeans_candidates",
        )
        print(f"novel prototype KMeans init: {prototype_init_stats}")
    for epoch in range(args.epochs):
        if (
            novel_head is not None
            and args.joint_prototype_init in {"kmeans", "kmeans_candidates"}
            and epoch == prototype_warmup
            and prototype_warmup > 0
        ):
            prototype_init_stats = initialize_novel_head_kmeans(
                novel_head,
                student,
                discovery_loader,
                device,
                args.joint_num_novel,
                args.seed,
                candidate_ratio=args.joint_prototype_candidate_ratio,
                candidates_only=args.joint_prototype_init == "kmeans_candidates",
            )
            print(f"novel prototype KMeans init: {prototype_init_stats}")
        discovery_knn_boundary_weight = scheduled_weight(
            epoch,
            warmup_epochs=args.discovery_knn_warmup_epochs,
            ramp_epochs=args.discovery_knn_ramp_epochs,
        )
        discovery_knn_support = None
        if (
            args.alpha_discovery_knn_boundary > 0.0
            and discovery_knn_boundary_weight > 0.0
        ):
            discovery_knn_support = collect_classwise_knn_support(
                student,
                knn_support_loader,
                device,
                len(bundle.known_classes),
                k=args.discovery_knn_k,
                quantile=args.discovery_knn_quantile,
            )
            observed_radii = discovery_knn_support["radii"][
                discovery_knn_support["counts"].to(device) > 0
            ]
            print(
                "known kNN support bank: "
                f"samples={int(discovery_knn_support['counts'].sum())}, "
                f"classes={int((discovery_knn_support['counts'] > 0).sum())}, "
                f"radius_median={observed_radii.median().item():.4f}"
            )
        selective_weight = scheduled_weight(
            epoch,
            warmup_epochs=args.discovery_selective_warmup_epochs,
            ramp_epochs=args.discovery_selective_ramp_epochs,
        )
        uncertainty_separation_weight = scheduled_weight(
            epoch,
            warmup_epochs=args.uncertainty_separation_warmup_epochs,
            ramp_epochs=args.uncertainty_separation_ramp_epochs,
        )
        discovery_uniform_weight = scheduled_weight(
            epoch,
            warmup_epochs=args.discovery_uniform_warmup_epochs,
            ramp_epochs=args.discovery_uniform_ramp_epochs,
        )
        stats = train_one_epoch_student(
            student,
            teacher,
            train_loader,
            optim,
            device,
            alpha_unc=args.alpha_unc,
            alpha_kd=args.alpha_kd,
            alpha_feat_kd=args.alpha_feat_kd,
            alpha_supcon=args.alpha_supcon,
            alpha_raw_supcon=args.alpha_raw_supcon,
            alpha_proto=args.alpha_proto,
            alpha_proto_repulsion=args.alpha_proto_repulsion,
            proto_repulsion_margin=args.proto_repulsion_margin,
            alpha_proxy=args.alpha_proxy,
            alpha_pseudo=args.alpha_pseudo,
            alpha_energy=args.alpha_energy,
            alpha_vos=args.alpha_vos,
            vos_mode=args.vos_mode,
            vos_gaussian_stats=vos_gaussian_stats,
            vos_tail_scale=args.vos_tail_scale,
            vos_noise_scale=args.vos_noise_scale,
            vos_uniform_weight=args.vos_uniform_weight,
            pseudo_mode=args.pseudo_mode,
            pseudo_feature_noise=args.pseudo_feature_noise,
            uncertainty_target_mode=args.uncertainty_target_mode,
            temperature=args.temperature,
            kd_mode=args.kd_mode,
            kd_uncertainty_source=args.kd_uncertainty_source,
            kd_mc_samples=args.kd_mc_samples,
            uncertainty_weight_mode=args.uncertainty_weight_mode,
            uncertainty_weight_min=args.uncertainty_weight_min,
            uncertainty_weight_max=args.uncertainty_weight_max,
            discovery_loader=discovery_loader,
            alpha_discovery=args.alpha_discovery,
            alpha_discovery_unknown=args.alpha_discovery_unknown,
            alpha_discovery_energy=args.alpha_discovery_energy,
            alpha_discovery_uniform=args.alpha_discovery_uniform * discovery_uniform_weight,
            alpha_discovery_feature_margin=args.alpha_discovery_feature_margin,
            discovery_feature_margin=args.discovery_feature_margin,
            alpha_discovery_feature_separation=args.alpha_discovery_feature_separation,
            discovery_feature_separation_margin=args.discovery_feature_separation_margin,
            discovery_feature_separation_temperature=args.discovery_feature_separation_temperature,
            discovery_feature_candidate_gating=args.discovery_feature_candidate_gating,
            discovery_pool_mode=args.discovery_pool_mode,
            alpha_discovery_boundary=args.alpha_discovery_boundary,
            discovery_boundary_margin=args.discovery_boundary_margin,
            discovery_boundary_prototype_weight=args.discovery_boundary_prototype_weight,
            discovery_boundary_temperature=args.discovery_boundary_temperature,
            alpha_discovery_knn_boundary=(
                args.alpha_discovery_knn_boundary * discovery_knn_boundary_weight
            ),
            discovery_knn_support=discovery_knn_support,
            discovery_knn_k=args.discovery_knn_k,
            discovery_knn_margin=args.discovery_knn_margin,
            alpha_discovery_uncertainty_separation=(
                args.alpha_discovery_uncertainty_separation
                * uncertainty_separation_weight
            ),
            discovery_uncertainty_loss=args.discovery_uncertainty_loss,
            discovery_uncertainty_margin=args.discovery_uncertainty_margin,
            alpha_discovery_uncertainty_pu=args.alpha_discovery_uncertainty_pu,
            discovery_uncertainty_known_prior=args.discovery_uncertainty_known_prior,
            alpha_discovery_selective_unknown=args.alpha_discovery_selective_unknown * selective_weight,
            alpha_discovery_selective_energy=args.alpha_discovery_selective_energy * selective_weight,
            discovery_select_ratio=args.discovery_select_ratio,
            discovery_select_mode=args.discovery_select_mode,
            discovery_loss_mode=args.discovery_loss,
            discovery_temperature=args.discovery_temperature,
            energy_margin=args.energy_margin,
            energy_temperature=args.energy_temperature,
            proxy_temperature=args.proxy_temperature,
            alpha_angular=args.alpha_angular,
            angular_margin=args.angular_margin,
            angular_scale=args.angular_scale,
            alpha_proxy_anchor=args.alpha_proxy_anchor,
            proxy_anchor_alpha=args.proxy_anchor_alpha,
            proxy_anchor_margin=args.proxy_anchor_margin,
            reciprocal_points=reciprocal_points,
            alpha_reciprocal=args.alpha_reciprocal,
            reciprocal_margin=args.reciprocal_margin,
            discovery_selection_model=discovery_selection_model,
            discovery_ema_decay=args.discovery_ema_decay,
            discovery_neighbor_filter=args.discovery_neighbor_filter,
            discovery_neighbor_k=args.discovery_neighbor_k,
            discovery_neighbor_min_votes=args.discovery_neighbor_min_votes,
            discovery_soft_weighting=args.discovery_soft_weighting,
            discovery_neighbor_temperature=args.discovery_neighbor_temperature,
            novel_head=novel_head,
            alpha_joint_discovery=(
                args.alpha_joint_discovery
                if args.joint_discovery and epoch >= prototype_warmup
                else 0.0
            ),
            joint_confidence_threshold=args.joint_confidence_threshold,
            joint_assignment_temperature=args.joint_assignment_temperature,
            alpha_joint_consistency=args.alpha_joint_consistency,
            alpha_joint_balance=args.alpha_joint_balance,
            alpha_joint_information=args.alpha_joint_information,
            alpha_joint_neighbor=args.alpha_joint_neighbor,
            alpha_joint_pseudo=args.alpha_joint_pseudo,
            alpha_joint_gate=args.alpha_joint_gate,
            joint_gate_margin=args.joint_gate_margin,
            joint_neighbor_k=args.joint_neighbor_k,
            alpha_joint_proto_repulsion=args.alpha_joint_proto_repulsion,
            joint_proto_repulsion_margin=args.joint_proto_repulsion_margin,
            joint_space=args.joint_space,
            alpha_joint_known_ce=args.alpha_joint_known_ce,
            joint_known_temperature=args.joint_known_temperature,
            joint_mixed_residual=args.joint_mixed_residual,
            joint_residual_temperature=args.joint_residual_temperature,
            joint_residual_floor=args.joint_residual_floor,
            joint_novel_mass=args.joint_novel_mass,
            joint_novel_neighbor_support=args.joint_novel_neighbor_support,
            joint_novel_neighbor_k=args.joint_novel_neighbor_k,
            joint_novel_ema_weights=args.joint_novel_ema_weights,
            joint_weighted_sinkhorn=args.joint_weighted_sinkhorn,
            alpha_joint_novel_margin=args.alpha_joint_novel_margin,
            joint_novel_margin=args.joint_novel_margin,
            joint_candidate_gating=args.joint_candidate_gating,
            joint_candidate_ratio=args.joint_candidate_ratio,
            joint_candidate_mode=args.joint_candidate_mode,
            joint_candidate_neighbor_filter=args.joint_candidate_neighbor_filter,
            joint_candidate_neighbor_k=args.joint_candidate_neighbor_k,
            joint_candidate_min_votes=args.joint_candidate_min_votes,
            joint_candidate_soft_weighting=args.joint_candidate_soft_weighting,
            joint_candidate_weight_floor=args.joint_candidate_weight_floor,
            joint_mixed_known_consistency=args.joint_mixed_known_consistency,
            alpha_joint_known_consistency=args.alpha_joint_known_consistency,
            joint_known_confidence_threshold=args.joint_known_confidence_threshold,
            joint_known_target=args.joint_known_target,
            outlier_loader=outlier_loader,
            alpha_outlier_uniform=args.alpha_outlier_uniform,
            alpha_outlier_energy=args.alpha_outlier_energy,
            alpha_outlier_uncertainty=args.alpha_outlier_uncertainty,
            alpha_outlier_feature_margin=args.alpha_outlier_feature_margin,
            outlier_feature_margin=args.outlier_feature_margin,
            freeze_bn_stats=args.freeze_bn_stats,
        )
        stats["discovery_selective_weight"] = selective_weight
        stats["uncertainty_separation_weight"] = uncertainty_separation_weight
        stats["discovery_uniform_weight"] = discovery_uniform_weight
        stats["discovery_knn_boundary_weight"] = discovery_knn_boundary_weight
        if (
            args.alpha_vos > 0.0
            and args.vos_mode == "gaussian"
            and epoch + 1 < args.epochs
        ):
            vos_gaussian_stats = refresh_vos_gaussian_stats()
        val_stats = evaluate_classification(student, val_loader, device)
        history.append({"epoch": epoch + 1, "train": stats, "validation": val_stats})
        print(f"[student][{epoch+1}/{args.epochs}] {stats} {val_stats}")
        if val_stats["known_acc"] > best_acc:
            best_acc = val_stats["known_acc"]
            best_state = copy.deepcopy(student.state_dict())
            best_novel_head_state = (
                copy.deepcopy(novel_head.state_dict()) if novel_head is not None else None
            )
            best_reciprocal_state = (
                reciprocal_points.detach().clone() if reciprocal_points is not None else None
            )
            best_epoch = epoch + 1
    if best_state is not None:
        student.load_state_dict(best_state)
    if novel_head is not None and best_novel_head_state is not None:
        novel_head.load_state_dict(best_novel_head_state)
    if reciprocal_points is not None and best_reciprocal_state is not None:
        reciprocal_points.data.copy_(best_reciprocal_state)
    run_dir = ensure_dir(args.work_dir)
    save_command_config(
        run_dir,
        args.command,
        {
            **vars(args),
            "best_epoch": best_epoch,
            "best_val_known_acc": best_acc,
            "prototype_init_stats": prototype_init_stats,
            "discovery_pool_semantics": (
                "oracle_filtered_novel_only (training class labels select the pool)"
                if args.discovery_pool_mode == "unknown"
                else "unlabeled_mixed_known_and_novel"
            ),
        },
    )
    save_json(run_dir / "train_history.json", history)
    proto_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
    prototypes = collect_prototypes(student, proto_loader, device, len(bundle.known_classes)).cpu()
    gaussian_stats = collect_diagonal_gaussian_stats(
        student, proto_loader, device, len(bundle.known_classes)
    )
    gaussian_stats = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in gaussian_stats.items()}
    student_ckpt = resolve_output_checkpoint(args.student_ckpt, args.work_dir, "student.pt")
    save_checkpoint(
        student,
        student_ckpt,
        extra={
            "known_classes": list(bundle.known_classes),
            "prototypes": prototypes,
            "gaussian_stats": gaussian_stats,
            "joint_novel_head": novel_head.state_dict() if novel_head is not None else None,
            "reciprocal_points": (
                reciprocal_points.detach().cpu() if reciprocal_points is not None else None
            ),
        },
    )
    if novel_head is not None:
        torch.save(
            {
                "num_novel": args.joint_num_novel,
                "temperature": args.joint_head_temperature,
                "known_temperature": args.joint_known_temperature,
                "state_dict": novel_head.state_dict(),
            },
            run_dir / "novel_head.pt",
        )
    print(f"saved to {student_ckpt} (best_epoch={best_epoch}, best_val_known_acc={best_acc:.4f})")


def discover(args):
    set_seed(args.seed)
    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=args.download,
        split_path=args.split_path,
        limit_train=args.limit_train or None,
        limit_val=args.limit_val or None,
        limit_test=args.limit_test or None,
        limit_discovery=args.limit_discovery or None,
        discovery_pool_mode=args.discovery_pool_mode,
        calibration_ratio=args.calibration_ratio,
        calibration_overlap_control=args.calibration_overlap_control,
        known_split_mode=args.known_split_mode,
        mixed_known_pool_ratio=args.mixed_known_pool_ratio,
        open_val_ratio=args.open_val_ratio,
    )
    device = resolve_device(args.device)
    print(f"device: {device}")
    test_loader = build_loader(bundle.test, args.batch_size, False, args.num_workers)
    calibration_dataset = bundle.calibration if bundle.calibration is not None else bundle.val
    val_loader = build_loader(calibration_dataset, args.batch_size, False, args.num_workers)
    model = build_model(
        len(bundle.known_classes),
        backbone=args.student_backbone or args.backbone,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
        cifar_stem=args.cifar_stem,
    ).to(device)
    ckpt_path = resolve_input_checkpoint(args.student_ckpt, args.work_dir, "student.pt")
    ckpt = load_checkpoint(model, ckpt_path, device)
    react_clip_value = None
    if args.react_percentile > 0.0:
        train_stats_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
        react_clip_value = collect_activation_clip_value(
            model, train_stats_loader, device, args.react_percentile
        )
        print(f"react activation cap: percentile={args.react_percentile:.2f}, value={react_clip_value:.6f}")
    knn_bank = None
    if args.knn_ood:
        if not 0.5 <= args.knn_support_quantile < 1.0:
            raise ValueError("--knn-support-quantile must be in [0.5, 1.0)")
        deterministic_train = clone_dataset_with_eval_transform(
            bundle.train, args.image_size
        )
        train_feature_loader = build_loader(
            deterministic_train, args.batch_size, False, args.num_workers
        )
        knn_bank = collect_knn_feature_bank(
            model,
            train_feature_loader,
            device,
            max_samples=args.knn_bank_size,
            seed=args.seed,
            feature_key=args.knn_feature,
            support_k=args.knn_k,
            support_quantile=args.knn_support_quantile,
        )
        print(f"knn feature bank: {len(knn_bank['features'])} vectors, k={args.knn_k}")
    vim_stats = None
    if args.vim_ood:
        train_feature_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
        vim_stats = collect_vim_stats(model, train_feature_loader, device, rank=args.vim_rank)
        print(f"vim principal subspace: rank={len(vim_stats['components'])}")
    novel_head = None
    novel_head_state = None
    novel_head_temperature = 1.0
    novel_known_temperature = (
        1.0 if args.novel_known_temperature is None else float(args.novel_known_temperature)
    )
    if args.novel_head_ckpt:
        novel_head_payload = torch.load(args.novel_head_ckpt, map_location=device)
        novel_head_state = novel_head_payload.get("state_dict", novel_head_payload)
        novel_head_temperature = float(novel_head_payload.get("temperature", 1.0))
        if args.novel_known_temperature is None:
            novel_known_temperature = float(novel_head_payload.get("known_temperature", 1.0))
    elif ckpt.get("joint_novel_head") is not None:
        novel_head_state = ckpt["joint_novel_head"]
    if novel_head_state is not None:
        novel_head = NovelPrototypeHead(
            model.encoder.out_dim,
            args.num_novel,
            temperature=novel_head_temperature,
        ).to(device)
        novel_head.load_state_dict(novel_head_state)
        novel_head.eval()
    outputs_test = extract_outputs(
        model,
        test_loader,
        device,
        mc_samples=args.mc_samples,
        odin_epsilon=args.odin_epsilon,
        odin_temperature=args.odin_temperature,
        react_clip_value=react_clip_value,
    )
    outputs_val = extract_outputs(
        model,
        val_loader,
        device,
        mc_samples=args.mc_samples,
        odin_epsilon=args.odin_epsilon,
        odin_temperature=args.odin_temperature,
        react_clip_value=react_clip_value,
    )
    if novel_head is not None:
        outputs_test = attach_novel_outputs(
            outputs_test, novel_head, device, known_temperature=novel_known_temperature
        )
        outputs_val = attach_novel_outputs(
            outputs_val, novel_head, device, known_temperature=novel_known_temperature
        )
    open_val_loader = build_loader(bundle.open_val, args.batch_size, False, args.num_workers) if bundle.open_val is not None else None
    outputs_open_val = (
        extract_outputs(
            model,
            open_val_loader,
            device,
            mc_samples=args.mc_samples,
            odin_epsilon=args.odin_epsilon,
            odin_temperature=args.odin_temperature,
            react_clip_value=react_clip_value,
        )
        if open_val_loader is not None
        else None
    )
    if novel_head is not None and outputs_open_val is not None:
        outputs_open_val = attach_novel_outputs(
            outputs_open_val, novel_head, device, known_temperature=novel_known_temperature
        )
    if args.score_mode in {"feature_rejector", "virtual_rejector"}:
        if args.rejector_training == "nnpu":
            if args.score_mode != "feature_rejector":
                raise ValueError("nnpu training is only available for feature_rejector")
            if args.discovery_pool_mode != "mixed":
                raise ValueError("nnpu rejector training requires --discovery-pool-mode mixed")
            if not 0.0 < args.rejector_known_prior < 1.0:
                raise ValueError("--rejector-known-prior must be in (0, 1) for nnpu")
        if (
            args.score_mode == "feature_rejector"
            and args.rejector_strict_mixed
            and args.discovery_pool_mode == "mixed"
            and args.open_val_ratio <= 0.0
        ):
            raise ValueError(
                "--rejector-strict-mixed with a mixed pool requires --open-val-ratio > 0"
            )
        if args.score_mode == "feature_rejector" and (
            bundle.discovery_pool is None or len(bundle.discovery_pool) == 0
        ):
            raise ValueError(
                "feature_rejector requires a non-empty discovery pool; "
                "use the unknown discovery pool or provide --limit-discovery"
            )
        if (
            args.score_mode == "feature_rejector"
            and args.rejector_strict_mixed
            and args.discovery_pool_mode == "mixed"
            and outputs_open_val is not None
        ):
            known_open_mask = np.asarray(outputs_open_val["is_known"], dtype=bool)
            rejector_known_outputs = {
                key: value[known_open_mask]
                if isinstance(value, np.ndarray) and value.shape[:1] == known_open_mask.shape
                else value
                for key, value in outputs_open_val.items()
            }
            if len(rejector_known_outputs["features"]) == 0:
                raise ValueError("strict mixed rejector found no known open-validation samples")
            print(
                "feature rejector strict mixed protocol: using "
                f"{len(rejector_known_outputs['features'])} held-out known samples"
            )
        else:
            rejector_known_loader = build_loader(
                bundle.train, args.batch_size, False, args.num_workers
            )
            rejector_known_outputs = extract_outputs(
                model,
                rejector_known_loader,
                device,
                mc_samples=(args.rejector_mc_samples if args.rejector_feature_mode in {
                    "uncertainty_augmented", "support_uncertainty_augmented"
                } else 1),
            )
        support_model_for_rejector = None
        if args.score_mode == "virtual_rejector":
            feature_rejector = fit_virtual_outlier_rejector(
                rejector_known_outputs, seed=args.seed
            )
            print(
                "virtual feature rejector: trained on "
                f"{len(rejector_known_outputs['features'])} known features and "
                "class-boundary virtual outliers"
            )
        else:
            rejector_unknown_loader = build_loader(
                bundle.discovery_pool, args.batch_size, False, args.num_workers
            )
            rejector_unknown_outputs = extract_outputs(
                model,
                rejector_unknown_loader,
                device,
                mc_samples=(args.rejector_mc_samples if args.rejector_feature_mode in {
                    "uncertainty_augmented", "support_uncertainty_augmented"
                } else 1),
            )
            if args.rejector_feature_mode in {"support_augmented", "support_uncertainty_augmented"}:
                support_model_for_rejector = fit_known_support_rejector(
                    rejector_known_outputs, quantile=args.support_quantile
                )
                attach_known_support_score(rejector_known_outputs, support_model_for_rejector)
                attach_known_support_score(rejector_unknown_outputs, support_model_for_rejector)
            if args.rejector_pool_selection == "high_risk":
                pool_logits = torch.from_numpy(rejector_unknown_outputs["logits"])
                pool_uncertainty = torch.from_numpy(rejector_unknown_outputs["head_uncertainty"])
                candidate_mask = select_discovery_candidates(
                    pool_logits,
                    pool_uncertainty,
                    ratio=args.rejector_candidate_ratio,
                    mode=args.rejector_candidate_mode,
                ).numpy()
                if int(candidate_mask.sum()) < 2:
                    raise ValueError("rejector high-risk selection produced fewer than two samples")
                rejector_unknown_outputs = {
                    key: value[candidate_mask]
                    if isinstance(value, np.ndarray) and value.shape[:1] == candidate_mask.shape
                    else value
                    for key, value in rejector_unknown_outputs.items()
                }
                print(
                    "feature rejector pseudo-unknown selection: "
                    f"{int(candidate_mask.sum())}/{len(candidate_mask)} samples"
                )
            if args.rejector_training == "soft_pu":
                feature_rejector = fit_pu_feature_rejector(
                    rejector_known_outputs,
                    rejector_unknown_outputs,
                    iterations=args.rejector_pu_iterations,
                    feature_mode=args.rejector_feature_mode,
                    seed=args.seed,
                )
            elif args.rejector_training == "nnpu":
                feature_rejector = fit_nnpu_feature_rejector(
                    rejector_known_outputs,
                    rejector_unknown_outputs,
                    known_prior=args.rejector_known_prior,
                    iterations=max(50, args.rejector_pu_iterations * 75),
                    feature_mode=args.rejector_feature_mode,
                    seed=args.seed,
                )
            else:
                feature_rejector = fit_feature_rejector(
                    rejector_known_outputs,
                    rejector_unknown_outputs,
                    feature_mode=args.rejector_feature_mode,
                    model_type=args.rejector_model,
                    seed=args.seed,
                )
        for extracted in (outputs_test, outputs_val, outputs_open_val):
            if extracted is not None:
                if support_model_for_rejector is not None:
                    attach_known_support_score(extracted, support_model_for_rejector)
                attach_feature_rejector_score(
                    extracted, feature_rejector, feature_mode=args.rejector_feature_mode
                )
        print(
            "feature rejector scores attached"
        )
    if args.score_mode == "known_support":
        support_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
        support_outputs = extract_outputs(model, support_loader, device, mc_samples=1)
        support_model = fit_known_support_rejector(
            support_outputs, quantile=args.support_quantile
        )
        for extracted in (outputs_test, outputs_val, outputs_open_val):
            if extracted is not None:
                attach_known_support_score(extracted, support_model)
        print(
            "known support detector: fitted on "
            f"{len(support_outputs['features'])} known features, "
            f"{len(support_model['classes'])} classes, q={args.support_quantile:.3f}"
        )
    if knn_bank is not None:
        attach_knn_distances(outputs_test, knn_bank, k=args.knn_k, feature_key=args.knn_feature)
        attach_knn_distances(outputs_val, knn_bank, k=args.knn_k, feature_key=args.knn_feature)
        if outputs_open_val is not None:
            attach_knn_distances(outputs_open_val, knn_bank, k=args.knn_k, feature_key=args.knn_feature)
    if vim_stats is not None:
        attach_vim_residual(outputs_test, vim_stats)
        attach_vim_residual(outputs_val, vim_stats)
        if outputs_open_val is not None:
            attach_vim_residual(outputs_open_val, vim_stats)
    temperature = fit_temperature(outputs_val) if args.temperature_calibration else 1.0
    if args.temperature_calibration:
        outputs_test = apply_temperature(outputs_test, temperature)
        outputs_val = apply_temperature(outputs_val, temperature)
        if outputs_open_val is not None:
            outputs_open_val = apply_temperature(outputs_open_val, temperature)
    proto = ckpt.get("prototypes")
    if proto is not None and torch.is_tensor(proto):
        proto = proto.detach().cpu().numpy()
    gaussian_stats = ckpt.get("gaussian_stats")
    if gaussian_stats is not None:
        gaussian_stats = {
            key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            for key, value in gaussian_stats.items()
        }
    reciprocal_points = ckpt.get("reciprocal_points")
    if reciprocal_points is not None:
        gaussian_stats = {} if gaussian_stats is None else gaussian_stats
        gaussian_stats["reciprocal_points"] = (
            reciprocal_points.detach().cpu().numpy()
            if torch.is_tensor(reciprocal_points)
            else np.asarray(reciprocal_points)
        )
    if args.auto_score_fast and (args.score_mode == "auto" or args.auto_calibrate_score):
        # Fast calibration does not compare Mahalanobis scores, so avoid
        # loading or fitting their high-dimensional statistics.
        gaussian_stats = None
    if (
        not args.auto_score_fast
        and (
            gaussian_stats is None
            or "precision" not in gaussian_stats
            or (
                args.score_mode in {"relative_mahalanobis", "normalized_entropy_relative_mahalanobis"}
                and "global_variances" not in gaussian_stats
            )
        )
        and _score_needs_shared_mahalanobis(args.score_mode, args.auto_calibrate_score)
    ):
        stats_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
        gaussian_stats = collect_diagonal_gaussian_stats(
            model,
            stats_loader,
            device,
            len(bundle.known_classes),
            include_shared_covariance=args.score_mode not in {
                "relative_mahalanobis",
                "normalized_entropy_relative_mahalanobis",
            },
        )
    if args.score_mode == "reciprocal" and (
        gaussian_stats is None or "reciprocal_points" not in gaussian_stats
    ):
        raise ValueError("--score-mode reciprocal requires a checkpoint trained with --reciprocal-points")
    if args.score_mode == "openmax" and (
        gaussian_stats is None or "weibull_shapes" not in gaussian_stats
    ):
        stats_loader = build_loader(bundle.train, args.batch_size, False, args.num_workers)
        gaussian_stats = collect_diagonal_gaussian_stats(
            model, stats_loader, device, len(bundle.known_classes)
        )
    selected_score_mode = args.score_mode
    needs_normalization = (
        selected_score_mode.startswith("normalized_")
        or selected_score_mode == "classwise_unified_novel_mass"
        or selected_score_mode == "classwise_mahalanobis"
    )
    score_normalization = (
        fit_score_normalization(
            outputs_val,
            proto,
            gaussian_stats if selected_score_mode != "classwise_unified_novel_mass" else None,
            include_mahalanobis=selected_score_mode != "classwise_unified_novel_mass",
        )
        if needs_normalization or args.score_mode == "auto" or args.auto_calibrate_score
        else {}
    )
    calibration_report = {
        "validation_protocol": {
            "checkpoint_selection_samples": int(len(bundle.val)),
            "threshold_calibration_samples": int(len(calibration_dataset)),
            "requested_calibration_ratio": float(args.calibration_ratio),
            "uses_disjoint_calibration_subset": bool(
                bundle.calibration is not None and not args.calibration_overlap_control
            ),
            "calibration_overlap_control": bool(args.calibration_overlap_control),
        },
        "temperature": float(temperature),
        "temperature_enabled": bool(args.temperature_calibration),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "odin": {
            "epsilon": float(args.odin_epsilon),
            "temperature": float(args.odin_temperature),
            "enabled": bool(args.odin_epsilon > 0.0),
        },
        "react": {
            "percentile": float(args.react_percentile),
            "enabled": bool(react_clip_value is not None),
            "clip_value": None if react_clip_value is None else float(react_clip_value),
        },
        "reliability": calibration_diagnostics(
            outputs_val["probs"], outputs_val["labels"], args.calibration_bins
        ),
        "uncertainty_error": uncertainty_error_diagnostics(outputs_val),
    }
    threshold = None
    if (args.score_mode == "auto" or args.auto_calibrate_score) and outputs_open_val is not None:
        candidate_modes = [
            "full",
            "entropy_proto",
            "entropy_only",
            "proto_only",
            "max_softmax",
            "max_logit",
            "logit_margin",
            "odin_msp",
            "energy",
            "normalized_full",
            "normalized_entropy_proto",
            "entropy_epistemic",
            "expected_entropy",
            "mahalanobis",
            "mahalanobis_diag",
            "mahalanobis_shared",
            "entropy_mahalanobis",
            "entropy_mahalanobis_shared",
            "normalized_entropy_mahalanobis",
            "normalized_entropy_mahalanobis_shared",
            "gaussian_nll",
            "normalized_entropy_gaussian_nll",
        ]
        if args.auto_score_fast:
            candidate_modes = [
                "full",
                "entropy_proto",
                "entropy_only",
                "proto_only",
                "max_softmax",
                "max_logit",
                "logit_margin",
                "energy",
                "normalized_full",
                "normalized_entropy_proto",
                "entropy_epistemic",
                "expected_entropy",
                "gaussian_nll",
                "normalized_entropy_gaussian_nll",
            ]
        if args.react_percentile > 0.0:
            candidate_modes.append("react_energy")
        if args.knn_ood:
            candidate_modes.extend(["knn_distance", "normalized_entropy_knn"])
        if args.vim_ood:
            candidate_modes.append("vim_residual")
        if novel_head is not None:
            # Let unified joint checkpoints compete with legacy OOD scores
            # during open-validation calibration.
            candidate_modes.extend(
                [
                    "novel_msp",
                    "novel_entropy",
                    "unified_novel_mass",
                    "normalized_unified_novel_mass_entropy",
                ]
            )
        selected, all_candidates = _select_score_mode(
            outputs_val,
            outputs_open_val,
            proto,
            candidate_modes,
            normalization=score_normalization,
            gaussian_stats=gaussian_stats,
        )
        selected_score_mode = selected["score_mode"]
        scores_val, _ = compute_open_score(
            outputs_val,
            prototypes=proto,
            score_mode=selected_score_mode,
            normalization=score_normalization,
            gaussian_stats=gaussian_stats,
        )
        scores_open_val = None
        if outputs_open_val is not None:
            scores_open_val, _ = compute_open_score(
                outputs_open_val,
                prototypes=proto,
                score_mode=selected_score_mode,
                normalization=score_normalization,
                gaussian_stats=gaussian_stats,
            )
        threshold, threshold_policy_report = _calibrate_discovery_threshold(
            args,
            scores_val,
            outputs_val,
            scores_open_val,
            outputs_open_val,
            len(bundle.known_classes),
        )
        calibration_report.update(
            {
                "selected": selected,
                "candidates": all_candidates,
                "open_val_size": int(len(outputs_open_val["labels"])),
                "open_val_unknown_size": int(np.sum(np.asarray(outputs_open_val["is_known"], dtype=bool) == 0)),
                "threshold_policy": threshold_policy_report,
                "calibrated_threshold": (
                    threshold.tolist() if isinstance(threshold, np.ndarray) else float(threshold)
                ),
            }
        )
    else:
        if selected_score_mode == "auto":
            selected_score_mode = "full"
        scores_val, _ = compute_open_score(
            outputs_val,
            prototypes=proto,
            score_mode=selected_score_mode,
            normalization=score_normalization,
            gaussian_stats=gaussian_stats,
        )
        scores_open_val = None
        if outputs_open_val is not None:
            scores_open_val, _ = compute_open_score(
                outputs_open_val,
                prototypes=proto,
                score_mode=selected_score_mode,
                normalization=score_normalization,
                gaussian_stats=gaussian_stats,
            )
        threshold, threshold_policy_report = _calibrate_discovery_threshold(
            args,
            scores_val,
            outputs_val,
            scores_open_val,
            outputs_open_val,
            len(bundle.known_classes),
        )
        calibration_report.update(
            {
                "threshold_policy": threshold_policy_report,
                "calibrated_threshold": (
                    threshold.tolist() if isinstance(threshold, np.ndarray) else float(threshold)
                ),
            }
        )

    # Assess candidate-pool purification on the reserved open-validation split.
    # This uses its known/unknown identity labels for diagnostics only; it does
    # not change the purifier mode/ratio or use final test labels for selection.
    if outputs_open_val is not None:
        open_val_score, _ = compute_open_score(
            outputs_open_val,
            prototypes=proto,
            score_mode=selected_score_mode,
            normalization=score_normalization,
            gaussian_stats=gaussian_stats,
        )
        open_val_pred_class = outputs_open_val["logits"].argmax(axis=1)
        threshold_array = np.asarray(threshold)
        if threshold_array.ndim == 0:
            open_val_pred_known = open_val_score <= float(threshold_array)
        else:
            open_val_pred_known = open_val_score <= threshold_array[
                np.clip(open_val_pred_class, 0, len(threshold_array) - 1)
            ]
        open_val_candidates = ~open_val_pred_known
        open_val_purified, _ = purify_candidate_mask(
            outputs_open_val,
            open_val_candidates,
            mode=args.candidate_purify,
            keep_ratio=args.candidate_keep_ratio,
            open_score=open_val_score,
        )
        calibration_report["candidate_purification_open_val"] = evaluate_candidate_purification(
            outputs_open_val, open_val_candidates, open_val_purified
        )

    result, scores_test, pred_known, detail = run_discovery(
        outputs_test,
        threshold,
        args.num_novel,
        prototypes=proto,
        score_mode=selected_score_mode,
        normalization=score_normalization,
        gaussian_stats=gaussian_stats,
        cluster_k=args.cluster_k,
        cluster_method=args.cluster_method,
        cluster_feature=args.cluster_feature,
        cluster_selection=args.cluster_selection,
        cluster_pca_dim=args.cluster_pca_dim,
        cluster_normalize=args.cluster_normalize,
        cluster_whiten=not args.cluster_no_whiten,
        cluster_n_init=args.cluster_n_init,
        cluster_stability_repeats=args.cluster_stability_repeats,
        max_auto_clusters=args.max_auto_clusters,
        candidate_purify=args.candidate_purify,
        candidate_keep_ratio=args.candidate_keep_ratio,
        enable_clustering=not args.skip_clustering,
    )
    # ``threshold_type`` describes the shape of the threshold (global versus
    # class-conditional).  Keep the calibration policy separately so a scalar
    # threshold calibrated for a fixed known coverage is not mistaken for the
    # historical global-percentile policy.
    policy_report = calibration_report.get("threshold_policy", {})
    result["threshold_policy"] = policy_report.get("type", args.threshold_policy)
    if "target_known_coverage" in policy_report:
        result["target_known_coverage"] = float(policy_report["target_known_coverage"])
        result["calibration_known_accept_rate"] = float(
            policy_report.get("known_accept_rate", float("nan"))
        )
    run_dir = ensure_dir(args.work_dir)
    save_command_config(
        run_dir,
        args.command,
        {
            **vars(args),
            "discovery_pool_semantics": (
                "oracle_filtered_novel_only (training class labels select the pool)"
                if args.discovery_pool_mode == "unknown"
                else "unlabeled_mixed_known_and_novel"
            ),
        },
    )
    torch.save(
        {
            "known_classes": list(bundle.known_classes),
            "novel_classes": list(bundle.novel_classes),
        },
        run_dir / "split.pt",
    )
    save_json(run_dir / "discovery_report.json", result)
    save_json(run_dir / "calibration_report.json", calibration_report)
    save_json(run_dir / "discovery_detail.json", {
        "score_mode": detail["score_mode"],
        "score": detail["score"].tolist(),
        "entropy": detail["entropy"].tolist(),
        "epistemic": detail["epistemic"].tolist(),
        "aleatoric": detail["aleatoric"].tolist(),
        "expected_entropy": detail["expected_entropy"].tolist(),
        "head_uncertainty": detail["head_uncertainty"].tolist(),
        "proto_dist": detail["proto_dist"].tolist(),
        "mahalanobis": detail["mahalanobis"].tolist(),
        "odin_msp": detail["odin_msp"].tolist(),
        "react_energy": detail["react_energy"].tolist(),
        "knn_distance": detail["knn_distance"].tolist(),
        "knn_predicted_class_distance": detail.get(
            "knn_predicted_class_distance", np.zeros_like(detail["knn_distance"])
        ).tolist(),
        "knn_predicted_class_relative_distance": detail.get(
            "knn_predicted_class_relative_distance",
            np.zeros_like(detail["knn_distance"]),
        ).tolist(),
        "knn_predicted_class_support": detail.get(
            "knn_predicted_class_support", np.zeros_like(detail["knn_distance"])
        ).tolist(),
        "vim_residual": detail["vim_residual"].tolist(),
        "pred_known": detail["pred_known"].astype(int).tolist(),
        "true_known": detail["true_known"].astype(int).tolist(),
        "pred_class": detail["pred_class"].tolist(),
        "true_label": detail["true_label"].tolist(),
        "raw_labels": detail["raw_labels"].tolist(),
        "pred_cluster": None if detail["pred_cluster"] is None else detail["pred_cluster"].tolist(),
        "cluster_k": detail.get("cluster_k"),
        "cluster_method": detail.get("cluster_method"),
        "cluster_feature": detail.get("cluster_feature"),
        "cluster_selection": detail.get("cluster_selection"),
        "cluster_diagnostics": detail.get("cluster_diagnostics", []),
        "threshold_type": result.get("threshold_type"),
        "candidate_purify": detail.get("candidate_purify"),
        "candidate_keep_ratio": detail.get("candidate_keep_ratio"),
        "candidate_purification": detail.get("candidate_purification"),
    })
    np.save(run_dir / "open_scores.npy", scores_test)
    print(result)
    if "selected" in calibration_report:
        selected = calibration_report["selected"]
        print(
            "calibration:",
            {
                "selected_score_mode": selected["score_mode"],
                "auroc": selected["auroc"],
                "calibrated_threshold": calibration_report["calibrated_threshold"],
                "threshold_policy": calibration_report["threshold_policy"]["type"],
            },
        )
    save_json(run_dir / "score_normalization.json", score_normalization)
    threshold_display = (
        f"classwise[{len(threshold)}]"
        if isinstance(threshold, np.ndarray)
        else f"{threshold:.6f}"
    )
    print(f"threshold={threshold_display}, temperature={temperature:.4f}")


def inspect_data(args):
    set_seed(args.seed)
    bundle = build_data_bundle(
        args.dataset,
        args.data_root,
        args.num_known,
        args.seed,
        args.image_size,
        download=args.download,
        split_path=args.split_path,
        limit_train=args.limit_train or None,
        limit_val=args.limit_val or None,
        limit_test=args.limit_test or None,
        limit_discovery=args.limit_discovery or None,
        discovery_pool_mode=args.discovery_pool_mode,
        calibration_ratio=args.calibration_ratio,
        known_split_mode=args.known_split_mode,
        mixed_known_pool_ratio=args.mixed_known_pool_ratio,
    )
    print("dataset:", args.dataset)
    print("known classes:", len(bundle.known_classes))
    print("novel classes:", len(bundle.novel_classes))
    print("train size:", len(bundle.train))
    print("val size:", len(bundle.val))
    print("calibration size:", len(bundle.calibration) if bundle.calibration is not None else 0)
    print("test size:", len(bundle.test))
    print("discovery pool size:", len(bundle.discovery_pool) if bundle.discovery_pool is not None else 0)
    print("first known classes:", list(bundle.known_classes)[: min(args.sample_count, len(bundle.known_classes))])
    print("first novel classes:", list(bundle.novel_classes)[: min(args.sample_count, len(bundle.novel_classes))])


def main():
    args = parse_args()
    if args.command == "train_teacher":
        fit_teacher(args)
    elif args.command == "train_student":
        fit_student(args)
    elif args.command == "discover":
        discover(args)
    elif args.command == "inspect_data":
        inspect_data(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
