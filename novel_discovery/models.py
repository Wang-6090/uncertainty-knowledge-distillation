from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def freeze_batchnorm_stats(module: nn.Module) -> None:
    """Keep BatchNorm running statistics fixed while leaving affine weights trainable."""
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = False,
        cifar_stem: bool = False,
    ) -> None:
        super().__init__()
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            out_dim = base.fc.in_features
        elif name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=weights)
            out_dim = base.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        if cifar_stem:
            # CIFAR images are much smaller than ImageNet images.  The
            # ImageNet 7x7/stride-2 stem followed by max-pooling can discard
            # useful local structure before the residual blocks see it.
            # Keep the option explicit so historical ImageNet-stem runs stay
            # exactly reproducible.
            old_conv = base.conv1
            new_conv = nn.Conv2d(
                3, old_conv.out_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    resized = F.interpolate(
                        old_conv.weight,
                        size=(3, 3),
                        mode="bilinear",
                        align_corners=False,
                    )
                    new_conv.weight.copy_(resized)
            base.conv1 = new_conv
            base.maxpool = nn.Identity()

        self.out_dim = out_dim
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class MobileNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = base.classifier[0].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class IndependentRejectionEncoder(nn.Module):
    """Small image encoder dedicated to rejection, not shared classification."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, out_dim)
        self.out_dim = int(out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.flatten(self.features(images), 1))


class UKDNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        proj_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = False,
        cifar_stem: bool = False,
        rejection_feature_dim: int = 0,
        independent_rejection_encoder: bool = False,
        independent_rejection_encoder_init: str = "shared_backbone",
        independent_rejection_input_norm: str = "none",
    ) -> None:
        super().__init__()
        if backbone == "mobilenet_v3_small":
            self.encoder = MobileNetBackbone(pretrained=pretrained)
        else:
            self.encoder = ResNetBackbone(
                backbone, pretrained=pretrained, cifar_stem=cifar_stem
            )
        self.dropout_p = dropout
        feat_dim = self.encoder.out_dim
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.rejection_feature_dim = int(rejection_feature_dim)
        self.independent_rejection_encoder_enabled = bool(
            independent_rejection_encoder
        )
        if independent_rejection_encoder_init not in {
            "scratch_cnn",
            "shared_backbone",
            "random_backbone",
        }:
            raise ValueError(
                "independent_rejection_encoder_init must be 'scratch_cnn', "
                "'shared_backbone', or 'random_backbone'"
            )
        self.independent_rejection_encoder_init = (
            independent_rejection_encoder_init
        )
        if independent_rejection_input_norm not in {"none", "layernorm"}:
            raise ValueError(
                "independent_rejection_input_norm must be 'none' or 'layernorm'"
            )
        if independent_rejection_input_norm != "none" and not independent_rejection_encoder:
            raise ValueError(
                "independent_rejection_input_norm requires an independent rejection encoder"
            )
        self.independent_rejection_input_norm = independent_rejection_input_norm
        if self.rejection_feature_dim < 0:
            raise ValueError("rejection_feature_dim must be non-negative")
        if self.rejection_feature_dim > 0:
            # Classification and rejection use separate representations. The
            # branch is opt-in so historical checkpoints remain unchanged.
            hidden_dim = max(feat_dim // 2, self.rejection_feature_dim)
            self.rejection_projector = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.rejection_feature_dim),
            )
            self.rejection_classifier = nn.Linear(
                self.rejection_feature_dim, num_classes
            )
            rejection_hidden = max(self.rejection_feature_dim // 2, 8)
            self.rejection_uncertainty_head = nn.Sequential(
                nn.Linear(self.rejection_feature_dim, rejection_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(rejection_hidden, 1),
            )
        else:
            if self.independent_rejection_encoder_enabled:
                raise ValueError(
                    "independent_rejection_encoder requires rejection_feature_dim > 0"
                )
            self.rejection_projector = None
            self.rejection_classifier = None
            self.rejection_uncertainty_head = None
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
        )
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
        # Build the optional encoder after every legacy module. Random
        # same-backbone initialization is additionally RNG-isolated below.
        self.rejection_encoder = (
            (
                copy.deepcopy(self.encoder)
                if independent_rejection_encoder_init == "shared_backbone"
                else self._build_independent_rejection_encoder(
                    backbone=backbone,
                    pretrained=pretrained,
                    cifar_stem=cifar_stem,
                    feat_dim=feat_dim,
                    init_mode=independent_rejection_encoder_init,
                )
            )
            if self.independent_rejection_encoder_enabled
            else None
        )
        self.rejection_input_norm = (
            nn.LayerNorm(feat_dim)
            if self.independent_rejection_encoder_enabled
            and independent_rejection_input_norm == "layernorm"
            else nn.Identity()
        )

    @staticmethod
    def _build_independent_rejection_encoder(
        *,
        backbone: str,
        pretrained: bool,
        cifar_stem: bool,
        feat_dim: int,
        init_mode: str,
    ) -> nn.Module:
        if init_mode == "scratch_cnn":
            return IndependentRejectionEncoder(feat_dim)
        if init_mode != "random_backbone":
            raise ValueError(f"Unsupported independent rejection init: {init_mode}")

        # Same architecture and parameter count as the copied-backbone arm,
        # but initialized randomly. Preserve the global RNG stream so creating
        # this optional encoder cannot change legacy heads or later sampling.
        with torch.random.fork_rng(devices=[]):
            if backbone == "mobilenet_v3_small":
                return MobileNetBackbone(pretrained=False)
            return ResNetBackbone(
                backbone, pretrained=False, cifar_stem=cifar_stem
            )

    def forward(self, x: torch.Tensor, stochastic: bool = False):
        feat = self.encoder(x)
        feat = F.dropout(feat, p=self.dropout_p, training=stochastic)
        logits = self.classifier(feat)
        if self.rejection_projector is None:
            rejection_raw_input = feat
            rejection_input = feat
            rejection_features = feat
            rejection_logits = None
            uncertainty = torch.sigmoid(self.uncertainty_head(feat)).squeeze(-1)
        else:
            rejection_raw_input = (
                self.rejection_encoder(x)
                if self.rejection_encoder is not None
                else feat
            )
            rejection_input = self.rejection_input_norm(rejection_raw_input)
            rejection_input = F.dropout(
                rejection_input, p=self.dropout_p, training=stochastic
            )
            rejection_features = self.rejection_projector(rejection_input)
            rejection_logits = self.rejection_classifier(rejection_features)
            uncertainty = torch.sigmoid(
                self.rejection_uncertainty_head(rejection_features)
            ).squeeze(-1)
        proj = F.normalize(self.projector(feat), dim=-1)
        return {
            "logits": logits,
            "features": feat,
            "rejection_raw_input": rejection_raw_input,
            "rejection_input": rejection_input,
            "rejection_features": rejection_features,
            "rejection_logits": rejection_logits,
            "proj": proj,
            "uncertainty": uncertainty,
        }

    def rejection_forward(self, features: torch.Tensor):
        """Apply the legacy shared-feature rejection projector.

        For an independent image encoder, use ``rejection_head_forward`` on
        the rejection embedding returned by ``forward`` instead.
        """
        if self.rejection_projector is None:
            uncertainty = torch.sigmoid(self.uncertainty_head(features)).squeeze(-1)
            return features, None, uncertainty
        if self.rejection_encoder is not None:
            raise RuntimeError(
                "independent rejection mode needs image inputs; use the "
                "rejection embedding returned by forward"
            )
        rejection_features = self.rejection_projector(features)
        return self.rejection_head_forward(rejection_features)

    def rejection_head_forward(self, rejection_features: torch.Tensor):
        """Apply rejection classifier and uncertainty head to embeddings."""
        if self.rejection_projector is None:
            uncertainty = torch.sigmoid(self.uncertainty_head(rejection_features)).squeeze(-1)
            return rejection_features, None, uncertainty
        rejection_logits = self.rejection_classifier(rejection_features)
        uncertainty = torch.sigmoid(
            self.rejection_uncertainty_head(rejection_features)
        ).squeeze(-1)
        return rejection_features, rejection_logits, uncertainty

    @torch.no_grad()
    def mc_predict(self, x: torch.Tensor, mc_samples: int = 8):
        logits_list = []
        prob_list = []
        uncertainty_list = []
        for _ in range(mc_samples):
            out = self.forward(x, stochastic=True)
            logits_list.append(out["logits"].unsqueeze(0))
            prob_list.append(out["logits"].softmax(dim=-1).unsqueeze(0))
            uncertainty_list.append(out["uncertainty"].unsqueeze(0))
        logits = torch.cat(logits_list, dim=0)
        probs = torch.cat(prob_list, dim=0)
        mean_logits = logits.mean(dim=0)
        mean_probs = probs.mean(dim=0)
        predictive_entropy = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(dim=-1)
        expected_entropy = -(
            probs * probs.clamp_min(1e-8).log()
        ).sum(dim=-1).mean(dim=0)
        # Predictive entropy decomposes into expected data uncertainty plus
        # mutual information, the standard MC-Dropout epistemic estimate.
        epistemic = (predictive_entropy - expected_entropy).clamp_min(0.0)
        head_uncertainty = torch.cat(uncertainty_list, dim=0).mean(dim=0)
        return {
            "mean_logits": mean_logits,
            "mean_probs": mean_probs,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "epistemic": epistemic,
            "aleatoric": expected_entropy,
            "head_uncertainty": head_uncertainty,
        }


def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    proj_dim: int = 128,
    dropout: float = 0.2,
    pretrained: bool = False,
    cifar_stem: bool = False,
    rejection_feature_dim: int = 0,
    independent_rejection_encoder: bool = False,
    independent_rejection_encoder_init: str = "shared_backbone",
    independent_rejection_input_norm: str = "none",
) -> UKDNet:
    return UKDNet(
        num_classes=num_classes,
        backbone=backbone,
        proj_dim=proj_dim,
        dropout=dropout,
        pretrained=pretrained,
        cifar_stem=cifar_stem,
        rejection_feature_dim=rejection_feature_dim,
        independent_rejection_encoder=independent_rejection_encoder,
        independent_rejection_encoder_init=independent_rejection_encoder_init,
        independent_rejection_input_norm=independent_rejection_input_norm,
    )
