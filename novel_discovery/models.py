from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = False) -> None:
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


class UKDNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        proj_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if backbone == "mobilenet_v3_small":
            self.encoder = MobileNetBackbone(pretrained=pretrained)
        else:
            self.encoder = ResNetBackbone(backbone, pretrained=pretrained)
        self.dropout_p = dropout
        feat_dim = self.encoder.out_dim
        self.classifier = nn.Linear(feat_dim, num_classes)
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

    def forward(self, x: torch.Tensor, stochastic: bool = False):
        feat = self.encoder(x)
        feat = F.dropout(feat, p=self.dropout_p, training=stochastic)
        logits = self.classifier(feat)
        uncertainty = torch.sigmoid(self.uncertainty_head(feat)).squeeze(-1)
        proj = F.normalize(self.projector(feat), dim=-1)
        return {
            "logits": logits,
            "features": feat,
            "proj": proj,
            "uncertainty": uncertainty,
        }

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
) -> UKDNet:
    return UKDNet(
        num_classes=num_classes,
        backbone=backbone,
        proj_dim=proj_dim,
        dropout=dropout,
        pretrained=pretrained,
    )
