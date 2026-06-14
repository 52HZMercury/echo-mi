from __future__ import annotations

import torch
import torch.nn as nn


class FeatureKnowledgeCalibrationPlugin(nn.Module):
    """Inject a knowledge vector into an already pooled backbone feature."""

    def __init__(
        self,
        feature_dim: int,
        knowledge_dim: int,
        initial_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.knowledge_dim = knowledge_dim
        self.output_dim = feature_dim
        self.knowledge_projection = nn.Linear(knowledge_dim, feature_dim)
        self.gate = nn.Parameter(torch.full((feature_dim,), initial_scale))

    def forward(
        self,
        feature: torch.Tensor,
        knowledge_vector: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 2 or knowledge_vector.ndim != 2:
            raise ValueError("Both plugin inputs must be 2D tensors shaped [B, C]")
        if feature.shape[0] != knowledge_vector.shape[0]:
            raise ValueError("Both plugin inputs must have the same batch size")
        if feature.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got {feature.shape[1]}"
            )
        if knowledge_vector.shape[1] != self.knowledge_dim:
            raise ValueError(
                f"Expected knowledge dim {self.knowledge_dim}, "
                f"got {knowledge_vector.shape[1]}"
            )

        calibrated_knowledge = self.knowledge_projection(knowledge_vector)
        return feature + self.gate * calibrated_knowledge


class DualViewSupplementPlugin(nn.Module):
    """Supplement a calibrated main-view feature with another view."""

    def __init__(self, main_feature_dim: int, supplement_feature_dim: int) -> None:
        super().__init__()
        self.main_feature_dim = main_feature_dim
        self.supplement_feature_dim = supplement_feature_dim
        self.output_dim = main_feature_dim + supplement_feature_dim

    def forward(
        self,
        main_feature: torch.Tensor,
        supplement_feature: torch.Tensor,
    ) -> torch.Tensor:
        if main_feature.ndim != 2 or supplement_feature.ndim != 2:
            raise ValueError("Both plugin inputs must be 2D tensors shaped [B, C]")
        if main_feature.shape[0] != supplement_feature.shape[0]:
            raise ValueError("Both plugin inputs must have the same batch size")
        if main_feature.shape[1] != self.main_feature_dim:
            raise ValueError(
                f"Expected main feature dim {self.main_feature_dim}, "
                f"got {main_feature.shape[1]}"
            )
        if supplement_feature.shape[1] != self.supplement_feature_dim:
            raise ValueError(
                f"Expected supplement feature dim {self.supplement_feature_dim}, "
                f"got {supplement_feature.shape[1]}"
            )

        return torch.cat([main_feature, supplement_feature], dim=1)


__all__ = ["FeatureKnowledgeCalibrationPlugin", "DualViewSupplementPlugin"]
