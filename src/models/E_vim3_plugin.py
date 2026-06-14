from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from .E_vim3 import EViM3, EViM3Config, MLP
from .DK_AC_plugin import (
    DualViewSupplementPlugin,
    FeatureKnowledgeCalibrationPlugin,
)


class EViM3Plugin(nn.Module):
    """Two-view E-ViM3 classifier with knowledge and view-supplement plugins."""

    def __init__(
        self,
        config: EViM3Config | None = None,
        num_outputs: int = 1,
        knowledge_dim: int | None = None,
        knowledge_calibration_plugin: nn.Module | None = None,
        dual_view_supplement_plugin: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EViM3Config()
        self.config.validate()
        self.knowledge_dim = knowledge_dim or self.config.final_dim

        self.view_encoder = EViM3(self.config, num_outputs=None)
        self.default_knowledge = nn.Parameter(torch.zeros(1, self.knowledge_dim))

        self.knowledge_calibration_plugin = knowledge_calibration_plugin or (
            FeatureKnowledgeCalibrationPlugin(
                feature_dim=self.config.final_dim,
                knowledge_dim=self.knowledge_dim,
            )
        )
        calibrated_dim = self._plugin_output_dim(
            self.knowledge_calibration_plugin,
            "knowledge_calibration_plugin",
        )

        self.dual_view_supplement_plugin = dual_view_supplement_plugin or (
            DualViewSupplementPlugin(
                main_feature_dim=calibrated_dim,
                supplement_feature_dim=self.config.final_dim,
            )
        )
        fused_dim = self._plugin_output_dim(
            self.dual_view_supplement_plugin,
            "dual_view_supplement_plugin",
        )
        self.classification_head = MLP(
            fused_dim,
            [self.config.final_dim] * (self.config.head_layers - 1)
            + [num_outputs],
        )

    @staticmethod
    def _plugin_output_dim(plugin: nn.Module, name: str) -> int:
        output_dim = getattr(plugin, "output_dim", None)
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"{name} must expose a positive integer output_dim")
        return output_dim

    def _encode_view(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 6:
            batch, clips, channels, frames, height, width = video.shape
            video = video.reshape(batch * clips, channels, frames, height, width)
        else:
            batch = video.shape[0]
            clips = 1

        feature = self.view_encoder.forward_features(video)
        return feature.reshape(batch, clips, -1).mean(dim=1)

    def _prepare_knowledge(
        self,
        knowledge_vector: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if knowledge_vector is None:
            return self.default_knowledge.expand(batch_size, -1)
        if knowledge_vector.shape != (batch_size, self.knowledge_dim):
            raise ValueError(
                f"knowledge_vector must have shape "
                f"[{batch_size}, {self.knowledge_dim}], "
                f"got {tuple(knowledge_vector.shape)}"
            )
        return knowledge_vector

    def forward(
        self,
        x_a2c: torch.Tensor | Mapping[str, torch.Tensor],
        x_a4c: torch.Tensor | None = None,
        knowledge_vector: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        if isinstance(x_a2c, Mapping):
            inputs = x_a2c
            x_a2c = inputs["x_a2c"]
            x_a4c = inputs["x_a4c"]
            knowledge_vector = inputs.get("knowledge_vector", knowledge_vector)
        if x_a4c is None:
            raise ValueError("x_a4c is required for two-view feature supplementation")

        a2c_feature = self._encode_view(x_a2c)
        a4c_feature = self._encode_view(x_a4c)
        knowledge_vector = self._prepare_knowledge(
            knowledge_vector,
            batch_size=a4c_feature.shape[0],
        )

        calibrated_feature = self.knowledge_calibration_plugin(
            a4c_feature,
            knowledge_vector,
        )
        fused_feature = self.dual_view_supplement_plugin(
            calibrated_feature,
            a2c_feature,
        )
        logits = self.classification_head(fused_feature)
        if return_features:
            return logits, fused_feature
        return logits


E_vim3_plugin = EViM3Plugin


__all__ = ["EViM3Plugin", "E_vim3_plugin"]
