from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from .DK_AC_plugin import (
    DualViewSupplementPlugin,
    FeatureKnowledgeCalibrationPlugin,
)
from .xf_mamba import VARIANT_CONFIGS, XFMamba


class XFMambaPlugin(XFMamba):
    """XF-Mamba with post-fusion knowledge calibration and view supplementation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        variant: str = "small",
        attention_downsampling: int = 4,
        fusion_depth: int = 1,
        fusion_drop_path_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        pretrained: Optional[Union[str, Path]] = None,
        knowledge_dim: int | None = None,
        knowledge_calibration_plugin: nn.Module | None = None,
        dual_view_supplement_plugin: nn.Module | None = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            variant=variant,
            attention_downsampling=attention_downsampling,
            fusion_depth=fusion_depth,
            fusion_drop_path_rate=fusion_drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            pretrained=pretrained,
        )
        hidden_dim = int(VARIANT_CONFIGS[variant]["hidden_dim"])
        self.knowledge_dim = knowledge_dim or hidden_dim
        self.default_knowledge = nn.Parameter(torch.zeros(1, self.knowledge_dim))
        self.feature_pool = nn.AdaptiveAvgPool2d(1)

        self.knowledge_calibration_plugin = knowledge_calibration_plugin or (
            FeatureKnowledgeCalibrationPlugin(hidden_dim, self.knowledge_dim)
        )
        calibrated_dim = self._plugin_output_dim(
            self.knowledge_calibration_plugin,
            "knowledge_calibration_plugin",
        )
        self.dual_view_supplement_plugin = dual_view_supplement_plugin or (
            DualViewSupplementPlugin(calibrated_dim, hidden_dim)
        )
        fused_dim = self._plugin_output_dim(
            self.dual_view_supplement_plugin,
            "dual_view_supplement_plugin",
        )
        self.classifier = nn.Linear(fused_dim, num_classes)

    @staticmethod
    def _plugin_output_dim(plugin: nn.Module, name: str) -> int:
        output_dim = getattr(plugin, "output_dim", None)
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"{name} must expose a positive integer output_dim")
        return output_dim

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
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        knowledge_vector: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        view_a = self._prepare_view(view_a, "view_a")
        view_b = self._prepare_view(view_b, "view_b")

        features_a = self.mamba_feature_extrac(view_a)[-1]
        features_b = self.mamba_feature_extrac(view_b)[-1]
        features_a, features_b = self.shallow_mamba_fusion(features_a, features_b)

        fused_map = self.final_conv(self.fusemamba(features_a, features_b))
        main_feature = self.feature_pool(fused_map).flatten(1)
        view_a_feature = self.feature_pool(features_a).flatten(1)
        knowledge_vector = self._prepare_knowledge(
            knowledge_vector,
            batch_size=main_feature.shape[0],
        )

        calibrated_feature = self.knowledge_calibration_plugin(
            main_feature,
            knowledge_vector,
        )
        fused_feature = self.dual_view_supplement_plugin(
            calibrated_feature,
            view_a_feature,
        )
        logits = self.classifier(fused_feature)
        if return_features:
            return logits, fused_feature
        return logits


def xfmamba_plugin_tiny(**kwargs) -> XFMambaPlugin:
    return XFMambaPlugin(variant="tiny", **kwargs)


def xfmamba_plugin_small(**kwargs) -> XFMambaPlugin:
    return XFMambaPlugin(variant="small", **kwargs)


def xfmamba_plugin_base(**kwargs) -> XFMambaPlugin:
    return XFMambaPlugin(variant="base", **kwargs)


xf_mamba_plugin = XFMambaPlugin


__all__ = [
    "XFMambaPlugin",
    "xf_mamba_plugin",
    "xfmamba_plugin_tiny",
    "xfmamba_plugin_small",
    "xfmamba_plugin_base",
]
