from __future__ import annotations

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig

from .bimamba import BiMambaLMHeadModel
from .DK_AC_plugin import (
    DualViewSupplementPlugin,
    FeatureKnowledgeCalibrationPlugin,
)


class BiMambaPlugin(nn.Module):
    """
    Two-sequence BiMamba classifier with knowledge and supplement plugins.

    The original ``bimamba.py`` model is a language model. This integration
    pools two BiMamba token sequences into classification features before
    applying the two DK plugins.
    """

    def __init__(
        self,
        config: MambaConfig,
        num_outputs: int = 1,
        knowledge_dim: int | None = None,
        knowledge_calibration_plugin: nn.Module | None = None,
        dual_view_supplement_plugin: nn.Module | None = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_dim = config.d_model
        self.knowledge_dim = knowledge_dim or self.feature_dim

        language_model = BiMambaLMHeadModel(
            config=config,
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
        )
        self.backbone = language_model.backbone
        self.default_knowledge = nn.Parameter(
            torch.zeros(1, self.knowledge_dim, device=device, dtype=dtype)
        )

        self.knowledge_calibration_plugin = knowledge_calibration_plugin or (
            FeatureKnowledgeCalibrationPlugin(self.feature_dim, self.knowledge_dim)
        )
        calibrated_dim = self._plugin_output_dim(
            self.knowledge_calibration_plugin,
            "knowledge_calibration_plugin",
        )
        self.dual_view_supplement_plugin = dual_view_supplement_plugin or (
            DualViewSupplementPlugin(calibrated_dim, self.feature_dim)
        )
        fused_dim = self._plugin_output_dim(
            self.dual_view_supplement_plugin,
            "dual_view_supplement_plugin",
        )
        self.classification_head = nn.Linear(
            fused_dim,
            num_outputs,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _plugin_output_dim(plugin: nn.Module, name: str) -> int:
        output_dim = getattr(plugin, "output_dim", None)
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"{name} must expose a positive integer output_dim")
        return output_dim

    @staticmethod
    def _pool_sequence(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "attention_mask must match the first two hidden-state dimensions"
            )
        weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

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
        primary_input_ids: torch.Tensor,
        supplement_input_ids: torch.Tensor,
        knowledge_vector: torch.Tensor | None = None,
        primary_attention_mask: torch.Tensor | None = None,
        supplement_attention_mask: torch.Tensor | None = None,
        return_features: bool = False,
        **mixer_kwargs,
    ):
        primary_hidden = self.backbone(primary_input_ids, **mixer_kwargs)
        supplement_hidden = self.backbone(supplement_input_ids, **mixer_kwargs)
        primary_feature = self._pool_sequence(
            primary_hidden,
            primary_attention_mask,
        )
        supplement_feature = self._pool_sequence(
            supplement_hidden,
            supplement_attention_mask,
        )
        knowledge_vector = self._prepare_knowledge(
            knowledge_vector,
            batch_size=primary_feature.shape[0],
        )

        calibrated_feature = self.knowledge_calibration_plugin(
            primary_feature,
            knowledge_vector,
        )
        fused_feature = self.dual_view_supplement_plugin(
            calibrated_feature,
            supplement_feature,
        )
        logits = self.classification_head(fused_feature)
        if return_features:
            return logits, fused_feature
        return logits


BiMambaPluginModel = BiMambaPlugin
bimamba_plugin = BiMambaPlugin


__all__ = ["BiMambaPlugin", "BiMambaPluginModel", "bimamba_plugin"]
