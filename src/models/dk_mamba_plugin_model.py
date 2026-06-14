from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DK_AC_plugin import (
    DualViewSupplementPlugin,
    FeatureKnowledgeCalibrationPlugin,
)
from .mi_mamba_echo_prime_text_video_model import (
    ClsMLP,
    KnowledgeFusion,
    MIMambaEchoPrimeTextVideo,
)


class KnowledgeCalibrationPlugin(nn.Module):
    """
    First calibration plugin: decode the backbone features, inject activated
    knowledge at every decoder stage, and pool the final decoded feature.

    The decoder stages are passed to ``forward`` instead of owned by this
    plugin, so the plugin can be reused with another compatible backbone.
    """

    def __init__(
        self,
        decoder_feature_dims: Sequence[int],
        knowledge_dim: int,
        pool_size: tuple[int, int, int] = (4, 4, 4),
        fusion_layers: Sequence[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if not decoder_feature_dims:
            raise ValueError("decoder_feature_dims must contain at least one stage")

        self.decoder_feature_dims = tuple(decoder_feature_dims)
        self.pool_size = pool_size
        self.output_dim = self.decoder_feature_dims[-1] * math.prod(pool_size)

        if fusion_layers is None:
            fusion_layers = [
                KnowledgeFusion(feature_dim, knowledge_dim)
                for feature_dim in self.decoder_feature_dims
            ]
        if len(fusion_layers) != len(self.decoder_feature_dims):
            raise ValueError("fusion_layers and decoder_feature_dims must have the same length")

        self.fusion_layers = nn.ModuleList(fusion_layers)

    def forward(
        self,
        hidden_feature: torch.Tensor,
        skip_features: Sequence[torch.Tensor],
        decoder_stages: Sequence[nn.Module],
        knowledge_vector: torch.Tensor,
    ) -> torch.Tensor:
        if len(skip_features) != len(self.fusion_layers):
            raise ValueError("skip_features and fusion_layers must have the same length")
        if len(decoder_stages) != len(self.fusion_layers):
            raise ValueError("decoder_stages and fusion_layers must have the same length")

        feature = hidden_feature
        for decoder, skip_feature, fusion in zip(
            decoder_stages, skip_features, self.fusion_layers
        ):
            feature = decoder(feature, skip_feature)
            feature = fusion(feature, knowledge_vector)

        pooled_feature = F.adaptive_avg_pool3d(feature, self.pool_size)
        return pooled_feature.flatten(start_dim=1)


class DKMambaPluginModel(MIMambaEchoPrimeTextVideo):
    """
    Plugin-based variant of MIMambaEchoPrimeTextVideo.

    Custom calibration plugins must expose ``output_dim`` and implement the
    same forward signatures as the default plugins.
    """

    def __init__(
        self,
        *args,
        knowledge_calibration_plugin: nn.Module | None = None,
        dual_view_supplement_plugin: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if knowledge_calibration_plugin is None:
            knowledge_calibration_plugin = KnowledgeCalibrationPlugin(
                decoder_feature_dims=(
                    self.feat_size[3],
                    self.feat_size[2],
                    self.feat_size[1],
                ),
                knowledge_dim=self.embed_dim,
            )
        self.knowledge_calibration_plugin = knowledge_calibration_plugin

        main_feature_dim = self._get_plugin_output_dim(
            self.knowledge_calibration_plugin,
            "knowledge_calibration_plugin",
        )
        if dual_view_supplement_plugin is None:
            dual_view_supplement_plugin = DualViewSupplementPlugin(
                main_feature_dim=main_feature_dim,
                supplement_feature_dim=self.embed_dim,
            )
        self.dual_view_supplement_plugin = dual_view_supplement_plugin

        classification_input_dim = self._get_plugin_output_dim(
            self.dual_view_supplement_plugin,
            "dual_view_supplement_plugin",
        )
        self.classification_head = ClsMLP(in_dim=classification_input_dim, out_dim=1)

        # The plugin owns the knowledge fusion layers in this model.
        del self.knowledge_fusions

    @staticmethod
    def _get_plugin_output_dim(plugin: nn.Module, plugin_name: str) -> int:
        output_dim = getattr(plugin, "output_dim", None)
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"{plugin_name} must expose a positive integer output_dim")
        return output_dim

    def forward(
        self,
        x_a2c: torch.Tensor,
        x_a4c: torch.Tensor,
        return_features: bool = False,
    ):
        batch_size = x_a2c.shape[0]

        video_feat_a2c = self.video_encoder(x_a2c)
        video_feat_a4c = self.video_encoder(x_a4c)

        with torch.no_grad():
            text_features_all = self.text_encoder(self.prompts_all).expand(
                batch_size, -1, -1
            )

        sim_a2c = F.cosine_similarity(
            video_feat_a2c.unsqueeze(1), text_features_all, dim=-1
        )
        sim_a4c = F.cosine_similarity(
            video_feat_a4c.unsqueeze(1), text_features_all, dim=-1
        )
        activation_weights = torch.max(sim_a2c, sim_a4c).clamp(min=0)
        knowledge_sequence = text_features_all * activation_weights.unsqueeze(-1)
        knowledge_sequence = knowledge_sequence + self.positional_embedding_knowledge
        knowledge_vector = knowledge_sequence.mean(dim=1)

        outs = self.mamba_encoder(x_a4c)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])

        calibrated_feature = self.knowledge_calibration_plugin(
            hidden_feature=enc_hidden,
            skip_features=(enc4, enc3, enc2),
            decoder_stages=(self.decoder5, self.decoder4, self.decoder3),
            knowledge_vector=knowledge_vector,
        )
        fused_feature = self.dual_view_supplement_plugin(
            main_feature=calibrated_feature,
            supplement_feature=video_feat_a2c,
        )

        logits = self.classification_head(fused_feature)
        if return_features:
            return logits, fused_feature
        return logits


# Short alias for configuration files that use the filename as the model name.
DKMambaPlugin = DKMambaPluginModel


__all__ = [
    "KnowledgeCalibrationPlugin",
    "FeatureKnowledgeCalibrationPlugin",
    "DualViewSupplementPlugin",
    "DKMambaPluginModel",
    "DKMambaPlugin",
]
