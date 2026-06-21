from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.echoprime_encoders import EchoPrimeTextEncoder, EchoPrimeVideoEncoder
from src.utils.prompts import (
    A2C_SPECIFIC_KNOWLEDGE,
    A4C_SPECIFIC_KNOWLEDGE,
    COMMON_KNOWLEDGE,
)


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


class DKACPlugin(nn.Module):
    """
    Complete, switchable DK-AC plugin.

    EchoPrimeTextEncoder produces activated knowledge for first-stage feature
    calibration. EchoPrimeVideoEncoder extracts the A2C feature used by the
    second-stage feature supplement.
    """

    def __init__(
        self,
        main_feature_dim: int,
        enabled: bool = True,
        video_encoder_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        frozen_video_encoder: bool = True,
        frozen_text_encoder: bool = True,
        echo_feature_dim: int = 512,
        echo_video_frames: int = 16,
        prompts: Optional[Sequence[str]] = None,
        initial_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.main_feature_dim = main_feature_dim
        self.echo_feature_dim = echo_feature_dim
        self.echo_video_frames = echo_video_frames
        self.frozen_video_encoder = frozen_video_encoder
        self.frozen_text_encoder = frozen_text_encoder

        if not enabled:
            self.output_dim = main_feature_dim
            return
        if video_encoder_path is None or text_encoder_path is None:
            raise ValueError(
                "video_encoder_path and text_encoder_path are required when "
                "DK-AC is enabled"
            )

        self.video_encoder = EchoPrimeVideoEncoder(
            pretrained_path=video_encoder_path,
            frozen=frozen_video_encoder,
        )
        self.text_encoder = EchoPrimeTextEncoder(
            pretrained_path=text_encoder_path,
            frozen=frozen_text_encoder,
        )
        self.prompts = list(
            prompts
            if prompts is not None
            else COMMON_KNOWLEDGE
            + A2C_SPECIFIC_KNOWLEDGE
            + A4C_SPECIFIC_KNOWLEDGE
        )
        self.positional_embedding = nn.Parameter(
            torch.randn(1, len(self.prompts), echo_feature_dim)
        )
        self.knowledge_calibration = FeatureKnowledgeCalibrationPlugin(
            feature_dim=main_feature_dim,
            knowledge_dim=echo_feature_dim,
            initial_scale=initial_scale,
        )
        self.a2c_supplement = DualViewSupplementPlugin(
            main_feature_dim=main_feature_dim,
            supplement_feature_dim=echo_feature_dim,
        )
        self.output_dim = self.a2c_supplement.output_dim

    def _prepare_echo_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 4:
            video = video.unsqueeze(2).expand(
                -1, -1, self.echo_video_frames, -1, -1
            )
        if video.ndim != 5:
            raise ValueError(
                f"EchoPrime input must be [B, C, T, H, W], got {tuple(video.shape)}"
            )
        if video.shape[1] == 1:
            video = video.expand(-1, 3, -1, -1, -1)
        elif video.shape[1] != 3:
            raise ValueError(
                f"EchoPrime input must have 1 or 3 channels, got {video.shape[1]}"
            )

        source_frames = video.shape[2]
        if source_frames != self.echo_video_frames:
            if source_frames < self.echo_video_frames:
                indices = (
                    torch.arange(self.echo_video_frames, device=video.device)
                    % source_frames
                )
            else:
                indices = torch.linspace(
                    0,
                    source_frames - 1,
                    steps=self.echo_video_frames,
                    device=video.device,
                ).long()
            video = video.index_select(2, indices)
        return video

    def _encode_text(self, batch_size: int) -> torch.Tensor:
        context = torch.no_grad() if self.frozen_text_encoder else nullcontext()
        with context:
            return self.text_encoder(self.prompts).expand(batch_size, -1, -1)

    def forward(
        self,
        main_feature: torch.Tensor,
        x_a2c: torch.Tensor,
        x_a4c: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return main_feature

        video_context = (
            torch.no_grad() if self.frozen_video_encoder else nullcontext()
        )
        with video_context:
            echo_a2c = self.video_encoder(self._prepare_echo_video(x_a2c))
            echo_a4c = self.video_encoder(self._prepare_echo_video(x_a4c))
        text_features = self._encode_text(main_feature.shape[0])

        sim_a2c = F.cosine_similarity(echo_a2c.unsqueeze(1), text_features, dim=-1)
        sim_a4c = F.cosine_similarity(echo_a4c.unsqueeze(1), text_features, dim=-1)
        activation_weights = torch.maximum(sim_a2c, sim_a4c).clamp_min(0)
        knowledge_sequence = text_features * activation_weights.unsqueeze(-1)
        knowledge_sequence = knowledge_sequence + self.positional_embedding
        knowledge_vector = knowledge_sequence.mean(dim=1)

        calibrated_feature = self.knowledge_calibration(
            main_feature,
            knowledge_vector,
        )
        return self.a2c_supplement(calibrated_feature, echo_a2c)


__all__ = [
    "FeatureKnowledgeCalibrationPlugin",
    "DualViewSupplementPlugin",
    "DKACPlugin",
]
