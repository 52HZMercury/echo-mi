from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from .DK_AC_plugin import DKACPlugin
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
        frame_chunk_size: int = 4,
        enable_dk_ac: bool = True,
        video_encoder_path: Optional[str] = "./model_weight/echo_prime_encoder.pt",
        text_encoder_path: Optional[str] = "./model_weight/echo_prime_text_encoder.pt",
        frozen_video_encoder: bool = True,
        frozen_text_encoder: bool = True,
        dk_ac_plugin: Optional[nn.Module] = None,
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
            frame_chunk_size=frame_chunk_size,
        )
        hidden_dim = int(VARIANT_CONFIGS[variant]["hidden_dim"])
        self.feature_pool = nn.AdaptiveAvgPool2d(1)
        self.dk_ac_plugin = dk_ac_plugin or DKACPlugin(
            main_feature_dim=hidden_dim,
            enabled=enable_dk_ac,
            video_encoder_path=video_encoder_path,
            text_encoder_path=text_encoder_path,
            frozen_video_encoder=frozen_video_encoder,
            frozen_text_encoder=frozen_text_encoder,
        )
        fused_dim = self._plugin_output_dim(self.dk_ac_plugin, "dk_ac_plugin")
        self.classifier = nn.Linear(fused_dim, num_classes)

    @staticmethod
    def _plugin_output_dim(plugin: nn.Module, name: str) -> int:
        output_dim = getattr(plugin, "output_dim", None)
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"{name} must expose a positive integer output_dim")
        return output_dim

    def forward(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        return_features: bool = False,
    ):
        echo_view_a = view_a
        echo_view_b = view_b
        view_a, view_b, batch_size, frames = self._prepare_video_views(
            view_a, view_b
        )
        main_feature = self._forward_fused_feature(view_a, view_b)
        main_feature = self._aggregate_frames(main_feature, batch_size, frames)
        fused_feature = self.dk_ac_plugin(
            main_feature,
            echo_view_a,
            echo_view_b,
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
