from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .BI_mamba_model import BIVideoMambaEncoder
from .DK_AC_plugin import DKACPlugin


class BIMambaPlugin(nn.Module):
    """Two-view BiMamba classifier with the DK-AC calibration plugins."""

    def __init__(
        self,
        in_chans: int = 3,
        num_outputs: int = 1,
        embed_dim: int = 192,
        depth: int = 6,
        patch_size: Sequence[int] = (2, 16, 16),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout_rate: float = 0.2,
        bimamba_type: str = "v3",
        nslices: int = 8,
        enable_dk_ac: bool = True,
        video_encoder_path: Optional[str] = "./model_weight/echo_prime_encoder.pt",
        text_encoder_path: Optional[str] = "./model_weight/echo_prime_text_encoder.pt",
        frozen_video_encoder: bool = True,
        frozen_text_encoder: bool = True,
        dk_ac_plugin: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.video_encoder = BIVideoMambaEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout_rate=dropout_rate,
            bimamba_type=bimamba_type,
            nslices=nslices,
        )
        base_feature_dim = embed_dim * 2
        self.dk_ac_plugin = dk_ac_plugin or DKACPlugin(
            main_feature_dim=base_feature_dim,
            enabled=enable_dk_ac,
            video_encoder_path=video_encoder_path,
            text_encoder_path=text_encoder_path,
            frozen_video_encoder=frozen_video_encoder,
            frozen_text_encoder=frozen_text_encoder,
        )
        fused_dim = self._plugin_output_dim(self.dk_ac_plugin, "dk_ac_plugin")
        self.classification_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(fused_dim, num_outputs),
        )

    @staticmethod
    def _plugin_output_dim(plugin: nn.Module, name: str) -> int:
        output_dim = getattr(plugin, "output_dim", None)
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"{name} must expose a positive integer output_dim")
        return output_dim

    def forward(
        self,
        x_a2c: torch.Tensor,
        x_a4c: torch.Tensor,
        return_features: bool = False,
    ):
        a2c_feature = self.video_encoder(x_a2c)
        a4c_feature = self.video_encoder(x_a4c)
        base_feature = torch.cat([a4c_feature, a2c_feature], dim=1)
        fused_feature = self.dk_ac_plugin(
            base_feature,
            x_a2c,
            x_a4c,
        )
        logits = self.classification_head(fused_feature)
        if return_features:
            return logits, fused_feature
        return logits


BI_mamba_plugin = BIMambaPlugin


__all__ = ["BIMambaPlugin", "BI_mamba_plugin"]
