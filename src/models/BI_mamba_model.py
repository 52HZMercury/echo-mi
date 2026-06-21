from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba


class BIVideoMambaLayer(nn.Module):
    """Pre-norm residual BiMamba layer compatible with mamba_simple.Mamba."""

    def __init__(
        self,
        embed_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout_rate: float,
        bimamba_type: str,
        nslices: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mamba = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type=bimamba_type,
            nslices=nslices,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.dropout(self.mamba(self.norm(tokens)))


class BIVideoMambaEncoder(nn.Module):
    """Patchify a video and encode its token sequence with BiMamba blocks."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        patch_size: Sequence[int] = (2, 16, 16),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        norm_epsilon: float = 1e-5,
        dropout_rate: float = 0.0,
        bimamba_type: str = "v3",
        nslices: int = 8,
    ) -> None:
        super().__init__()
        if len(patch_size) != 3:
            raise ValueError("patch_size must contain temporal, height, and width")

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = tuple(patch_size)
        self.nslices = nslices
        self.patch_embed = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.layers = nn.ModuleList(
            [
                BIVideoMambaLayer(
                    embed_dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout_rate=dropout_rate,
                    bimamba_type=bimamba_type,
                    nslices=nslices,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(
                f"video must have shape [B, C, T, H, W], got {tuple(video.shape)}"
            )
        if video.shape[1] != self.in_chans:
            raise ValueError(
                f"video must have {self.in_chans} channels, got {video.shape[1]}"
            )

        tokens = self.patch_embed(video).flatten(2).transpose(1, 2)
        if tokens.shape[1] % self.nslices != 0:
            raise ValueError(
                f"Patch token count {tokens.shape[1]} must be divisible by "
                f"nslices={self.nslices}. Adjust nslices or patch_size."
            )
        hidden_states = self.dropout(tokens)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states).mean(dim=1)


class BIMamba(nn.Module):
    """Shared-weight, two-view BiMamba classifier for echocardiography videos."""

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
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 2, num_outputs),
        )

    def forward(
        self,
        x_a2c: torch.Tensor,
        x_a4c: torch.Tensor,
        return_features: bool = False,
    ):
        a2c_feature = self.video_encoder(x_a2c)
        a4c_feature = self.video_encoder(x_a4c)
        fused_feature = torch.cat([a4c_feature, a2c_feature], dim=1)
        logits = self.classification_head(fused_feature)
        if return_features:
            return logits, fused_feature
        return logits


BI_mamba = BIMamba


__all__ = ["BIVideoMambaLayer", "BIVideoMambaEncoder", "BIMamba", "BI_mamba"]
