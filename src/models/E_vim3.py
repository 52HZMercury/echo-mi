"""Standalone E-ViM3 model.

This file contains the complete model definition and does not import any other
file from this repository. Runtime dependencies:

    pip install torch mamba-ssm

Input shape:
    [batch, channels, frames, height, width]

Example:
    model = e_vim3_p4(num_outputs=1).cuda()
    video = torch.randn(2, 1, 64, 112, 112, device="cuda")
    prediction = model(video)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError as error:
    raise ImportError(
        "E_vim3.py requires the third-party package 'mamba-ssm'. "
        "Install mamba-ssm and its matching causal-conv1d package first."
    ) from error


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: list[int],
        activation: type[nn.Module] = nn.SiLU,
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, features in enumerate(out_features):
            layers.append(nn.Linear(in_features, features))
            if index < len(out_features) - 1 or final_activation:
                layers.append(activation())
            in_features = features
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchEmbedding3D(nn.Module):
    def __init__(
        self,
        video_shape: tuple[int, int, int],
        patch_shape: tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.video_shape = video_shape
        self.patch_shape = patch_shape
        self.grid_size = tuple(
            video_size // patch_size
            for video_size, patch_size in zip(video_shape, patch_shape)
        )
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 4, 1)


class ResidualMamba(nn.Module):
    """Pre-norm residual Mamba layer."""

    def __init__(
        self,
        embed_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        drop_path: float,
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
        self.dropout = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaStack(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        d_state: int,
        d_conv: int,
        expand: int,
        drop_path: float,
        bimamba_type: str,
        nslices: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResidualMamba(
                    embed_dim,
                    d_state,
                    d_conv,
                    expand,
                    drop_path,
                    bimamba_type,
                    nslices,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Mamba3DBlock(nn.Module):
    """Six directional scans over the three axes of a video token volume."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        d_state: int,
        d_conv: int,
        expand: int,
        drop_path: float,
        bimamba_type: str,
        nslices: int,
    ) -> None:
        super().__init__()

        def make_stack() -> MambaStack:
            return MambaStack(
                embed_dim,
                depth,
                d_state,
                d_conv,
                expand,
                drop_path,
                bimamba_type,
                nslices,
            )

        self.hwl = make_stack()
        self.reverse_hwl = make_stack()
        self.lwh = make_stack()
        self.reverse_lwh = make_stack()
        self.lhw = make_stack()
        self.reverse_lhw = make_stack()

    @staticmethod
    def _bidirectional_scan(
        x: torch.Tensor,
        forward_mamba: nn.Module,
        reverse_mamba: nn.Module,
    ) -> torch.Tensor:
        x = forward_mamba(x)
        return reverse_mamba(x.flip(1)).flip(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, height, width, dim = x.shape

        x = x.permute(0, 2, 3, 1, 4).reshape(batch, height * width * length, dim)
        x = self._bidirectional_scan(x, self.hwl, self.reverse_hwl)
        x = x.reshape(batch, height, width, length, dim)

        x = x.permute(0, 3, 2, 1, 4).reshape(batch, length * width * height, dim)
        x = self._bidirectional_scan(x, self.lwh, self.reverse_lwh)
        x = x.reshape(batch, length, width, height, dim)

        x = x.permute(0, 1, 3, 2, 4).reshape(batch, length * height * width, dim)
        x = self._bidirectional_scan(x, self.lhw, self.reverse_lhw)
        return x.reshape(batch, length, height, width, dim)


@dataclass
class EViM3Config:
    """Default E-ViM3-p4 configuration."""

    in_channels: int = 1
    video_length: int = 64
    image_size: int = 112
    time_patch_size: int = 2
    patch_size: int = 4
    embed_dim: int = 192
    l_reg_num: int = 1
    h_reg_num: int = 1
    w_reg_num: int = 1
    n_mamba_per_block: int = 1
    macro_block_num: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    drop_path: float = 0.0
    bimamba_type: str = "v3"
    nslices: int = 1
    final_aggregate_mode: str = "all_concat"
    final_dim: int = 512
    head_layers: int = 3

    def validate(self) -> None:
        positive_fields = (
            "in_channels",
            "video_length",
            "image_size",
            "time_patch_size",
            "patch_size",
            "embed_dim",
            "n_mamba_per_block",
            "macro_block_num",
            "d_state",
            "d_conv",
            "expand",
            "nslices",
            "final_dim",
            "head_layers",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        if self.video_length % self.time_patch_size != 0:
            raise ValueError("video_length must be divisible by time_patch_size")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if min(self.l_reg_num, self.h_reg_num, self.w_reg_num) < 0:
            raise ValueError("register token counts cannot be negative")
        if self.bimamba_type != "v3":
            raise ValueError("this project requires bimamba_type='v3'")
        if self.final_aggregate_mode not in {
            "all_mean",
            "all_max",
            "all_concat",
            "all_enclosure_mean",
            "two_corner_concat",
        }:
            raise ValueError(
                f"unsupported final_aggregate_mode: {self.final_aggregate_mode}"
            )


class EViM3Backbone(nn.Module):
    """E-ViM3 patch encoder with learnable enclosure and register tokens."""

    def __init__(self, config: EViM3Config) -> None:
        super().__init__()
        self.config = config
        patch_shape = (
            config.time_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.patch_embed = PatchEmbedding3D(
            (config.video_length, config.image_size, config.image_size),
            patch_shape,
            config.in_channels,
            config.embed_dim,
        )

        length, height, width = self.patch_embed.grid_size
        self.enclosed_grid_size = (
            length + config.l_reg_num + 2,
            height + config.h_reg_num + 2,
            width + config.w_reg_num + 2,
        )

        self.face_tokens = nn.Parameter(torch.zeros(6, 1, 1, 1, 1, config.embed_dim))
        self.edge_tokens = nn.Parameter(torch.zeros(12, 1, 1, 1, 1, config.embed_dim))
        self.corner_tokens = nn.Parameter(torch.zeros(8, 1, 1, 1, 1, config.embed_dim))
        self.register_token = nn.Parameter(torch.zeros(1, 1, 1, 1, config.embed_dim))

        enclosed_l, enclosed_h, enclosed_w = self.enclosed_grid_size
        self.l_pos_embed = nn.Parameter(
            torch.randn(1, enclosed_l, 1, 1, config.embed_dim) * 0.02
        )
        self.h_pos_embed = nn.Parameter(
            torch.randn(1, 1, enclosed_h, 1, config.embed_dim) * 0.02
        )
        self.w_pos_embed = nn.Parameter(
            torch.randn(1, 1, 1, enclosed_w, config.embed_dim) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                Mamba3DBlock(
                    config.embed_dim,
                    config.n_mamba_per_block,
                    config.d_state,
                    config.d_conv,
                    config.expand,
                    config.drop_path,
                    config.bimamba_type,
                    config.nslices,
                )
                for _ in range(config.macro_block_num)
            ]
        )

        self.register_buffer(
            "l_global_indices",
            self._global_indices(enclosed_l, config.l_reg_num),
            persistent=False,
        )
        self.register_buffer(
            "h_global_indices",
            self._global_indices(enclosed_h, config.h_reg_num),
            persistent=False,
        )
        self.register_buffer(
            "w_global_indices",
            self._global_indices(enclosed_w, config.w_reg_num),
            persistent=False,
        )
        self.register_buffer(
            "real_tokens_mask",
            self._make_real_tokens_mask(),
            persistent=False,
        )

    @staticmethod
    def _global_indices(size: int, register_count: int) -> torch.Tensor:
        return torch.linspace(0, size - 1, register_count + 2).round().long()

    def _make_real_tokens_mask(self) -> torch.Tensor:
        length, height, width = self.enclosed_grid_size
        mask = torch.ones(1, length, height, width, dtype=torch.bool)
        mask[:, self.l_global_indices, :, :] = False
        mask[:, :, self.h_global_indices, :] = False
        mask[:, :, :, self.w_global_indices] = False
        return mask

    def _add_enclosure_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        batch, _, _, _, dim = patches.shape
        length, height, width = self.enclosed_grid_size
        x = (
            self.register_token.to(device=patches.device, dtype=patches.dtype)
            .expand(batch, length, height, width, dim)
            .clone()
        )

        x[:, :1, 1:-1, 1:-1] = self.face_tokens[0]
        x[:, -1:, 1:-1, 1:-1] = self.face_tokens[1]
        x[:, 1:-1, :1, 1:-1] = self.face_tokens[2]
        x[:, 1:-1, -1:, 1:-1] = self.face_tokens[3]
        x[:, 1:-1, 1:-1, :1] = self.face_tokens[4]
        x[:, 1:-1, 1:-1, -1:] = self.face_tokens[5]

        x[:, :1, :1, 1:-1] = self.edge_tokens[0]
        x[:, -1:, :1, 1:-1] = self.edge_tokens[1]
        x[:, :1, -1:, 1:-1] = self.edge_tokens[2]
        x[:, -1:, -1:, 1:-1] = self.edge_tokens[3]
        x[:, :1, 1:-1, :1] = self.edge_tokens[4]
        x[:, -1:, 1:-1, :1] = self.edge_tokens[5]
        x[:, :1, 1:-1, -1:] = self.edge_tokens[6]
        x[:, -1:, 1:-1, -1:] = self.edge_tokens[7]
        x[:, 1:-1, :1, :1] = self.edge_tokens[8]
        x[:, 1:-1, -1:, :1] = self.edge_tokens[9]
        x[:, 1:-1, :1, -1:] = self.edge_tokens[10]
        x[:, 1:-1, -1:, -1:] = self.edge_tokens[11]

        x[:, :1, :1, :1] = self.corner_tokens[0]
        x[:, -1:, :1, :1] = self.corner_tokens[1]
        x[:, :1, -1:, :1] = self.corner_tokens[2]
        x[:, -1:, -1:, :1] = self.corner_tokens[3]
        x[:, :1, :1, -1:] = self.corner_tokens[4]
        x[:, -1:, :1, -1:] = self.corner_tokens[5]
        x[:, :1, -1:, -1:] = self.corner_tokens[6]
        x[:, -1:, -1:, -1:] = self.corner_tokens[7]

        real_mask = self.real_tokens_mask.expand(batch, -1, -1, -1)
        x[real_mask] = patches.reshape(-1, dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._add_enclosure_tokens(self.patch_embed(x))
        x = x + self.l_pos_embed + self.h_pos_embed + self.w_pos_embed
        for block in self.blocks:
            x = block(x)
        return x


class EViM3Aggregator(nn.Module):
    def __init__(self, config: EViM3Config) -> None:
        super().__init__()
        self.mode = config.final_aggregate_mode
        self.final_dim = config.final_dim
        self.norm = nn.LayerNorm(config.embed_dim)

        global_count = (
            (config.l_reg_num + 2)
            * (config.h_reg_num + 2)
            * (config.w_reg_num + 2)
        )
        if self.mode == "all_concat":
            in_features = global_count * config.embed_dim
        elif self.mode == "two_corner_concat":
            in_features = 2 * config.embed_dim
        else:
            in_features = config.embed_dim
        self.projection = MLP(in_features, [config.final_dim], final_activation=True)

    def forward(self, x: torch.Tensor, backbone: EViM3Backbone) -> torch.Tensor:
        if self.mode == "all_enclosure_mean":
            mask = ~backbone.real_tokens_mask.expand(x.shape[0], -1, -1, -1)
            x = self.norm(x[mask].reshape(x.shape[0], -1, x.shape[-1])).mean(1)
        elif self.mode == "two_corner_concat":
            x = torch.stack((x[:, 0, 0, 0], x[:, -1, -1, -1]), dim=1)
            x = self.norm(x).flatten(1)
        else:
            x = x.index_select(1, backbone.l_global_indices)
            x = x.index_select(2, backbone.h_global_indices)
            x = x.index_select(3, backbone.w_global_indices)
            x = self.norm(x.flatten(1, -2))
            if self.mode == "all_mean":
                x = x.mean(1)
            elif self.mode == "all_max":
                x = x.max(1).values
            elif self.mode == "all_concat":
                x = x.flatten(1)
        return self.projection(x)


class EViM3(nn.Module):
    """Complete standalone E-ViM3 video model."""

    def __init__(
        self,
        config: Optional[EViM3Config] = None,
        num_outputs: Optional[int] = 1,
    ) -> None:
        super().__init__()
        self.config = config or EViM3Config()
        self.config.validate()
        self.num_outputs = num_outputs

        self.backbone = EViM3Backbone(self.config)
        self.aggregate = EViM3Aggregator(self.config)
        self.head = (
            MLP(
                self.config.final_dim,
                [self.config.final_dim] * (self.config.head_layers - 1)
                + [num_outputs],
            )
            if num_outputs is not None
            else nn.Identity()
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return self.aggregate(self.backbone(x), self.backbone)

    def forward(self, x: Union[torch.Tensor, Mapping[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, Mapping):
            x = x["x"]

        if x.ndim == 6:
            batch, clips, channels, frames, height, width = x.shape
            x = x.reshape(batch * clips, channels, frames, height, width)
        else:
            batch = x.shape[0]
            clips = 1

        output = self.head(self.forward_features(x))
        return output.reshape(batch, clips, -1).mean(1)

    def load_checkpoint(
        self,
        checkpoint: Union[str, Path, Mapping[str, Any]],
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                checkpoint = checkpoint[key]
                break
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }
        return self.load_state_dict(state_dict, strict=strict)

    def _validate_input(self, x: torch.Tensor) -> None:
        expected = (
            self.config.in_channels,
            self.config.video_length,
            self.config.image_size,
            self.config.image_size,
        )
        if x.ndim != 5 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected [batch, {expected}], got {tuple(x.shape)}")


def e_vim3_p4(num_outputs: Optional[int] = 1, **overrides: Any) -> EViM3:
    return EViM3(EViM3Config(**overrides), num_outputs=num_outputs)


E_vim3 = EViM3


if __name__ == "__main__":
    model = e_vim3_p4()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(model)
    print(f"Parameters: {parameters:,}")
    print("Expected input: [batch, 1, 64, 112, 112]")
