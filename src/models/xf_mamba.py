"""Complete high-level XF-Mamba model definition.

XF-Mamba combines a shared VMamba backbone with shallow feature exchange and
deep cross-fusion. The low-level VMamba and selective-scan operators remain in
``models/fusion_vmamba.py`` and its CUDA/Triton dependencies.

Example:
    model = XFMamba(in_channels=1, num_classes=2, variant="small")
    logits = model(view_a, view_b)
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_vmamba import (
    Backbone_VSSM,
    CSSFVSSLayer_v5,
    ShallowFusionBlock_v4,
)


VARIANT_CONFIGS: Dict[str, Dict[str, Union[int, float, list]]] = {
    "tiny": {
        "depths": [2, 2, 8, 2],
        "dims": 96,
        "backbone_drop_path_rate": 0.2,
        "ssm_ratio": 1.0,
        "hidden_dim": 768,
    },
    "small": {
        "depths": [2, 2, 15, 2],
        "dims": 96,
        "backbone_drop_path_rate": 0.3,
        "ssm_ratio": 2.0,
        "hidden_dim": 768,
    },
    "base": {
        "depths": [2, 2, 15, 2],
        "dims": 128,
        "backbone_drop_path_rate": 0.6,
        "ssm_ratio": 2.0,
        "hidden_dim": 1024,
    },
}


class XFMamba(nn.Module):
    """Cross-Fusion Mamba for two-view image classification.

    Args:
        in_channels: Number of channels in each input view.
        num_classes: Number of output classes/tasks.
        variant: VMamba backbone size: ``tiny``, ``small``, or ``base``.
        attention_downsampling: Downsampling factor used by deep cross-fusion.
        fusion_depth: Number of deep cross-fusion blocks.
        fusion_drop_path_rate: Maximum stochastic-depth rate in fusion blocks.
        attn_drop_rate: Dropout rate inside fusion Mamba blocks.
        d_state: State dimension used by fusion Mamba blocks.
        pretrained: Optional VMamba backbone checkpoint path.
    """

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
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if variant not in VARIANT_CONFIGS:
            choices = ", ".join(VARIANT_CONFIGS)
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {choices}")
        if fusion_depth < 1:
            raise ValueError(f"fusion_depth must be positive, got {fusion_depth}")

        config = VARIANT_CONFIGS[variant]
        hidden_dim = int(config["hidden_dim"])

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.variant = variant
        self.frame_chunk_size = frame_chunk_size

        self.mamba_feature_extrac = Backbone_VSSM(
            depths=config["depths"],
            dims=config["dims"],
            drop_path_rate=config["backbone_drop_path_rate"],
            ssm_ratio=config["ssm_ratio"],
            in_chans=3,
            pretrained=str(pretrained) if pretrained is not None else None,
        )

        self.shallow_mamba_fusion = ShallowFusionBlock_v4(
            hidden_dim=hidden_dim,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
        )

        drop_paths = [
            value.item()
            for value in torch.linspace(
                0, fusion_drop_path_rate, steps=fusion_depth
            )
        ]
        self.fusemamba = CSSFVSSLayer_v5(
            hidden_dim=hidden_dim,
            depth=fusion_depth,
            drop_path=drop_paths,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            downsampling=attention_downsampling,
        )

        self.final_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.classifier = nn.Sequential(
            OrderedDict(
                avgpool=nn.AdaptiveAvgPool2d(1),
                flatten=nn.Flatten(1),
                head=nn.Linear(hidden_dim, num_classes),
            )
        )

    def _prepare_view(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"{name} must have shape [batch, channels, height, width], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"{name} must have {self.in_channels} channels, got {x.shape[1]}"
            )

        if self.in_channels == 1:
            return x.expand(-1, 3, -1, -1)
        if self.in_channels == 3:
            return x
        raise ValueError(
            "The VMamba backbone accepts one-channel images (expanded to RGB) "
            "or three-channel images."
        )

    def _prepare_video_views(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
    ):
        if view_a.ndim != view_b.ndim:
            raise ValueError("Both views must have the same number of dimensions")
        if view_a.shape[0] != view_b.shape[0]:
            raise ValueError("Both views must have the same batch size")

        if view_a.ndim == 4:
            batch_size, frames = view_a.shape[0], 1
        elif view_a.ndim == 5:
            batch_size, channels, frames, height, width = view_a.shape
            if view_b.shape[2] != frames:
                raise ValueError("Both video views must have the same frame count")
            view_a = view_a.permute(0, 2, 1, 3, 4).reshape(
                batch_size * frames, channels, height, width
            )
            view_b = view_b.permute(0, 2, 1, 3, 4).reshape(
                batch_size * frames, view_b.shape[1], view_b.shape[3], view_b.shape[4]
            )
        else:
            raise ValueError(
                "Each view must be an image [B, C, H, W] or video "
                f"[B, C, T, H, W], got {tuple(view_a.shape)}"
            )

        return (
            self._prepare_view(view_a, "view_a"),
            self._prepare_view(view_b, "view_b"),
            batch_size,
            frames,
        )

    def _forward_fused_map(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
    ) -> torch.Tensor:
        features_a = self.mamba_feature_extrac(view_a)[-1]
        features_b = self.mamba_feature_extrac(view_b)[-1]
        features_a, features_b = self.shallow_mamba_fusion(
            features_a, features_b
        )
        fused = self.fusemamba(features_a, features_b)
        return self.final_conv(fused)

    def _forward_fused_feature(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = self.frame_chunk_size or view_a.shape[0]
        if chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive or zero")

        features = []
        for start in range(0, view_a.shape[0], chunk_size):
            fused_map = self._forward_fused_map(
                view_a[start : start + chunk_size],
                view_b[start : start + chunk_size],
            )
            features.append(F.adaptive_avg_pool2d(fused_map, 1).flatten(1))
        return torch.cat(features, dim=0)

    @staticmethod
    def _aggregate_frames(
        feature: torch.Tensor,
        batch_size: int,
        frames: int,
    ) -> torch.Tensor:
        return feature.reshape(batch_size, frames, -1).mean(dim=1)

    def forward(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        return_features: bool = False,
    ):
        """Return classification logits for paired image or video views."""
        view_a, view_b, batch_size, frames = self._prepare_video_views(
            view_a, view_b
        )
        fused_feature = self._forward_fused_feature(view_a, view_b)
        fused_feature = self._aggregate_frames(fused_feature, batch_size, frames)
        logits = self.classifier.head(fused_feature)
        if return_features:
            return logits, fused_feature
        return logits

    def load_model_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        strict: bool = True,
    ):
        """Load either a raw state dict or a common wrapped checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        return self.load_state_dict(checkpoint, strict=strict)


def xfmamba_tiny(
    in_channels: int = 1, num_classes: int = 2, **kwargs
) -> XFMamba:
    return XFMamba(
        in_channels=in_channels,
        num_classes=num_classes,
        variant="tiny",
        **kwargs,
    )


def xfmamba_small(
    in_channels: int = 1, num_classes: int = 2, **kwargs
) -> XFMamba:
    return XFMamba(
        in_channels=in_channels,
        num_classes=num_classes,
        variant="small",
        **kwargs,
    )


def xfmamba_base(
    in_channels: int = 1, num_classes: int = 2, **kwargs
) -> XFMamba:
    return XFMamba(
        in_channels=in_channels,
        num_classes=num_classes,
        variant="base",
        **kwargs,
    )


class TwoViewXFMambaTop(XFMamba):
    """Compatibility wrapper using the original repository's argument names."""

    def __init__(
        self,
        in_channels: int,
        outputs: int,
        attention_downsampling: int = 4,
        hidden_dim: Optional[int] = None,
        depth: int = 1,
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        drop_path_rate: float = 0.1,
        pretrained: Optional[Union[str, Path]] = None,
        type: str = "small",
    ) -> None:
        if type not in VARIANT_CONFIGS:
            choices = ", ".join(VARIANT_CONFIGS)
            raise ValueError(f"Unknown type '{type}'. Choose from: {choices}")

        expected_hidden_dim = int(VARIANT_CONFIGS[type]["hidden_dim"])
        if hidden_dim is not None and hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"variant '{type}' requires hidden_dim={expected_hidden_dim}, "
                f"got {hidden_dim}"
            )

        super().__init__(
            in_channels=in_channels,
            num_classes=outputs,
            variant=type,
            attention_downsampling=attention_downsampling,
            fusion_depth=depth,
            fusion_drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            pretrained=pretrained,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xfmamba_small(num_classes=2).to(device).eval()
    view_a = torch.randn(1, 1, 224, 224, device=device)
    view_b = torch.randn(1, 1, 224, 224, device=device)

    with torch.inference_mode():
        output = model(view_a, view_b)

    print(f"device: {device}")
    print(f"view_a: {tuple(view_a.shape)}")
    print(f"view_b: {tuple(view_b.shape)}")
    print(f"logits: {tuple(output.shape)}")
