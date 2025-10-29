from __future__ import annotations

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F
# --- Cross-SSM Fusion + DualMIMamba integration ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip

        return out


# class MambaLayer(nn.Module):
#     def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
#         super().__init__()
#         self.dim = dim
#         self.norm = nn.LayerNorm(dim)
#         # store nslices (may be None)
#         self.nslices = num_slices
#         self.mamba = Mamba(
#             d_model=dim,  # Model dimension d_model
#             d_state=d_state,  # SSM state expansion factor
#             d_conv=d_conv,  # Local convolution width
#             expand=expand,  # Block expansion factor
#             bimamba_type="v3",
#             nslices=num_slices,
#         )
#
#     def forward(self, x):
#         B, C = x.shape[:2]
#         x_skip = x
#         assert C == self.dim
#         n_tokens = x.shape[2:].numel()  # N = D*H*W
#         img_dims = x.shape[2:]
#
#         # Flatten to (B, N, C) but we will transpose for LayerNorm like before
#         x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # (B, N, C)
#
#         # If nslices is set and does not divide n_tokens, pad tokens to next multiple
#         if self.nslices is not None and (n_tokens % self.nslices != 0):
#             # compute padding length to make n_tokens divisible by nslices
#             pad_len = (-n_tokens) % self.nslices  # smallest pad to make divisible
#             # pad on token dimension (dim=1 after transpose)
#             pad_tensor = x_flat.new_zeros(B, pad_len, C)
#             x_flat_padded = torch.cat([x_flat, pad_tensor], dim=1)  # (B, N+pad, C)
#             # normalize and run mamba on padded
#             x_norm = self.norm(x_flat_padded)
#             x_mamba = self.mamba(x_norm)  # (B, N+pad, C)
#             # unpad back to original N
#             x_mamba = x_mamba[:, :n_tokens, :].contiguous()
#         else:
#             x_norm = self.norm(x_flat)
#             x_mamba = self.mamba(x_norm)  # (B, N, C)
#
#         out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
#         out = out + x_skip
#
#         return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        # num_slices_list = [64, 32, 16, 8]
        num_slices_list = [32, 16, 8, 4]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


# Reuse Mamba, MambaEncoder, UnetrBasicBlock from your existing file.

# class CrossSSMFusion(nn.Module):
#     """
#     Cross-SSM Fusion module operating on token/state outputs of two Mamba SSMs.
#     - mode='gated': compute cross-view channel gates from token-averaged SSM states (efficient).
#     - mode='attn' : compute lightweight cross-attention on token states (optionally on pooled resolution).
#     Inputs:
#         feat_a, feat_b: [B, C, D, H, W]
#     Outputs:
#         a_out, b_out: [B, C, D, H, W]
#     """
#
#     def __init__(
#             self,
#             dim: int,
#             d_state: int = 16,
#             d_conv: int = 4,
#             expand: int = 2,
#             nslices: Optional[int] = None,
#             reduction: int = 8,
#             attn_pool: Optional[Tuple[int, int, int]] = None,
#             mode: str = "gated",
#     ):
#         super().__init__()
#         assert mode in ("gated", "attn")
#         self.dim = dim
#         self.mode = mode
#         self.attn_pool = attn_pool
#
#         # ✅ 修复 nslices 为 None 的问题
#         nslices = 1 if nslices is None else nslices
#
#         # Two Mamba SSM instances (one per view)
#         self.mamba_a = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, bimamba_type="v3",
#                              nslices=nslices)
#         self.mamba_b = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, bimamba_type="v3",
#                              nslices=nslices)
#
#         # token-layernorm (applies to (B,N,C))
#         self.norm_a = nn.LayerNorm(dim)
#         self.norm_b = nn.LayerNorm(dim)
#
#         # gating MLPs (cross-view gates)
#         mid = max(dim // reduction, 8)
#         self.gate_a = nn.Sequential(
#             nn.Linear(dim, mid),
#             nn.ReLU(inplace=True),
#             nn.Linear(mid, dim),
#             nn.Sigmoid(),
#         )
#         self.gate_b = nn.Sequential(
#             nn.Linear(dim, mid),
#             nn.ReLU(inplace=True),
#             nn.Linear(mid, dim),
#             nn.Sigmoid(),
#         )
#
#         # attention-mode projections
#         if self.mode == "attn":
#             proj_dim = max(dim // reduction, 8)
#             self.q_proj = nn.Linear(dim, proj_dim)
#             self.k_proj = nn.Linear(dim, proj_dim)
#             self.v_proj = nn.Linear(dim, proj_dim)
#             self.out_proj = nn.Linear(proj_dim, dim)
#
#         # small residual scaling to stabilize early training
#         self.res_scale = nn.Parameter(torch.tensor(0.1))
#
#     def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
#         """
#         feat_a, feat_b: [B, C, D, H, W]
#         returns: a_out, b_out (same shape)
#         """
#         B, C, D, H, W = feat_a.shape
#         assert C == self.dim
#
#         N = D * H * W
#         # flatten tokens -> (B, N, C)
#         a_flat = feat_a.view(B, C, N).transpose(1, 2)
#         b_flat = feat_b.view(B, C, N).transpose(1, 2)
#
#         # normalize tokens
#         a_norm = self.norm_a(a_flat)
#         b_norm = self.norm_b(b_flat)
#
#         # run per-view Mamba SSM (expects (B, N, C) -> (B, N, C))
#         a_ssm = self.mamba_a(a_norm)
#         b_ssm = self.mamba_b(b_norm)
#
#         if self.mode == "gated":
#             # token average pooling -> (B, C)
#             a_pool = a_ssm.mean(dim=1)
#             b_pool = b_ssm.mean(dim=1)
#
#             # cross-derived gates
#             g_a = self.gate_a(b_pool).unsqueeze(1)  # gate for A computed from B
#             g_b = self.gate_b(a_pool).unsqueeze(1)  # gate for B computed from A
#
#             # apply cross gates to token states (channel-wise scaling)
#             a_fused_tokens = a_ssm * (1.0 + g_a)  # +1 residual style to keep original scale
#             b_fused_tokens = b_ssm * (1.0 + g_b)
#
#             # small symmetric cross-add (stabilized by res_scale)
#             cross_add_a = (b_ssm * g_a) * self.res_scale
#             cross_add_b = (a_ssm * g_b) * self.res_scale
#             a_fused_tokens = a_fused_tokens + cross_add_a
#             b_fused_tokens = b_fused_tokens + cross_add_b
#
#             a_out = a_fused_tokens.transpose(1, 2).view(B, C, D, H, W)
#             b_out = b_fused_tokens.transpose(1, 2).view(B, C, D, H, W)
#             return a_out, b_out
#
#         else:
#             # --- attn mode ---
#             # optionally pool to attn_pool resolution to reduce N (dp,hp,wp)
#             if self.attn_pool is not None:
#                 dp, hp, wp = self.attn_pool
#                 a_small = feat_a
#                 b_small = feat_b
#                 a_small = F.adaptive_avg_pool3d(a_small, (dp, hp, wp))
#                 b_small = F.adaptive_avg_pool3d(b_small, (dp, hp, wp))
#                 Bs, Cs, Ds, Hs, Ws = a_small.shape
#                 Ns = Ds * Hs * Ws
#                 a_tokens = a_small.view(Bs, Cs, Ns).transpose(1, 2)  # (B, Ns, C)
#                 b_tokens = b_small.view(Bs, Cs, Ns).transpose(1, 2)
#                 pooled = True
#             else:
#                 a_tokens = a_ssm
#                 b_tokens = b_ssm
#                 Ns = a_tokens.size(1)
#                 pooled = False
#
#             # qkv & attention
#             Q = self.q_proj(a_tokens)  # (B, M, C')
#             K = self.k_proj(b_tokens)  # (B, M, C')
#             V = self.v_proj(b_tokens)
#
#             attn_logits = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
#             attn = torch.softmax(attn_logits, dim=-1)
#             attn_out = torch.bmm(attn, V)  # (B, M, C')
#             attn_out = self.out_proj(attn_out)  # back to dim C
#
#             # symmetrically from b<-a
#             Qb = self.q_proj(b_tokens)
#             Kb = self.k_proj(a_tokens)
#             Vb = self.v_proj(a_tokens)
#             attn_logits_b = torch.bmm(Qb, Kb.transpose(1, 2)) / (Qb.size(-1) ** 0.5)
#             attn_b = torch.softmax(attn_logits_b, dim=-1)
#             attn_out_b = torch.bmm(attn_b, Vb)
#             attn_out_b = self.out_proj(attn_out_b)
#
#             if pooled:
#                 # upsample attn outputs to full spatial-temporal resolution
#                 dp, hp, wp = self.attn_pool
#                 Cc = attn_out.shape[-1]
#                 attn_small = attn_out.transpose(1, 2).view(B, Cc, dp, hp, wp)
#                 attn_up = F.interpolate(attn_small, size=(D, H, W), mode="trilinear", align_corners=False)
#                 attn_tokens_up = attn_up.view(B, Cc, D * H * W).transpose(1, 2)  # (B, N, C)
#                 a_fused_tokens = a_ssm + attn_tokens_up
#                 attn_small_b = attn_out_b.transpose(1, 2).view(B, Cc, dp, hp, wp)
#                 attn_up_b = F.interpolate(attn_small_b, size=(D, H, W), mode="trilinear", align_corners=False)
#                 attn_tokens_up_b = attn_up_b.view(B, Cc, D * H * W).transpose(1, 2)
#                 b_fused_tokens = b_ssm + attn_tokens_up_b
#             else:
#                 a_fused_tokens = a_ssm + attn_out
#                 b_fused_tokens = b_ssm + attn_out_b
#
#             a_out = a_fused_tokens.transpose(1, 2).view(B, C, D, H, W)
#             b_out = b_fused_tokens.transpose(1, 2).view(B, C, D, H, W)
#             return a_out, b_out

class CrossSSMFusion(nn.Module):
    """
    Cross-SSM Fusion module operating on token/state outputs of two Mamba SSMs.
    mode='gated' : channel-wise cross gating using global pooled features.
    mode='attn'  : token-level cross attention using Mamba SSM token embeddings.
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        nslices: Optional[int] = None,
        reduction: int = 8,
        attn_pool: Optional[Tuple[int,int,int]] = None,
        mode: str = "gated",
    ):
        super().__init__()
        assert mode in ("gated", "attn")
        self.dim = dim
        self.mode = mode
        self.attn_pool = attn_pool

        # 修复 nslices=None 导致 chunk 报错
        nslices = 1 if nslices is None else nslices

        # Two Mamba SSMs (per view)
        self.mamba_a = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=nslices,
        )
        self.mamba_b = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=nslices,
        )

        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

        # --- Gated mode parameters ---
        if self.mode == "gated":
            mid = max(dim // reduction, 8)
            self.gate_a = nn.Sequential(
                nn.Linear(dim, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, dim),
                nn.Sigmoid(),
            )
            self.gate_b = nn.Sequential(
                nn.Linear(dim, mid),
                nn.ReLU(inplace=True),
                nn.Linear(mid, dim),
                nn.Sigmoid(),
            )

        # --- Attention mode parameters ---
        if self.mode == "attn":
            proj_dim = max(dim // reduction, 8)
            self.q_proj = nn.Linear(dim, proj_dim)
            self.k_proj = nn.Linear(dim, proj_dim)
            self.v_proj = nn.Linear(dim, proj_dim)
            self.out_proj = nn.Linear(proj_dim, dim)

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """
        feat_a, feat_b: [B, C, D, H, W]
        returns: fused_a, fused_b
        """

        B, C, D, H, W = feat_a.shape
        N = D * H * W
        a_flat = feat_a.view(B, C, N).transpose(1, 2)
        b_flat = feat_b.view(B, C, N).transpose(1, 2)

        a_norm = self.norm_a(a_flat)
        b_norm = self.norm_b(b_flat)

        a_ssm = self.mamba_a(a_norm)
        b_ssm = self.mamba_b(b_norm)

        if self.mode == "gated":
            a_pool = a_ssm.mean(dim=1)
            b_pool = b_ssm.mean(dim=1)
            g_a = self.gate_a(b_pool).unsqueeze(1)
            g_b = self.gate_b(a_pool).unsqueeze(1)

            a_fused = a_ssm * (1 + g_a) + (b_ssm * g_a) * self.res_scale
            b_fused = b_ssm * (1 + g_b) + (a_ssm * g_b) * self.res_scale

        else:  # --- attn ---
            Q = self.q_proj(a_ssm)
            K = self.k_proj(b_ssm)
            V = self.v_proj(b_ssm)
            attn = torch.softmax(Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5), dim=-1)
            attn_out_a = self.out_proj(attn @ V)

            Qb = self.q_proj(b_ssm)
            Kb = self.k_proj(a_ssm)
            Vb = self.v_proj(a_ssm)
            attn_b = torch.softmax(Qb @ Kb.transpose(-2, -1) / (Qb.size(-1) ** 0.5), dim=-1)
            attn_out_b = self.out_proj(attn_b @ Vb)

            a_fused = a_ssm + attn_out_a
            b_fused = b_ssm + attn_out_b

        a_out = a_fused.transpose(1, 2).view(B, C, D, H, W)
        b_out = b_fused.transpose(1, 2).view(B, C, D, H, W)
        return a_out, b_out


class DualMIMamba_CrossSSM(nn.Module):
    """
    Dual-branch Mamba with Cross-SSM fusion.
    Inputs:
      x_a2c, x_a4c: [B, C, D, H, W]
    Outputs:
      logit/prob (after sigmoid)
    """

    def __init__(
            self,
            in_chans=1,
            num_classes=1,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            fusion_kwargs: dict = None,
            hidden_size: int = 768,
            norm_name="instance",
            res_block: bool = True,
            spatial_dims=3,
            dropout_rate: float = 0.5,
    ):
        super().__init__()
        # self.drop_path_rate = drop_path_rate
        self.fusion_weights = nn.Parameter(torch.ones(2))
        if fusion_kwargs is None:
            fusion_kwargs = {}

        self.encoder_a2c = MambaEncoder(in_chans, depths, feat_size, drop_path_rate, layer_scale_init_value)
        self.encoder_a4c = MambaEncoder(in_chans, depths, feat_size, drop_path_rate, layer_scale_init_value)

        # 最后一层特征做高层融合
        final_dim = feat_size[-1]

        # head to map fused features to hidden_size (reuse UnetrBasicBlock)
        # encoder head (convert to hidden_size)
        self.encoder_head = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=final_dim,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # --- Stage 1: Gated fusion ---
        gated_kwargs = dict(fusion_kwargs)
        gated_kwargs["mode"] = "gated"
        self.fusion_gated = CrossSSMFusion(dim=hidden_size, **gated_kwargs)

        # --- Stage 2: Attention fusion ---
        attn_kwargs = dict(fusion_kwargs)
        attn_kwargs["mode"] = "attn"
        self.fusion_attn = CrossSSMFusion(dim=hidden_size, **attn_kwargs)

        # classifier (binary / regression)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_size, 512),

            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),

            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes),
        )
        # self.output_activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x_a2c: torch.Tensor, x_a4c: torch.Tensor, return_features: bool = False):
        # feats_a = self.encoder_a2c(x_a2c)
        feats_b = self.encoder_a4c(x_a4c)

        # take last-level features
        # feat_a = feats_a[-1]
        feat_b = feats_b[-1]

        # project through encoder heads
        # head_a = self.encoder_head(feat_a)
        head_b = self.encoder_head(feat_b)

        # # --- Stage 1: Gated fusion ---
        # a_gate, b_gate = self.fusion_gated(head_a, head_b)
        #
        # --- Stage 2: Attention fusion ---
        # a_attn, b_attn = self.fusion_attn(head_a, head_b)

        # 注意力权重进行加权融合
        # weights = torch.softmax(self.fusion_weights, dim=0)
        # fused_features = weights[0] * head_a + weights[1] * head_b
        fused_features = head_b

        logit = self.classifier(fused_features)
        # out = self.output_activation(logit)

        if return_features:
            return logit, fused_features
        return logit


if __name__ == "__main__":
    B = 2
    C = 1
    D, H, W = 16, 224, 224
    model = DualMIMamba_CrossSSM(
        in_chans=C,
        num_classes=1,
        depths=[1, 1, 1, 1],
        feat_size=[16, 32, 64, 128],
        fusion_kwargs=dict(d_state=8, reduction=8, mode="gated"),
        hidden_size=128,
    )
    model = model.cuda()
    model.eval()

    x1 = torch.randn(B, C, D, H, W).cuda()
    x2 = torch.randn(B, C, D, H, W).cuda()

    with torch.no_grad():
        y = model(x1, x2)
    print("DualMIMamba_CrossSSM output:", y.shape)  # expect [B, 1]
    print("DualMIMamba_CrossSSM output.pred:", y)
