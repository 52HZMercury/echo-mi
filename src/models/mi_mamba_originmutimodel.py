from __future__ import annotations

import math
import torch.nn as nn
import torch, einops
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F

# 确保 EchoPrimeTextEncoder 被导入 (保持引用路径不变)
from .components.echoprime_encoders import EchoPrimeVideoEncoder, EchoPrimeTextEncoder
from src.utils.prompts import COMMON_KNOWLEDGE, A2C_SPECIFIC_KNOWLEDGE, A4C_SPECIFIC_KNOWLEDGE


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


# ==============================================================================
# 核心改造 1: 支持文本引导的 Mamba Layer
# ==============================================================================
class TextGuidedMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.num_slices = num_slices  # 记录切片数
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )

    def forward(self, x, text_tokens=None):
        B, C = x.shape[:2]
        x_skip = x
        img_dims = x.shape[2:]
        n_visual_tokens = x.shape[2:].numel()

        # 1. 展平图像特征
        x_flat = x.reshape(B, C, n_visual_tokens).transpose(-1, -2)  # (B, L_vis, C)

        # 2. 拼接文本
        if text_tokens is not None:
            combined_seq = torch.cat([text_tokens, x_flat], dim=1)  # (B, L_total, C)
        else:
            combined_seq = x_flat

        original_length = combined_seq.shape[1]

        # --- 核心修复：填充序列以适配 nslices ---
        if self.num_slices is not None:
            pad_len = (self.num_slices - (original_length % self.num_slices)) % self.num_slices
            if pad_len > 0:
                # 在序列末尾填充 0
                combined_seq = F.pad(combined_seq, (0, 0, 0, pad_len))  # (B, L_padded, C)
        # ---------------------------------------

        # 3. Norm & Mamba
        x_norm = self.norm(combined_seq)
        x_mamba_out = self.mamba(x_norm)

        # 4. 剔除填充部分并分离文本
        if self.num_slices is not None and pad_len > 0:
            x_mamba_out = x_mamba_out[:, :original_length, :]

        if text_tokens is not None:
            n_text = text_tokens.shape[1]
            x_visual_out = x_mamba_out[:, n_text:, :]
        else:
            x_visual_out = x_mamba_out

        # 5. 还原形状
        out = x_visual_out.transpose(-1, -2).reshape(B, C, *img_dims)
        return out + x_skip


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


# ==============================================================================
# 核心改造 2: 能够处理文本投影的 Encoder
# ==============================================================================
class TextGuidedMambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 text_embed_dim=512):  # 新增 text_embed_dim
        super().__init__()

        self.downsample_layers = nn.ModuleList()
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

        # 文本投影层：将 512 维的文本向量映射到各层的通道维度 [48, 96, 192, 384]
        self.text_projectors = nn.ModuleList()

        num_slices_list = [32, 16, 8, 4]

        for i in range(4):
            # 1. 文本投影
            self.text_projectors.append(nn.Linear(text_embed_dim, dims[i]))

            # 2. Mamba Stage (使用 TextGuidedMambaLayer)
            gsc = GSC(dims[i])
            stage = nn.Sequential(
                *[TextGuidedMambaLayer(dim=dims[i], num_slices=num_slices_list[i])
                  for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x, text_embedding=None):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)

            current_text_tokens = None
            if text_embedding is not None:
                current_text_tokens = self.text_projectors[i](text_embedding)

            # 修改这里：显式遍历 Sequential 中的每一层
            stage_module = self.stages[i]
            for layer in stage_module:
                # 现在的 layer 是 TextGuidedMambaLayer，可以接收 text_tokens
                x = layer(x, current_text_tokens)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x, text_embedding=None):
        x = self.forward_features(x, text_embedding)
        return x


class ClsMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.BatchNorm1d(in_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_dim * 2, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# ==============================================================================
# 改造后的主模型
# ==============================================================================

class MIMambaOriginMutiModel(nn.Module):
    def __init__(
            self,
            in_chans=3,
            out_chans=1,
            depths=[2, 2, 2, 2],
            feat_size=[16, 32, 64, 128],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 256,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
            embed_dim=512,  # EchoPrime 编码器的输出维度
            video_encoder_path=None,
            text_encoder_path=None,
            frozen_video_encoder=True,
            frozen_text_encoder=True,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.embed_dim = embed_dim

        self.spatial_dims = spatial_dims

        # --- 1. 文本引导的 Mamba Encoder ---
        self.mamba_encoder = TextGuidedMambaEncoder(
            in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            text_embed_dim=embed_dim  # 传入文本维度
        )

        # --- UNet Decoder 部分 (主要用于 A4C 分割流) ---
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        # ------------------- 知识获取模块 -------------------

        # 文本编码器
        self.text_encoder = EchoPrimeTextEncoder(
            pretrained_path=text_encoder_path,
            frozen=frozen_text_encoder
        )

        # 视频编码器 (EchoPrime, 辅助用于筛选知识，或者直接把所有知识送入 Mamba 让 Mamba 筛选)
        # 根据需求：让 Mamba 决定使用哪些文本。
        # 因此我们将所有相关知识提取出来，拼接送入 Mamba。
        self.video_encoder = EchoPrimeVideoEncoder(
            pretrained_path=video_encoder_path,
            frozen=frozen_video_encoder
        )

        # 知识库
        self.prompts_all = COMMON_KNOWLEDGE + A2C_SPECIFIC_KNOWLEDGE + A4C_SPECIFIC_KNOWLEDGE
        self.num_knowledge_prompts = len(self.prompts_all)

        # ------------------- 分类头 -------------------
        # 我们将 A2C 和 A4C 经过 Mamba 提取后的特征拼接
        # 特征维度: dec1_flat (来自A4C Mamba路径) + Mamba_A2C_flat
        # dec1 大小为 feat_size[1] (32), pooling后为 32*4*4*4 = 2048

        mamba_feat_dim = self.feat_size[1] * (4 * 4 * 4)  # 2048

        self.classification_head = ClsMLP(
            in_dim=mamba_feat_dim * 2,  # A2C + A4C
            out_dim=1
        )

    def forward(self, x_a2c: torch.Tensor, x_a4c: torch.Tensor, return_features: bool = False):
        bs = x_a2c.shape[0]

        # 1. 准备文本知识向量
        # 提取所有知识的 embedding: [B, N_prompts, 512]
        with torch.no_grad():
            text_features_all = self.text_encoder(self.prompts_all).unsqueeze(0).expand(bs, -1, -1)

        # 2. 特征提取 - A4C Stream 分类
        # 将文本和 A4C 视频一起送入 Mamba
        # Mamba 内部: [Text, A4C_Visual] -> SSM -> Updated Visual
        outs_a4c = self.mamba_encoder(x_a4c, text_embedding=text_features_all)

        # 3. 特征提取 - A2C Stream (主要用于分类)
        # 同样使用 Mamba 处理 A2C，实现"A2C和A4C都使用mamba"
        outs_a2c = self.mamba_encoder(x_a2c, text_embedding=text_features_all)

        # --- A4C 路径  ---
        x2_a4c = outs_a4c[0]
        enc2_a4c = self.encoder2(x2_a4c)
        x3_a4c = outs_a4c[1]
        enc3_a4c = self.encoder3(x3_a4c)
        x4_a4c = outs_a4c[2]
        enc4_a4c = self.encoder4(x4_a4c)
        enc_hidden_a4c = self.encoder5(outs_a4c[3])

        # Decoder (A4C)
        dec3_a4c = self.decoder5(enc_hidden_a4c, enc4_a4c)
        dec2_a4c = self.decoder4(dec3_a4c, enc3_a4c)
        dec1_a4c = self.decoder3(dec2_a4c, enc2_a4c)  # (B, 32, D/2, H/2, W/2)

        # --- A2C 特征处理 ---
        x2_a2c = outs_a2c[0]
        enc2_a2c = self.encoder2(x2_a2c)
        x3_a2c = outs_a2c[1]
        enc3_a2c = self.encoder3(x3_a2c)
        x4_a2c = outs_a2c[2]
        enc4_a2c = self.encoder4(x4_a2c)
        enc_hidden_a2c = self.encoder5(outs_a2c[3])

        # Decoder (A2C)
        dec3_a2c = self.decoder5(enc_hidden_a2c, enc4_a2c)
        dec2_a2c = self.decoder4(dec3_a2c, enc3_a2c)
        dec1_a2c = self.decoder3(dec2_a2c, enc2_a2c)  # (B, 32, D/2, H/2, W/2)

        # --- 池化与分类 ---
        # A4C 特征 (来自 Decoder 输出)
        dec1_pooled_a4c = F.adaptive_avg_pool3d(dec1_a4c, (4, 4, 4))
        flat_a4c = dec1_pooled_a4c.view(bs, -1)  # 2048

        # A2C 特征 (来自 Mamba Encoder 中间层)
        # 使用 A2C 对应的层级特征
        feat_a2c_pooled = F.adaptive_avg_pool3d(dec1_a2c, (4, 4, 4))
        flat_a2c = feat_a2c_pooled.view(bs, -1)  # 2048

        # 融合
        fused_feat = torch.cat([flat_a4c, flat_a2c], dim=1)  # [B, 4096]

        # 分类
        logits = self.classification_head(fused_feat)

        if return_features:
            return logits, fused_feat

        return logits