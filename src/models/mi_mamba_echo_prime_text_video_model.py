from __future__ import annotations

import math
import torch.nn as nn
import torch, einops
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F

# 确保 EchoPrimeTextEncoder 被导入
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

        # 后三维展平为一维进行扫描
        # 横向
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        # 恢复为原来的形状
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        out = out + x_skip

        return out


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
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
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
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i])
                  for j in
                  range(depths[i])]
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


# 假设 ClsMLP 是一个三层 MLP
class ClsMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 三层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),

            nn.BatchNorm1d(in_dim * 2),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加Dropout

            nn.Linear(in_dim * 2, in_dim),
            nn.BatchNorm1d(in_dim),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加Dropout

            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class KnowledgeFusion(nn.Module):
    """将知识向量注入到 decoder 的中间特征层"""

    def __init__(self, feat_dim, knowledge_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(knowledge_dim, feat_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(knowledge_dim, feat_dim)

    def forward(self, feat, knowledge_vector):
        """
        feat: (B, C, D, H, W)
        knowledge_vector: (B, Dk)
        """
        B, C, D, H, W = feat.shape
        k_proj = self.proj(knowledge_vector).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1, 1]
        gate = 0.01
        fused = feat + gate * k_proj
        return fused


# ==============================================================================
# 改造后的 Mamba 模型
# ==============================================================================

class MIMambaEchoPrimeTextVideo(nn.Module):
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

            # --- 新增编码器相关参数 ---
            embed_dim=512,  # EchoPrime 编码器的输出维度
            video_encoder_path=None,  # 视频预训练权重路径
            text_encoder_path=None,  # 文本预训练权重路径
            frozen_video_encoder=True,
            frozen_text_encoder=True,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.embed_dim = embed_dim  # 视频/文本特征维度

        self.spatial_dims = spatial_dims
        self.mamba_encoder = MambaEncoder(in_chans,
                                          depths=depths,
                                          dims=feat_size,
                                          drop_path_rate=drop_path_rate,
                                          layer_scale_init_value=layer_scale_init_value,
                                          )

        # --- UNet/SegMamba 骨架块初始化 (省略部分参数，与原代码一致) ---
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
        # self.decoder2 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[1],
        #     out_channels=self.feat_size[0],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder1 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[0],
        #     out_channels=self.feat_size[0],
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )

        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=out_chans)

        # --- 知识融合层 ---
        self.knowledge_fusions = nn.ModuleList([
            KnowledgeFusion(self.feat_size[3], embed_dim),  # 128
            KnowledgeFusion(self.feat_size[2], embed_dim),  # 64
            KnowledgeFusion(self.feat_size[1], embed_dim),  # 32
            # KnowledgeFusion(self.feat_size[0], embed_dim),  # 16
        ])

        # ------------------- 改造：新增视频、文本与分类模块 -------------------

        # 1. 视频编码器 (使用提供的 EchoPrimeVideoEncoder)
        self.video_encoder = EchoPrimeVideoEncoder(
            pretrained_path=video_encoder_path,
            frozen=frozen_video_encoder
        )

        # 2. 文本编码器 (新增)
        self.text_encoder = EchoPrimeTextEncoder(
            pretrained_path=text_encoder_path,
            frozen=frozen_text_encoder
        )

        # 3. 加载知识 (新增)  通用知识
        self.prompts_all = COMMON_KNOWLEDGE + A2C_SPECIFIC_KNOWLEDGE + A4C_SPECIFIC_KNOWLEDGE
        self.num_knowledge_prompts = len(self.prompts_all)
        self.positional_embedding_knowledge = nn.Parameter(torch.randn(1, self.num_knowledge_prompts, embed_dim))

        # 4. 通道对齐层
        dec_feat_channel = self.feat_size[1] * (4 * 4 * 4)

        # 5. 分类 MLP (修改)
        # 输入维度:
        # 1. dec1_flat (2048)
        # 2. video_feat_a2c (512)
        self.classification_head = ClsMLP(
            in_dim=dec_feat_channel + embed_dim,
            out_dim=1
        )

    def forward(self, x_a2c: torch.Tensor, x_a4c: torch.Tensor, return_features: bool = False):
        # --------------------- echo_prime 视频提取及知识激活路径 ---------------------
        bs = x_a2c.shape[0]  # 获取批次大小

        # 视频特征提取 - 用于知识激活
        video_feat_a2c = self.video_encoder(x_a2c)
        video_feat_a4c = self.video_encoder(x_a4c)
        # a. 将所有知识提示编码为特征向量。
        with torch.no_grad():  # 激活过程不参与梯度计算
            text_features_all = self.text_encoder(self.prompts_all).expand(bs, -1, -1)  # [B, N_k, D]

        # b. 逐一计算视觉特征与知识特征的余弦相似度。
        sim_a2c = F.cosine_similarity(video_feat_a2c.unsqueeze(1), text_features_all, dim=-1)  # -> [B, N]
        sim_a4c = F.cosine_similarity(video_feat_a4c.unsqueeze(1), text_features_all, dim=-1)  # -> [B, N]

        # c. 为每条知识生成最终的激活权重
        activation_weights = torch.max(sim_a2c, sim_a4c).clamp(min=0)  # -> [B, N]

        # d. 构建激活的知识序列
        knowledge_sequence = text_features_all * activation_weights.unsqueeze(-1)  # [B, N_k, D]
        knowledge_sequence += self.positional_embedding_knowledge

        # e. 池化知识，得到用于分类的 "知识向量"
        knowledge_vector = knowledge_sequence.mean(dim=1)  # [B, D]

        a2c_fused_features = video_feat_a2c + knowledge_vector * 0.01

        # --------------------- Mamba 路径 (使用 x_a4c) ---------------------
        outs = self.mamba_encoder(x_a4c)  # Mamba Encoder features (x2, x3, x4, x_hidden)
        # enc1 = self.encoder1(x_a4c)  # (B, 16, D, H, W)
        x2 = outs[0]
        enc2 = self.encoder2(x2)  # (B, 32, D/2, H/2, W/2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)  # (B, 64, D/4, H/4, W/4)
        x4 = outs[2]
        enc4 = self.encoder4(x4)  # (B, 128, D/8, H/8, W/8)
        enc_hidden = self.encoder5(outs[3])  # (B, 256, D/16, H/16, W/16)

        # --- Decoder + 知识融合 ---
        dec3 = self.decoder5(enc_hidden, enc4)  # (B, 128, D/8, H/8, W/8)
        dec3 = self.knowledge_fusions[0](dec3, knowledge_vector)

        dec2 = self.decoder4(dec3, enc3)  # (B, 64, D/4, H/4, W/4)
        dec2 = self.knowledge_fusions[1](dec2, knowledge_vector)

        dec1 = self.decoder3(dec2, enc2)  # (B, 32, D/2, H/2, W/2)
        dec1 = self.knowledge_fusions[2](dec1, knowledge_vector)

        # 提取 Mamba 路径的特征用于分类
        # dec1: (B, 32, D/2, H/2, W/2) -> (B, 32*4*4*4) = (B, 2048)
        dec1_pooled = F.adaptive_avg_pool3d(dec1, (4, 4, 4))
        dec1_flat = dec1_pooled.view(dec1.size(0), -1)

        # --------------------- 融合与分类 ---------------------
        # 1. 特征融合（通道拼接）
        # [B, 2048] + [B, 512] -> [B, 3072]
        fused_feat = torch.cat([dec1_flat, a2c_fused_features], dim=1)

        # 2. 送入 MLP 进行二分类
        # cls_out: (B, 1)
        logits = self.classification_head(fused_feat)

        # 返回分割输出和分类输出
        if return_features:
            return logits, fused_feat

        return logits
