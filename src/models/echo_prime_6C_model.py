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


# ==============================================================================
# 改造后的 Mamba 模型
# ==============================================================================

class EchoPrime6CVideo(nn.Module):
    def __init__(
            self,
            in_chans=3,
            # --- 新增编码器相关参数 ---
            embed_dim=512,  # EchoPrime 编码器的输出维度
            video_encoder_path=None,  # 视频预训练权重路径
            frozen_video_encoder=True,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim  # 视频/文本特征维度

        # ------------------- 改造：新增视频、文本与分类模块 -------------------
        # 1. 视频编码器 (使用提供的 EchoPrimeVideoEncoder)
        self.video_encoder = EchoPrimeVideoEncoder(
            pretrained_path=video_encoder_path,
            frozen=frozen_video_encoder
        )

        self.classification_head = ClsMLP(
            in_dim=embed_dim * 6,
            out_dim=1
        )

    def forward(self, x_a2c: torch.Tensor, x_a3c: torch.Tensor, x_a4c: torch.Tensor, x_apsax: torch.Tensor, x_mvsax: torch.Tensor, x_pmsax: torch.Tensor, return_features: bool = False):
        # --------------------- echo_prime 视频提取及知识激活路径 ---------------------
        # 视频特征提取 - 用于知识激活
        video_feat_a2c = self.video_encoder(x_a2c)
        video_feat_a3c = self.video_encoder(x_a3c)
        video_feat_a4c = self.video_encoder(x_a4c)
        video_feat_apsax = self.video_encoder(x_apsax)
        video_feat_mvsax = self.video_encoder(x_mvsax)
        video_feat_pmsax = self.video_encoder(x_pmsax)


        # --------------------- 融合与分类 ---------------------
        # 1. 特征融合（通道拼接）
        # [B, 2048] + [B, 512] -> [B, 3072]
        fused_feat = torch.cat([video_feat_a2c, video_feat_a3c, video_feat_a4c, video_feat_apsax, video_feat_mvsax, video_feat_pmsax], dim=1)

        # 2. 送入 MLP 进行二分类
        # cls_out: (B, 1)
        logits = self.classification_head(fused_feat)

        # 返回分割输出和分类输出
        if return_features:
            return logits, fused_feat

        return logits
