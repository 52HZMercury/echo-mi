from __future__ import annotations
import torch.nn as nn
import torch
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F


# from .components.adapters import ViewAdapter # 未使用，移除
# from .components.attention import AttnFusion # 未使用，移除

# --- 复制您提供的基础模块 ---

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


# --- MambaEncoder (从您提供的文件中复制) ---
# 我们不会直接实例化MambaEncoder，但我们会复用它的架构逻辑
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
        num_slices_list = [64, 32, 16, 8]
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
            x_stage = self.stages[i](x)  # Mamba stage output

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_stage)  # Norm
                x_out = self.mlps[i](x_out)  # MLP
                outs.append(x_out)

            x = x_stage  # Pass Mamba output to next downsample layer

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


# --- 新增的融合模块 ---

class DynamicGatingModule(nn.Module):
    """
    动态门控加权融合模块。
    它学习一个“门”，该“门”决定如何加权融合两个特征图。
    """

    def __init__(self, dim):
        super().__init__()
        # 门控卷积，输入为两个特征拼接，输出为单个门控图
        self.gate_conv = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # 权重初始化
        nn.init.constant_(self.gate_conv[0].bias, 0.5)

    def forward(self, x1, x2):
        # 拼接两个输入
        x_cat = torch.cat([x1, x2], dim=1)
        # 计算动态门控权重（gate值接近1，则x1权重高；接近0，则x2权重高）
        gate = self.gate_conv(x_cat)
        # 应用门控：gate * x1 + (1 - gate) * x2
        return (gate * x1) + ((1.0 - gate) * x2)


class CrossAttentionModule(nn.Module):
    """
    交叉注意力融合模块。
    允许A2C特征作为Query，查询A4C特征（Key/Value），反之亦然。
    这有助于模型捕捉两个视图之间的协同运动信息。
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # A2C作为Query，A4C作为Key/Value
        self.q_proj_2c = nn.Linear(dim, dim)
        self.k_proj_4c = nn.Linear(dim, dim)
        self.v_proj_4c = nn.Linear(dim, dim)
        self.attn_2c_to_4c = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_2c = nn.LayerNorm(dim)

        # A4C作为Query，A2C作为Key/Value
        self.q_proj_4c = nn.Linear(dim, dim)
        self.k_proj_2c = nn.Linear(dim, dim)
        self.v_proj_2c = nn.Linear(dim, dim)
        self.attn_4c_to_2c = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_4c = nn.LayerNorm(dim)

        # 输出融合
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x_2c, x_4c):
        # 输入形状: (B, C, D, H, W)
        B, C, D, H, W = x_2c.shape
        # 展平为序列: (B, N, C) where N = D*H*W
        x_2c_flat = x_2c.flatten(2).transpose(1, 2)
        x_4c_flat = x_4c.flatten(2).transpose(1, 2)

        # 1. A2C attends to A4C
        q_2c = self.q_proj_2c(x_2c_flat)
        k_4c = self.k_proj_4c(x_4c_flat)
        v_4c = self.v_proj_4c(x_4c_flat)
        attn_out_2c, _ = self.attn_2c_to_4c(q_2c, k_4c, v_4c)
        fused_2c = self.norm_2c(x_2c_flat + attn_out_2c)  # 残差连接 + 归一化

        # 2. A4C attends to A2C
        q_4c = self.q_proj_4c(x_4c_flat)
        k_2c = self.k_proj_2c(x_2c_flat)
        v_2c = self.v_proj_2c(x_2c_flat)
        attn_out_4c, _ = self.attn_4c_to_2c(q_4c, k_2c, v_2c)
        fused_4c = self.norm_4c(x_4c_flat + attn_out_4c)  # 残差连接 + 归一化

        # 3. 最终融合 (简单相加)
        final_fused = self.out_norm(fused_2c + fused_4c)

        # 恢复形状: (B, C, D, H, W)
        return final_fused.transpose(1, 2).reshape(B, C, D, H, W)


# --- 确保所有基础模块 (LayerNorm, MambaLayer, GSC, MlpChannel,
# --- DynamicGatingModule, CrossAttentionModule) 已被定义 ---
# (这些代码与上一条回答中相同，此处不再重复)


class MIMambaHierarchical(nn.Module):
    def __init__(
            self,
            in_chans=1,  # 单个视图的输入通道数
            num_classes=2,  # 二分类
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],  # 编码器各阶段的维度
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,  # 最终特征维度
            norm_name="instance",
            res_block: bool = True,
            spatial_dims=3,
            dropout_rate=0.5
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depths = depths
        self.feat_size = feat_size

        # ---------------------------------
        # 1. 独立编码器路径 (A2C 和 A4C)
        # ---------------------------------
        self.fused_in_chans = in_chans * 2  # 早期融合：通道拼接

        # --- A2C 路径组件 ---
        self.downsample_layers_2c = nn.ModuleList()
        self.stages_2c = nn.ModuleList()
        self.gscs_2c = nn.ModuleList()
        self.mlps_2c = nn.ModuleList()

        # --- A4C 路径组件 ---
        self.downsample_layers_4c = nn.ModuleList()
        self.stages_4c = nn.ModuleList()
        self.gscs_4c = nn.ModuleList()
        self.mlps_4c = nn.ModuleList()

        num_slices_list = [64, 32, 16, 8]  # Mamba切片数

        # ==================== [核心修改点] ====================
        # 为2C和4C路径定义不同的Mamba超参数

        # origin {"d_state": 16, "d_conv": 4, "expand": 2}
        # 2C路径: 聚焦局部运动 (更小的 d_state)
        self.mamba_params_2c = {"d_state": 16, "d_conv": 4, "expand": 2}

        # 4C路径: 聚焦全局协调 (更大的 d_state)
        self.mamba_params_4c = {"d_state": 16, "d_conv": 4, "expand": 2}

        # ======================================================

        # 构建Stem（第一个下采样层）
        stem_2c = nn.Sequential(
            nn.Conv3d(self.fused_in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers_2c.append(stem_2c)

        stem_4c = nn.Sequential(
            nn.Conv3d(self.fused_in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers_4c.append(stem_4c)

        # 构建后续的3个下采样层和4个Mamba阶段
        for i in range(4):
            # A2C
            gsc_2c = GSC(feat_size[i])
            # [修改] 应用2C的专用参数
            stage_2c = nn.Sequential(
                *[MambaLayer(
                    dim=feat_size[i],
                    num_slices=num_slices_list[i],
                    **self.mamba_params_2c
                ) for j in range(depths[i])]
            )
            norm_2c = nn.InstanceNorm3d(feat_size[i])
            mlp_2c = MlpChannel(feat_size[i], 2 * feat_size[i])

            self.gscs_2c.append(gsc_2c)
            self.stages_2c.append(stage_2c)
            self.add_module(f'norm{i}_2c', norm_2c)  # 注册norm层
            self.mlps_2c.append(mlp_2c)

            # A4C
            gsc_4c = GSC(feat_size[i])
            # [修改] 应用4C的专用参数
            stage_4c = nn.Sequential(
                *[MambaLayer(
                    dim=feat_size[i],
                    num_slices=num_slices_list[i],
                    **self.mamba_params_4c
                ) for j in range(depths[i])]
            )
            norm_4c = nn.InstanceNorm3d(feat_size[i])
            mlp_4c = MlpChannel(feat_size[i], 2 * feat_size[i])

            self.gscs_4c.append(gsc_4c)
            self.stages_4c.append(stage_4c)
            self.add_module(f'norm{i}_4c', norm_4c)  # 注册norm层
            self.mlps_4c.append(mlp_4c)

            # 添加下采样层 (除了最后一个阶段)
            if i < 3:
                ds_layer_2c = nn.Sequential(
                    nn.InstanceNorm3d(feat_size[i]),
                    nn.Conv3d(feat_size[i], feat_size[i + 1], kernel_size=2, stride=2),
                )
                self.downsample_layers_2c.append(ds_layer_2c)

                ds_layer_4c = nn.Sequential(
                    nn.InstanceNorm3d(feat_size[i]),
                    nn.Conv3d(feat_size[i], feat_size[i + 1], kernel_size=2, stride=2),
                )
                self.downsample_layers_4c.append(ds_layer_4c)

        # ---------------------------------
        # 2. 中期融合模块 (动态门控)
        # ---------------------------------
        self.mid_fusion_1 = DynamicGatingModule(feat_size[1])  # 融合 96 维特征
        self.mid_fusion_2 = DynamicGatingModule(feat_size[2])  # 融合 192 维特征

        # ---------------------------------
        # 3. 晚期融合模块 (交叉注意力)
        # ---------------------------------
        self.final_enc_2c = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.final_enc_4c = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.late_fusion = CrossAttentionModule(dim=self.hidden_size, num_heads=8)

        # ---------------------------------
        # 4. 分类器
        # ---------------------------------
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(self.hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 1)  # 输出1个logit，用于二分类
        )

    def forward(self, a2c_video, a4c_video, return_features=False):
        # 1. 早期融合 (通道拼接)
        x_fused_early = torch.cat([a2c_video, a4c_video], dim=1)

        x_2c = x_fused_early
        x_4c = x_fused_early

        outs_2c = []
        outs_4c = []

        # 2. 独立编码器
        for i in range(4):  # 4个阶段
            # (a) 下采样
            x_2c = self.downsample_layers_2c[i](x_2c)
            x_4c = self.downsample_layers_4c[i](x_4c)

            # (b) GSC + Mamba Stage
            x_2c = self.gscs_2c[i](x_2c)
            x_4c = self.gscs_4c[i](x_4c)

            x_stage_2c = self.stages_2c[i](x_2c)  # 使用专用的Mamba 2C
            x_stage_4c = self.stages_4c[i](x_4c)  # 使用专用的Mamba 4C

            # (c) 特征后处理 (Norm + MLP)
            norm_layer_2c = getattr(self, f'norm{i}_2c')
            norm_layer_4c = getattr(self, f'norm{i}_4c')

            x_out_2c = self.mlps_2c[i](norm_layer_2c(x_stage_2c))
            x_out_4c = self.mlps_4c[i](norm_layer_4c(x_stage_4c))

            outs_2c.append(x_out_2c)
            outs_4c.append(x_out_4c)

            # (d) 中期融合 (动态门控)
            x_2c = x_stage_2c
            x_4c = x_stage_4c

            if i == 1:
                fused_mid_1 = self.mid_fusion_1(x_out_2c, x_out_4c)
                x_2c = x_2c + fused_mid_1
                x_4c = x_4c + fused_mid_1

            elif i == 2:
                fused_mid_2 = self.mid_fusion_2(x_out_2c, x_out_4c)
                x_2c = x_2c + fused_mid_2
                x_4c = x_4c + fused_mid_2

        # 3. 晚期融合 (交叉注意力)
        final_feat_2c = outs_2c[3]
        final_feat_4c = outs_4c[3]

        enc_2c = self.final_enc_2c(final_feat_2c)
        enc_4c = self.final_enc_4c(final_feat_4c)

        fused_late = self.late_fusion(enc_2c, enc_4c)

        # 4. 分类器
        logits = self.classifier(fused_late)

        if return_features:
            return logits, fused_late
        return logits


# --- 使用示例 ---
if __name__ == "__main__":
    # 配置模型参数
    model = MIMambaHierarchical(
        in_chans=3,  # 假设输入是单通道 (e.g., 灰度视频)
        num_classes=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],  # 必须与MambaEncoder的默认值匹配
        hidden_size=768,  # 最终特征维度
        drop_path_rate=0.1
    )

    model = model.cuda()
    model.eval()  # 切换到评估模式

    # 测试输入 (B, C, D, H, W)
    # 假设 batch_size=2, 1通道, 32帧, 224x224 分辨率
    a2c_input = torch.randn(16, 3, 32, 224, 224).cuda()
    a4c_input = torch.randn(16, 3, 32, 224, 224).cuda()

    # 前向传播
    with torch.no_grad():
        output = model(a2c_input, a4c_input)

    print(f"Model created successfully.")
    print(f"A2C Input shape: {a2c_input.shape}")
    print(f"A4C Input shape: {a4c_input.shape}")
    print(f"Output (logits) shape: {output.shape}")  # 应该是 torch.Size([2, 1])
    print(f"Output (logits): {output}")
    print(f"Output (logits): {torch.sigmoid(output)}")