from __future__ import annotations

from __future__ import annotations

import math
import torch.nn as nn
import torch, einops
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F

from .components.echoprime_encoders import  EchoPrimeVideoEncoder, EchoPrimeTextEncoder
# from components.echoprime_encoders import  EchoPrimeVideoEncoder, EchoPrimeTextEncoder

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



# ==============================================================================
# 改造后的 SegMamba 模型
# ==============================================================================

class MIMamba6C3(nn.Module):
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

    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.hidden_size = hidden_size
        self.feat_size = feat_size

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans,
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                                )

        # --- UNet/SegMamba 骨架块初始化 (省略部分参数，与原代码一致) ---
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

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=out_chans)


        # 2. 通道对齐层
        dec_feat_channel = self.feat_size[1] * 64 * 6

        # 3. 分类 MLP
        # 输入维度: dec1_channels
        self.classification_head = ClsMLP(
            in_dim=dec_feat_channel,
            out_dim=3
        )


    def forward(self, x_a2c: torch.Tensor, x_a3c: torch.Tensor, x_a4c: torch.Tensor, x_apsax: torch.Tensor, x_mvsax: torch.Tensor, x_pmsax: torch.Tensor, return_features: bool = False):

        # --------------------- Mamba 路径 ---------------------
        # 处理 A2C 切面
        outs_a2c = self.vit(x_a2c)
        enc2_a2c = self.encoder2(outs_a2c[0])
        enc3_a2c = self.encoder3(outs_a2c[1])
        enc4_a2c = self.encoder4(outs_a2c[2])
        enc_hidden_a2c = self.encoder5(outs_a2c[3])
        dec3_a2c = self.decoder5(enc_hidden_a2c, enc4_a2c)
        dec2_a2c = self.decoder4(dec3_a2c, enc3_a2c)
        dec1_a2c = self.decoder3(dec2_a2c, enc2_a2c)  # (B, 32, D/2, H/2, W/2)

        # 处理 A3C 切面
        outs_a3c = self.vit(x_a3c)
        enc2_a3c = self.encoder2(outs_a3c[0])
        enc3_a3c = self.encoder3(outs_a3c[1])
        enc4_a3c = self.encoder4(outs_a3c[2])
        enc_hidden_a3c = self.encoder5(outs_a3c[3])
        dec3_a3c = self.decoder5(enc_hidden_a3c, enc4_a3c)
        dec2_a3c = self.decoder4(dec3_a3c, enc3_a3c)
        dec1_a3c = self.decoder3(dec2_a3c, enc2_a3c)  # (B, 32, D/2, H/2, W/2)

        # 处理 A4C 切面
        outs_a4c = self.vit(x_a4c)
        enc2_a4c = self.encoder2(outs_a4c[0])
        enc3_a4c = self.encoder3(outs_a4c[1])
        enc4_a4c = self.encoder4(outs_a4c[2])
        enc_hidden_a4c = self.encoder5(outs_a4c[3])
        dec3_a4c = self.decoder5(enc_hidden_a4c, enc4_a4c)
        dec2_a4c = self.decoder4(dec3_a4c, enc3_a4c)
        dec1_a4c = self.decoder3(dec2_a4c, enc2_a4c)  # (B, 32, D/2, H/2, W/2)

        # 处理 APSAX 切面
        outs_apsax = self.vit(x_apsax)
        enc2_apsax = self.encoder2(outs_apsax[0])
        enc3_apsax = self.encoder3(outs_apsax[1])
        enc4_apsax = self.encoder4(outs_apsax[2])
        enc_hidden_apsax = self.encoder5(outs_apsax[3])
        dec3_apsax = self.decoder5(enc_hidden_apsax, enc4_apsax)
        dec2_apsax = self.decoder4(dec3_apsax, enc3_apsax)
        dec1_apsax = self.decoder3(dec2_apsax, enc2_apsax)  # (B, 32, D/2, H/2, W/2)

        # 处理 MVSAX 切面
        outs_mvsax = self.vit(x_mvsax)
        enc2_mvsax = self.encoder2(outs_mvsax[0])
        enc3_mvsax = self.encoder3(outs_mvsax[1])
        enc4_mvsax = self.encoder4(outs_mvsax[2])
        enc_hidden_mvsax = self.encoder5(outs_mvsax[3])
        dec3_mvsax = self.decoder5(enc_hidden_mvsax, enc4_mvsax)
        dec2_mvsax = self.decoder4(dec3_mvsax, enc3_mvsax)
        dec1_mvsax = self.decoder3(dec2_mvsax, enc2_mvsax)  # (B, 32, D/2, H/2, W/2)

        # 处理 PMSAX 切面
        outs_pmsax = self.vit(x_pmsax)
        enc2_pmsax = self.encoder2(outs_pmsax[0])
        enc3_pmsax = self.encoder3(outs_pmsax[1])
        enc4_pmsax = self.encoder4(outs_pmsax[2])
        enc_hidden_pmsax = self.encoder5(outs_pmsax[3])
        dec3_pmsax = self.decoder5(enc_hidden_pmsax, enc4_pmsax)
        dec2_pmsax = self.decoder4(dec3_pmsax, enc3_pmsax)
        dec1_pmsax = self.decoder3(dec2_pmsax, enc2_pmsax)  # (B, 32, D/2, H/2, W/2)

        # 保留4x4x4的空间结构
        # dec1: (B, 32, D, H, W) -> (B, 32*4*4*4) = (B, 512)
        dec1_pooled_a2c = F.adaptive_avg_pool3d(dec1_a2c, (4, 4, 4))
        dec1_flat_a2c = dec1_pooled_a2c.view(dec1_a2c.size(0), -1)


        dec1_pooled_a3c = F.adaptive_avg_pool3d(dec1_a3c, (4, 4, 4))
        dec1_flat_a3c = dec1_pooled_a3c.view(dec1_a3c.size(0), -1)

        dec1_pooled_a4c = F.adaptive_avg_pool3d(dec1_a4c, (4, 4, 4))
        dec1_flat_a4c = dec1_pooled_a4c.view(dec1_a4c.size(0), -1)

        dec1_pooled_apsax = F.adaptive_avg_pool3d(dec1_apsax, (4, 4, 4))
        dec1_flat_apsax = dec1_pooled_apsax.view(dec1_apsax.size(0), -1)

        dec1_pooled_mvsax = F.adaptive_avg_pool3d(dec1_mvsax, (4, 4, 4))
        dec1_flat_mvsax = dec1_pooled_mvsax.view(dec1_mvsax.size(0), -1)

        dec1_pooled_pmsax = F.adaptive_avg_pool3d(dec1_pmsax, (4, 4, 4))
        dec1_flat_pmsax = dec1_pooled_pmsax.view(dec1_pmsax.size(0), -1)

        # --------------------- 融合与分类 ---------------------
        # 1. 特征融合（通道拼接）
        fused_feat = torch.cat([dec1_flat_a2c, dec1_flat_a3c, dec1_flat_a4c, dec1_flat_apsax, dec1_flat_mvsax, dec1_flat_pmsax], dim=1)

        # 2. 送入 MLP 进行分类
        # cls_out: (B, 1)
        logits = self.classification_head(fused_feat)

        # 返回分割输出和分类输出
        if return_features:
            return logits, fused_feat
        # return seg_out, logits
        return logits


