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

class MIMambaEchoPrimeVideo(nn.Module):
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

            # 新增视频编码器相关参数
            embed_dim=512,  # EchoPrimeVideoEncoder 的输出维度
            video_encoder_path=None,  # 预训练权重路径
            frozen_video_encoder=True,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.embed_dim = embed_dim  # 视频特征维度

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans,
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
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=out_chans)

        # ------------------- 改造：新增视频与分类模块 -------------------

        # 1. 视频编码器 (使用提供的 EchoPrimeVideoEncoder)
        self.video_encoder = EchoPrimeVideoEncoder(
            pretrained_path=video_encoder_path,
            frozen=frozen_video_encoder
        )


        # 2. 通道对齐层
        dec_feat_channel = self.feat_size[1] * 64

        # 3. 分类 MLP
        # 输入维度: dec1_channels + video_embed_dim
        self.classification_head = ClsMLP(
            in_dim=dec_feat_channel + embed_dim,
            out_dim=1
        )


    def forward(self, x_a2c: torch.Tensor, x_a4c: torch.Tensor, return_features: bool = False):

        # --------------------- Mamba 路径 ---------------------
        outs = self.vit(x_a4c)  # Mamba Encoder features (x2, x3, x4, x_hidden)
        enc1 = self.encoder1(x_a4c)  # (B, 16, D, H, W)
        x2 = outs[0]
        enc2 = self.encoder2(x2)  # (B, 32, D/2, H/2, W/2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)  # (B, 64, D/4, H/4, W/4)
        x4 = outs[2]
        enc4 = self.encoder4(x4)  # (B, 128, D/8, H/8, W/8)
        enc_hidden = self.encoder5(outs[3])  # (B, 256, D/16, H/16, W/16)
        dec3 = self.decoder5(enc_hidden, enc4)  # (B, 128, D/8, H/8, W/8)

        # 倒数第二个 decoder 层的特征 (feat_size[2] = 64)
        dec2 = self.decoder4(dec3, enc3)  # (B, 64, D/4, H/4, W/4)

        # 倒数第二个 decoder 层的特征 (feat_size[1] = 32)
        dec1 = self.decoder3(dec2, enc2)  # (B, 32, D/2, H/2, W/2)

        # 倒数第一个 decoder 层的特征 (feat_size[1] = 16)
        dec0 = self.decoder2(dec1, enc1)  # (B, 16, D, H, W)

        seg_out = self.decoder1(dec0)  # (B, 16, D, H, W)
        # seg_out = self.out(seg_out)  # (B, out_chans, D, H, W)

        # --------------------- echo_prime路径 ---------------------

        # 1. 视频特征提取
        # video_in: (B, C_video, T, H, W) -> video_feat: (B, 512)
        video_feat = self.video_encoder(x_a2c)

        # 保留4x4x4的空间结构
        # dec1: (B, 32, D, H, W) -> (B, 32*4*4*4) = (B, 512)
        dec1_pooled = F.adaptive_avg_pool3d(dec1, (4, 4, 4))
        dec1_flat = dec1_pooled.view(dec1.size(0), -1)

        #
        # --------------------- 融合与分类 ---------------------
        # 1. 特征融合（通道拼接）
        fused_feat = torch.cat([dec1_flat, video_feat], dim=1)

        # # 2. 可训练权重融合
        # weights = self.fusion_softmax(self.fusion_weight)  # 归一化的融合权重
        # dec1_weighted = weights[0] * dec1_flat
        # video_weighted = weights[1] * video_feat
        # fused_feat = torch.cat([dec1_weighted, video_weighted], dim=1)

        # 2. 送入 MLP 进行二分类
        # cls_out: (B, 1)
        logits = self.classification_head(fused_feat)

        # 返回分割输出和分类输出
        if return_features:
            return logits, fused_feat
        # return seg_out, logits
        return logits


if __name__ == "__main__":

    # 创建模型实例
    model = MIMambaEchoPrimeVideo(
        in_chans=3,
        out_chans=1,
        depths= [2, 2, 2, 2],
        feat_size= [16, 32, 64, 128],
        hidden_size= 256,
        video_embed_dim= 512,
        video_encoder_path= '../../model_weight/echo_prime_encoder.pt',  # 在实际测试中可能需要提供预训练路径
        frozen_video_encoder= False  # 为了测试，暂时不解冻
     ).cuda()

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 主输入 (医学图像)
    x_in = torch.randn(2, 3, 16, 224, 224).cuda()

    # 设置模型为评估模式
    model.eval()

    seg_out, cls_out = model(x_in)

    # 打印输出信息
    print(f"\nSegmentation output shape: {seg_out.shape}")
    print(f"Classification output shape: {cls_out.shape}")

    sigmoid_outputs = torch.sigmoid(cls_out)
    print(f"Classification output (sigmoid): {sigmoid_outputs}")

