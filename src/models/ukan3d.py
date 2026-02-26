import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
# from monai.utils import alias, deprecated_arg, export

# from simlvseg.model.utils.kan import KANBlock
from .components.kan import KANBlock

class UNet3DbottleKAN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64, 64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128, 128, 128, 128, 128])
        # self.encoder5 = self._create_encoder_block(128, [256, 256, 256])

        self.bottleKAN = KANBlock(dim=128, num_heads=8)
        
        self.decoder4 = self._create_decoder_block(128*2, 128, False)
        self.decoder3 = self._create_decoder_block(64*2, 64, False)
        self.decoder2 = self._create_decoder_block(32*2, 32, False)
        self.decoder1 = self._create_decoder_block(16*2, 1, True)
        
        self.upconv5 = self._create_up_conv(128, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)
        
        self.maxpool = nn.MaxPool3d(2, 2)
    
    def forward(self, x):
        _, _, h, w, d = x.shape
        
        if (h%16 != 0) or (w%16 != 0) or (d%16 != 0):
            raise ValueError(f"Invalid volume size ({h}, {w}, {d}). The dimension need to be divisible by 16.")
        
        x1 = self.encoder1(x)
        x  = self.maxpool(x1)
        
        x2 = self.encoder2(x)
        x  = self.maxpool(x2)
        
        x3 = self.encoder3(x)
        x  = self.maxpool(x3)
        
        x4 = self.encoder4(x)
        x  = self.maxpool(x4)
        
        # x = self.encoder5(x)  # [1, 128, 14, 14, 4] -> [1, 256, 7 ,7, 2]
        x = self.bottleKAN(x)
        
        x = self.upconv5(x)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)
        
        return x
    
    def _create_encoder_block(
        self,
        in_channel,
        channels,
        down_sampling=False,
        ):
        
        def _create_residual_unit(
            in_channels, out_channels, strides,
            ):
            return ResidualUnit(
                3,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=3,
                subunits=2,
                act=Act.PRELU,
                norm=Norm.INSTANCE,
                dropout=0.0,
                bias=True,
                adn_ordering="NDA",
            )
        
        _channels = [in_channel, *channels]
        
        units = []
        for i in range(len(channels) - 1):
            units.append(_create_residual_unit(_channels[i], _channels[i+1], 1))
        units.append(
            _create_residual_unit(channels[-2], channels[-1], 2 if down_sampling else 1)
        )
        
        return nn.Sequential(*units)
    
    def _create_decoder_block(
        self,
        in_channels,
        out_channels,
        is_top,
    ):
        res_unit = ResidualUnit(
            3,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=is_top,
            adn_ordering="NDA",
        )
        
        return res_unit
    
    def _create_up_conv(
        self,
        in_channels,
        out_channels,
    ):
        return Convolution(
            3,
            in_channels,
            out_channels,
            strides=2,
            kernel_size=3,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            is_transposed=True,
            adn_ordering="NDA",
        )

class UNet3DSmall(UNet3DbottleKAN):
    def __init__(self):
        super().__init__()
        
        self.encoder1 = self._create_encoder_block(3, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128])
        # self.encoder5 = self._create_encoder_block(128, [256, 256])
        
        self.decoder4 = self._create_decoder_block(128*2, 128, False)
        self.decoder3 = self._create_decoder_block(64*2, 64, False)
        self.decoder2 = self._create_decoder_block(32*2, 32, False)
        self.decoder1 = self._create_decoder_block(16*2, 1, True)
        
        self.upconv5 = self._create_up_conv(128, 128)
        self.upconv4 = self._create_up_conv(128, 64)
        self.upconv3 = self._create_up_conv(64, 32)
        self.upconv2 = self._create_up_conv(32, 16)
        
        self.maxpool = nn.MaxPool3d(2, 2)


class UKAN3DClassification(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, dropout_prob=0.2):
        super().__init__()

        # --- 1. Encoder 部分 (与 UNet3DbottleKAN 保持一致) ---
        # 注意：原代码硬编码了 input=3，这里改为参数传入，增加灵活性
        self.encoder1 = self._create_encoder_block(in_channels, [16, 16])
        self.encoder2 = self._create_encoder_block(16, [32, 32, 32])
        self.encoder3 = self._create_encoder_block(32, [64, 64, 64, 64])
        self.encoder4 = self._create_encoder_block(64, [128, 128, 128, 128, 128, 128])

        self.maxpool = nn.MaxPool3d(2, 2)

        # --- 2. Bottleneck (KANBlock) ---
        # 这里的 dim=128 对应 encoder4 的输出通道
        self.bottleKAN = KANBlock(dim=128, num_heads=8)

        # --- 3. Classification Head (分类头) ---
        # 将 (Batch, 128, D, H, W) -> (Batch, 128, 1, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout(p=dropout_prob)

        # 二分类通常输出 1 (配合 BCEWithLogitsLoss)
        # 如果是多分类，num_classes 设为对应类别数
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x_a2c: torch.Tensor, x_a4c: torch.Tensor,return_features: bool = False):
        # 检查输入尺寸 (可选)
        _, _, h, w, d = x_a4c.shape
        # 原始代码有尺寸检查，为了防止 encoder 下采样出错，保留是好习惯
        # if (h%16 != 0) or (w%16 != 0) or (d%16 != 0):
        #     warnings.warn(f"Input shape ({h},{w},{d}) is not divisible by 16, this might cause shape mismatch in skipped connections if it were a UNet. For classification, it is safer but ensure size > 16.")

        # --- Encoder Forward ---
        x = self.encoder1(x_a4c)
        x = self.maxpool(x)

        x = self.encoder2(x)
        x = self.maxpool(x)

        x = self.encoder3(x)
        x = self.maxpool(x)

        x = self.encoder4(x)
        x = self.maxpool(x)

        # --- Bottleneck ---
        # 输出形状: [Batch, 128, H/16, W/16, D/16]
        x = self.bottleKAN(x)

        # --- Classification Head ---
        # 1. 全局平均池化
        x = self.avg_pool(x)

        # 2. 展平: [Batch, 128, 1, 1, 1] -> [Batch, 128]
        x = x.flatten(1)

        # 3. Dropout & Linear
        x = self.dropout(x)
        logits = self.fc(x)

        # 返回分类输出
        if return_features:
            return logits, x

        return logits

    # 复制原类中的 helper method，确保独立运行
    def _create_encoder_block(self, in_channel, channels, down_sampling=False):
        def _create_residual_unit(in_channels, out_channels, strides):
            return ResidualUnit(
                3,  # spatial_dims
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=3,
                subunits=2,
                act=Act.PRELU,
                norm=Norm.INSTANCE,
                dropout=0.0,
                bias=True,
                adn_ordering="NDA",
            )

        _channels = [in_channel, *channels]

        units = []
        for i in range(len(channels) - 1):
            units.append(_create_residual_unit(_channels[i], _channels[i + 1], 1))
        units.append(
            _create_residual_unit(channels[-2], channels[-1], 2 if down_sampling else 1)
        )

        return nn.Sequential(*units)

