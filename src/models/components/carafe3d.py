import torch
import torch.nn as nn
from torch.nn import functional as F
# from mmcv.ops.carafe import CARAFEPack


def pixel_shuffle_3d(input, upscale_factor):
    """
    自定义 3D pixel shuffle 操作。

    :param input: 输入张量，形状为 [N, C, H, W, D]
    :param upscale_factor: 上采样因子
    :return: 输出张量，形状为 [N, C // (upscale_factor^3), H*upscale_factor, W*upscale_factor, D*upscale_factor]
    """
    N, C, H, W, D = input.size()
    r = upscale_factor

    # 检查通道数是否可以被 r^3 整除
    assert C % (r ** 3) == 0, f"输入通道数 {C} 必须能被 upscale_factor^3 ({r ** 3}) 整除"

    # 新的通道数
    new_C = C // (r ** 3)

    # 调整形状: 将通道维度展开到空间维度
    output = input.view(N, new_C, r, r, r, H, W, D)

    # 交换维度: 将 (r, r, r) 转移到 (H, W, D)
    output = output.permute(0, 1, 5, 2, 6, 3, 7, 4)

    # 合并空间维度
    output = output.reshape(N, new_C, H * r, W * r, D * r)

    return output


class CARAFE(nn.Module):
    # CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        # print("up shape:",out_tensor.shape)
        return out_tensor


# class CARAFE3D(nn.Module):
#     def __init__(self, c1, c2, spatial_up_factor=2, temporal_up_factor=2, kernel_size=3):
#         super(CARAFE3D, self).__init__()
#         self.temporal_up_factor = temporal_up_factor
#         self.spatial_upsample = CARAFE(c1, c2, kernel_size=kernel_size, up_factor=spatial_up_factor)
#         # self.spatial_upsample = CARAFEPack(channels=c1, scale_factor=spatial_up_factor)
#         self.temporal_upsample = nn.ConvTranspose3d(
#             in_channels=c2, out_channels=c2,
#             kernel_size=(temporal_up_factor, 1, 1),
#             stride=(temporal_up_factor, 1, 1),
#             padding=(0, 0, 0),
#         )
#
#     def forward(self, x):
#         # Input shape: (N, C, H, W, T)
#         N, C, H, W, T = x.shape
#         x = x.permute(0, 4, 2, 1, 3).reshape(N * T, C, H, W)  # (N*T, C, H, W)
#         x = self.spatial_upsample(x)  # (N*T, C', H', W')
#         C_up, H_up, W_up = x.shape[1], x.shape[2], x.shape[3]
#         x = x.reshape(N, T, C_up, H_up, W_up).permute(0, 2, 3, 4, 1)  # (N, C', T, H', W')
#
#         # 时间维度的上采样使用转置卷积
#         x = self.temporal_upsample(x)  # (N, C', T', H', W')
#
#         # 时间维度的上采样使用双线性插值
#         # T_up = T * self.temporal_up_factor
#         # x = F.interpolate(
#         #     x, size=(H_up, W_up, T_up), mode='trilinear', align_corners=False
#         # )  # Interpolate along temporal dimension
#         return x


class CARAFE3D(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE3D, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor

        # 降维模块，用1x1x1卷积减少通道数到 c1 // 4
        self.down = nn.Conv3d(c1, c1 // 4, kernel_size=1)

        # 编码生成上采样卷积核
        self.encoder = nn.Conv3d(
            c1 // 4,
            self.up_factor ** 3 * self.kernel_size ** 3,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )

        # 输出模块，用1x1x1卷积调整输出通道数到 c2
        self.out = nn.Conv3d(c1, c2, kernel_size=1)

    def forward(self, x):
        """
        前向传播函数，输入为 [N, C, H, W, D]
        """
        N, C, H, W, D = x.size()

        # Kernel Prediction Module
        kernel_tensor = self.down(x)  # 降维操作
        kernel_tensor = self.encoder(kernel_tensor)  # 编码生成上采样卷积核
        kernel_tensor = kernel_tensor.contiguous()  # 确保内存连续性
        kernel_tensor = pixel_shuffle_3d(kernel_tensor, self.up_factor)  # 像素重排操作
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # 权重归一化

        # unfold 操作适配 3D 空间
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.unfold(4, self.up_factor, step=self.up_factor)
        kernel_tensor = kernel_tensor.reshape(
            N, self.kernel_size ** 3, H, W, D, self.up_factor ** 3
        )
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 4, 1, 5)

        # Content-Aware ReAssembly Module
        x = F.pad(
            x,
            pad=(self.kernel_size // 2,) * 6,
            mode='constant',
            value=0
        )
        x = x.unfold(2, self.kernel_size, step=1)
        x = x.unfold(3, self.kernel_size, step=1)
        x = x.unfold(4, self.kernel_size, step=1)
        x = x.reshape(N, C, H, W, D, -1)
        x = x.permute(0, 2, 3, 4, 1, 5)

        out_tensor = torch.matmul(x, kernel_tensor)  # 点积操作
        out_tensor = out_tensor.reshape(N, H, W, D, -1)
        out_tensor = out_tensor.permute(0, 4, 1, 2, 3)
        out_tensor = out_tensor.contiguous()  # 确保内存连续性

        # 调试打印 Pixel Shuffle 之前的张量形状
        # print("Pixel Shuffle 前形状:", out_tensor.shape)

        out_tensor = pixel_shuffle_3d(out_tensor, self.up_factor)  # 像素重排操作

        # 调试打印 Pixel Shuffle 之后的张量形状
        # print("Pixel Shuffle 后形状:", out_tensor.shape)

        out_tensor = self.out(out_tensor)  # 调整输出通道数

        return out_tensor  # 返回上采样后的特征图


class CARAFE3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, dropout=0.0):
        super(CARAFE3DBlock, self).__init__()

        # 转置卷积
        # self.conv = nn.ConvTranspose3d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=kernel_size // 2,
        #     output_padding=stride - 1,  # 确保输出尺寸正确
        #     bias=True
        # )

        # self.conv = CARAFE3D(in_channels, out_channels, spatial_up_factor=stride, temporal_up_factor=stride, kernel_size=3)
        self.conv = CARAFE3D(in_channels, out_channels, up_factor=2, kernel_size=3)

        # Instance Normalization
        self.instance_norm = nn.InstanceNorm3d(out_channels)

        # PReLU 激活函数
        self.activation = nn.PReLU()

        # Dropout 层
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.conv(x)  # 转置卷积
        x = self.instance_norm(x)  # Instance Normalization
        x = self.activation(x)  # 激活函数
        if self.dropout:  # 可选的 Dropout
            x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Test the VideoUpsample module
    input_tensor = torch.randn(1, 128, 32, 32, 8).cuda()  # Example video input (batch size 1, 64 channels, 8 frames, 32x32 size)
    model = CARAFE3D(c1=128, c2=64, up_factor=4).cuda()
    # model = CARAFE3DBlock(128, 64).cuda()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
