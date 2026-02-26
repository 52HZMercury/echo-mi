import torch
import math
import copy
from torch import nn
from einops import rearrange
from functools import partial
import torch.nn.functional as F
# 假设 CARAFE3D 依然可用，如果不可用可以注释掉或者用 nn.Upsample 替代
# from simlvseg.model.utils.carafe3d import CARAFE3D
from .components.carafe3d import CARAFE3D

def build_segformer3d_model(config=None):
    model = SegFormer3D(
        in_channels=config["model_parameters"]["in_channels"],
        sr_ratios=config["model_parameters"]["sr_ratios"],
        embed_dims=config["model_parameters"]["embed_dims"],
        patch_kernel_size=config["model_parameters"]["patch_kernel_size"],
        patch_stride=config["model_parameters"]["patch_stride"],
        patch_padding=config["model_parameters"]["patch_padding"],
        mlp_ratios=config["model_parameters"]["mlp_ratios"],
        num_heads=config["model_parameters"]["num_heads"],
        depths=config["model_parameters"]["depths"],
        decoder_head_embedding_dim=config["model_parameters"]["decoder_head_embedding_dim"],
        num_classes=config["model_parameters"]["num_classes"],
        decoder_dropout=config["model_parameters"]["decoder_dropout"],
    )
    return model

#  ClsMLP 是一个三层 MLP
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

class SegFormer3D(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            sr_ratios: list = [4, 2, 1, 1],
            embed_dims: list = [32, 64, 160, 256],
            patch_kernel_size: list = [7, 3, 3, 3],
            patch_stride: list = [4, 2, 2, 2],
            patch_padding: list = [3, 1, 1, 1],
            mlp_ratios: list = [4, 4, 4, 4],
            num_heads: list = [1, 2, 5, 8],
            depths: list = [2, 2, 2, 2],
            decoder_head_embedding_dim: int = 256,
            num_classes: int = 1,
            decoder_dropout: float = 0.0,
    ):
        super().__init__()
        self.segformer_encoder = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )
        reversed_embed_dims = embed_dims[::-1]
        self.segformer_decoder = SegFormerDecoderHead(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )
        self.apply(self._init_weights)

        self.classification_head = ClsMLP(
            in_dim=64,
            out_dim=1
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x_a2c: torch.Tensor, x_a4c: torch.Tensor, return_features: bool = False):
        x = self.segformer_encoder(x_a4c)
        c1, c2, c3, c4 = x[0], x[1], x[2], x[3]
        x = self.segformer_decoder(c1, c2, c3, c4)

        x_pooled = F.adaptive_avg_pool3d(x, (4, 4, 4))
        x = x_pooled.view(x.size(0), -1)

        logits = self.classification_head(x)
        # 返回分割输出和分类输出
        if return_features:
            return logits, x

        return logits



# ----------------------------------------------------- encoder -----------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channel: int = 3,
            embed_dim: int = 768,
            kernel_size: int = 7,
            stride: int = 4,
            padding: int = 3,
    ):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: B, C, D, H, W
        x = self.patch_embeddings(x)
        # 记录下卷积后的空间尺寸
        B, C, D, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        # 返回 embeddings 和 当前的空间尺寸 (D, H, W)
        return x, (D, H, W)


class SelfAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 8,
            sr_ratio: int = 2,
            qkv_bias: bool = False,
            attn_dropout: float = 0.0,
            proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        self.attention_head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        D, H, W = spatial_shape  # 解包传入的非立方体尺寸

        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            # 使用传入的 D, H, W 进行 reshape
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            # 经过 sr 卷积层下采样
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]

        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.num_heads)
        attnention_prob = attention_score.softmax(dim=-1)
        attnention_prob = self.attn_dropout(attnention_prob)
        out = (attnention_prob @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            mlp_ratio: int = 2,
            num_heads: int = 8,
            sr_ratio: int = 2,
            qkv_bias: bool = False,
            attn_dropout: float = 0.0,
            proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        # 将 dropout 参数传递给 _MLP
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=proj_dropout)

    def forward(self, x, spatial_shape):
        # 传递 spatial_shape
        x = x + self.attention(self.norm1(x), spatial_shape)
        x = x + self.mlp(self.norm2(x), spatial_shape)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            sr_ratios: list = [8, 4, 2, 1],
            embed_dims: list = [64, 128, 320, 512],
            patch_kernel_size: list = [7, 3, 3, 3],
            patch_stride: list = [4, 2, 2, 2],
            patch_padding: list = [3, 1, 1, 1],
            mlp_ratios: list = [2, 2, 2, 2],
            num_heads: list = [1, 2, 5, 8],
            depths: list = [2, 2, 2, 2],
    ):
        super().__init__()

        # Patch Embeddings
        self.embed_1 = PatchEmbedding(
            in_channel=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=patch_kernel_size[0],
            stride=patch_stride[0],
            padding=patch_padding[0],
        )
        self.embed_2 = PatchEmbedding(
            in_channel=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=patch_kernel_size[1],
            stride=patch_stride[1],
            padding=patch_padding[1],
        )
        self.embed_3 = PatchEmbedding(
            in_channel=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=patch_kernel_size[2],
            stride=patch_stride[2],
            padding=patch_padding[2],
        )
        self.embed_4 = PatchEmbedding(
            in_channel=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=patch_kernel_size[3],
            stride=patch_stride[3],
            padding=patch_padding[3],
        )

        # Transformer Blocks
        self.tf_block1 = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                sr_ratio=sr_ratios[0],
                qkv_bias=True,
            ) for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.tf_block2 = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                sr_ratio=sr_ratios[1],
                qkv_bias=True,
            ) for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.tf_block3 = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                sr_ratio=sr_ratios[2],
                qkv_bias=True,
            ) for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.tf_block4 = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                sr_ratio=sr_ratios[3],
                qkv_bias=True,
            ) for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        out = []

        # stage 1
        x, (D1, H1, W1) = self.embed_1(x)  # 接收形状
        for blk in self.tf_block1:
            x = blk(x, (D1, H1, W1))  # 传入形状
        x = self.norm1(x)
        # 恢复形状用于输出: (B, N, C) -> (B, C, D, H, W)
        x_out = x.transpose(1, 2).reshape(-1, x.shape[2], D1, H1, W1)
        out.append(x_out)

        # stage 2
        x, (D2, H2, W2) = self.embed_2(x_out)
        for blk in self.tf_block2:
            x = blk(x, (D2, H2, W2))
        x = self.norm2(x)
        x_out = x.transpose(1, 2).reshape(-1, x.shape[2], D2, H2, W2)
        out.append(x_out)

        # stage 3
        x, (D3, H3, W3) = self.embed_3(x_out)
        for blk in self.tf_block3:
            x = blk(x, (D3, H3, W3))
        x = self.norm3(x)
        x_out = x.transpose(1, 2).reshape(-1, x.shape[2], D3, H3, W3)
        out.append(x_out)

        # stage 4
        x, (D4, H4, W4) = self.embed_4(x_out)
        for blk in self.tf_block4:
            x = blk(x, (D4, H4, W4))
        x = self.norm4(x)
        x_out = x.transpose(1, 2).reshape(-1, x.shape[2], D4, H4, W4)
        out.append(x_out)

        return out


class _MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_shape):
        x = self.fc1(x)
        x = self.dwconv(x, spatial_shape)  # 传入形状到 DWConv
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        D, H, W = spatial_shape
        # 使用传入的 D, H, W 将序列恢复为 3D 体素
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ----------------------------------------------------- decoder -------------------
class MLP_(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        x = self.bn(x)
        return x


class SegFormerDecoderHead(nn.Module):
    def __init__(
            self,
            input_feature_dims: list = [512, 320, 128, 64],
            decoder_head_embedding_dim: int = 256,
            num_classes: int = 3,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.linear_c4 = MLP_(input_dim=input_feature_dims[0], embed_dim=decoder_head_embedding_dim)
        self.linear_c3 = MLP_(input_dim=input_feature_dims[1], embed_dim=decoder_head_embedding_dim)
        self.linear_c2 = MLP_(input_dim=input_feature_dims[2], embed_dim=decoder_head_embedding_dim)
        self.linear_c1 = MLP_(input_dim=input_feature_dims[3], embed_dim=decoder_head_embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        # 如果没有 CARAFE3D，可以使用下面的 Upsample
        # self.upsample_volume = nn.Upsample(scale_factor=4.0, mode="trilinear", align_corners=False)
        # self.final_conv = nn.Conv3d(decoder_head_embedding_dim, num_classes, 1)

        self.upsample_volume = CARAFE3D(decoder_head_embedding_dim, num_classes, up_factor=4)

    def forward(self, c1, c2, c3, c4):
        # c4 shape: B, C, D, H, W
        n, _, _, _, _ = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4]).contiguous()
        _c4 = torch.nn.functional.interpolate(_c4, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4]).contiguous()
        _c3 = torch.nn.functional.interpolate(_c3, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4]).contiguous()
        _c2 = torch.nn.functional.interpolate(_c2, size=c1.size()[2:], mode="trilinear", align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4]).contiguous()

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.upsample_volume(x)
        return x


# 测试 Main 函数
def main():
    import time

    # 模拟非正方体输入
    # (Batch, Channels, Depth, Height, Width)
    # PyTorch 3D 卷积标准输入顺序通常是 (B, C, D, H, W)
    # 你的注释中写的是 (1, 3, 128, 128, 16) -> 这就是非正方体
    INPUT_SIZE = (1, 3, 32, 128, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input Shape: {INPUT_SIZE}")

    model = SegFormer3D().to(device)
    model.eval()

    dummy_input = torch.randn(INPUT_SIZE).to(device)

    # 简单测试一下前向传播是否跑通
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Success! Output Shape: {output.shape}")
    except RuntimeError as e:
        print(f"Inference failed: {e}")
        return

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("-" * 30)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # 计算 FPS
    print("-" * 30)
    print("Calculating FPS...")
    num_iterations = 20
    warmup = 5

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_iterations / total_time

    print(f"Average time per inference: {total_time / num_iterations:.4f} seconds")
    print(f"FPS: {fps:.2f}")
    print("-" * 30)


if __name__ == '__main__':
    main()