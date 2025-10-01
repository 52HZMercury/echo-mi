# src/models/components/encoders.py

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from open_clip import create_model_and_transforms
import copy


class VideoEncoder(pl.LightningModule):
    """视频编码器，包装了预训练的 mvit_v2_s 模型."""

    def __init__(self, pretrained_path, frozen=True):
        super().__init__()
        self.model = torchvision.models.video.mvit_v2_s()
        self.model.head[-1] = nn.Linear(self.model.head[-1].in_features, 512)

        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint.items()}

        self.model.load_state_dict(new_state_dict)

        for param in self.model.parameters():
            param.requires_grad = not frozen

    def forward(self, x):
        return self.model(x)


class TextEncoder(pl.LightningModule):
    """
    文本编码器，包装了预训练的 echo-clip 模型。
    新版本支持从本地权重文件加载，并能智能处理键名不匹配问题。
    """

    def __init__(self, pretrained_path, frozen=True):
        super().__init__()
        # 1. 构建基础模型架构，此时它已经从Hugging Face Hub加载了默认的echo-clip权重
        model, _, _ = create_model_and_transforms(
            "hf-hub:mkaichristensen/echo-clip",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 2. 如果提供了本地权重路径，则用它来覆盖模型中对应的部分
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            # ** 核心修正点: 创建一个智能的键名映射 **
            # 我们只关心文本编码器部分，在open_clip中它叫 'transformer'
            # 在EchoPrime权重中，它叫 'backbone.bert'
            new_state_dict = {}
            for k, v in checkpoint.items():
                # 将 'backbone.bert.embeddings' 映射到 'token_embedding'
                if 'backbone.bert.embeddings' in k:
                    new_key = k.replace('backbone.bert.embeddings', 'token_embedding')
                    new_state_dict[new_key] = v
                # 将 'backbone.bert.encoder.layer' 映射到 'transformer.resblocks'
                elif 'backbone.bert.encoder.layer' in k:
                    new_key = k.replace('backbone.bert.encoder.layer', 'transformer.resblocks')
                    new_state_dict[new_key] = v
                # 将 'text_projection' 直接映射
                elif k.startswith('text_projection'):
                    new_state_dict[k] = v

            # 使用 load_state_dict 并设置 strict=False
            # 这会只加载匹配上的键，并忽略不匹配的键（例如视觉部分），从而避免报错
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

            print("TextEncoder weights loaded with the following status:")
            if missing_keys:
                print(f"  - Missing keys (these are expected, mostly visual part): {len(missing_keys)}")
            if unexpected_keys:
                print(f"  - Unexpected keys (should be empty): {unexpected_keys}")
            print(f"Successfully loaded weights from: {pretrained_path}")

        # 3. 设置冻结状态
        if frozen:
            for param in model.parameters():
                param.requires_grad = False

        self.transformer = model.encode_text

    def forward(self, x):
        return self.transformer(x)


class VideoEncoderForMultiTask(pl.LightningModule):
    """为多任务学习设计的视频编码器，输出两个分支的特征。"""

    def __init__(self, pretrained_path, frozen=True):
        super().__init__()
        base_encoder = torchvision.models.video.mvit_v2_s()
        base_encoder.head[-1] = nn.Linear(base_encoder.head[-1].in_features, 512)
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        base_encoder.load_state_dict(checkpoint)

        for param in base_encoder.parameters():
            param.requires_grad = not frozen

        self.conv_proj = base_encoder.conv_proj
        self.pos_encoding = base_encoder.pos_encoding
        self.blocks = base_encoder.blocks[:-1]
        self.last_block_mi = base_encoder.blocks[-1]
        self.norm = base_encoder.norm
        self.head = base_encoder.head

        self.last_block_view = copy.deepcopy(base_encoder.blocks[-1])
        for param in self.last_block_view.parameters():
            param.requires_grad = True

    def _unsqueeze(self, x):
        if x.dim() == 4:
            return x.unsqueeze(2)
        return x

    def forward(self, x):
        x = self._unsqueeze(x)
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size

        for block in self.blocks:
            x, thw = block(x, thw)

        x_mi, thw_mi = self.last_block_mi(x, thw)
        feature_mi = self.norm(x_mi)[:, 0]
        feature_mi = self.head(feature_mi)

        x_view, _ = self.last_block_view(x, thw)
        feature_view = self.norm(x_view)[:, 0]
        feature_view = self.head(feature_view)

        return feature_view, feature_mi