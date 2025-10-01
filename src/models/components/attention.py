import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """自注意力模块，带有残差连接。"""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class AttnFusion(nn.Module):
    """
    使用自注意力和CLS Token进行多视图特征融合。
    """

    def __init__(self, view_count=2, embed_dim=512, layers=1):
        super().__init__()
        transformer_heads = embed_dim // 64
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.empty(view_count + 1, 1, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)]
        )

    def forward(self, x):
        batch_size = x.size(1)
        cls_tokens = self.cls_token.expand(1, batch_size, -1)
        x = torch.cat([cls_tokens, x], dim=0)
        x = x + self.positional_embedding
        x = self.resblocks(x)
        return x[0]  # 返回CLS Token的输出


class CrossAttentionBlock(nn.Module):
    """交叉注意力模块"""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_1(y), self.ln_1(y), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttention(nn.Module):
    """多层交叉注意力"""

    def __init__(self, d_model: int, n_head: int, layers: int):
        super().__init__()
        self.blocks = nn.Sequential(
            *[CrossAttentionBlock(d_model, n_head) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.blocks(x)


class VideoTextCrossAttention(nn.Module):
    """视频-文本交叉注意力模块"""

    def __init__(self, embed_dim=512, layers=4):
        super().__init__()
        transformer_heads = embed_dim // 64
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_text = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer = CrossAttention(embed_dim, transformer_heads, layers)

    def forward(self, video_feature, text_feature):
        view_size, batch_size, emb_dim = video_feature.shape
        prompt_size, _ = text_feature.shape

        video_feature = video_feature.view(1, -1, emb_dim)
        text_feature = text_feature.unsqueeze(1).expand(-1, view_size * batch_size, -1)

        cls_tokens_video = self.cls_token_video.expand(1, batch_size * view_size, -1)
        cls_tokens_text = self.cls_token_text.expand(1, batch_size * view_size, -1)

        video_feature = torch.cat([cls_tokens_video, video_feature], dim=0)
        text_feature = torch.cat([cls_tokens_text, text_feature], dim=0)

        # 取文本的CLS token作为query
        query = text_feature[0].unsqueeze(0)
        # 视频特征作为key和value
        output = self.transformer(query, video_feature)

        return output.view(view_size, batch_size, emb_dim)



