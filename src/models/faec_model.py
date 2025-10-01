# src/models/faec_model.py

import torch
import torch.nn as nn
from .base_model import BaseModel
# ** 核心修正点 1: 导入新的、分离的编码器模块 **
from .components.echoprime_encoders import EchoPrimeVideoEncoder, EchoPrimeTextEncoder
from .components.mlps import ClsMLP


class FAECModel(BaseModel):
    """
    FAECModel - 已最终升级为使用与EchoPrime完全一致的视频和文本编码器。
    """

    def __init__(self,
                 video_encoder_path,
                 text_encoder_path,
                 embed_dim=512,
                 num_layers=6,
                 n_head=8,
                 frozen_video_encoder=True,
                 frozen_text_encoder=True,
                 supervision_indices=(2, 4, 5),
                 num_knowledge_prompts=36):
        super().__init__()

        # ** 核心修正点 2: 分别创建视频和文本编码器实例 **
        self.video_encoder = EchoPrimeVideoEncoder(
            pretrained_path=video_encoder_path,
            frozen=frozen_video_encoder
        )
        self.text_encoder = EchoPrimeTextEncoder(
            pretrained_path=text_encoder_path,
            frozen=frozen_text_encoder
        )

        self.supervision_indices = supervision_indices

        self.belief_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding_evidence = nn.Parameter(torch.randn(1, 3, embed_dim))
        self.positional_embedding_knowledge = nn.Parameter(torch.randn(1, num_knowledge_prompts, embed_dim))

        self.transformer_layers = nn.ModuleList([
            KnowledgeGuidedLayer(
                embed_dim=embed_dim, n_head=n_head,
                dim_feedforward=embed_dim * 4, dropout=0.1
            ) for _ in range(num_layers)
        ])

        self.classifiers = nn.ModuleDict()
        for i in self.supervision_indices:
            self.classifiers[f'head_{i}'] = ClsMLP(embed_dim, 1)

    @property
    def device(self):
        return self.belief_token.device

    def forward(self, a2c_video, a4c_video, text_prompts, return_attention=False, return_features=False):
        # ** 核心修正点 3: 通过新的编码器模块获取特征 **
        a2c_feat = self.video_encoder(a2c_video).unsqueeze(1)
        a4c_feat = self.video_encoder(a4c_video).unsqueeze(1)

        # 使用新的文本编码器直接处理原始文本列表
        text_features_all = self.text_encoder(text_prompts).expand(a2c_feat.shape[0], -1, -1)

        # --- 后续逻辑保持不变 ---
        bs = a2c_feat.shape[0]
        belief_tokens = self.belief_token.expand(bs, -1, -1)

        evidence_sequence = torch.cat([belief_tokens, a2c_feat, a4c_feat], dim=1)
        evidence_sequence += self.positional_embedding_evidence

        knowledge_sequence = text_features_all + self.positional_embedding_knowledge

        logits_dict = {}
        all_self_attns = []
        all_cross_attns = []
        final_belief_state = None

        for i, layer in enumerate(self.transformer_layers):
            evidence_sequence, self_attn, cross_attn = layer(evidence_sequence, knowledge_sequence,
                                                             need_weights=return_attention)

            if return_attention:
                all_self_attns.append(self_attn)
                all_cross_attns.append(cross_attn)

            belief_state = evidence_sequence[:, 0, :]
            if i in self.supervision_indices:
                key = 'final' if i == max(self.supervision_indices) else f'aux_{i}'
                logits_dict[key] = self.classifiers[f'head_{i}'](belief_state)
                if key == 'final':
                    final_belief_state = belief_state

        if return_attention:
            return logits_dict, all_self_attns, all_cross_attns

        if return_features:
            return logits_dict, final_belief_state

        return logits_dict


# KnowledgeGuidedLayer 保持不变
class KnowledgeGuidedLayer(nn.Module):
    def __init__(self, embed_dim, n_head, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, evidence_sequence, knowledge_sequence, need_weights=False):
        self_attn_out, self_attn_weights = self.self_attn(evidence_sequence, evidence_sequence, evidence_sequence,
                                                          need_weights=need_weights, average_attn_weights=False)
        evidence_sequence = self.norm1(evidence_sequence + self.dropout1(self_attn_out))

        belief_token = evidence_sequence[:, 0:1, :]

        cross_attn_out, cross_attn_weights = self.cross_attn(query=self.norm2(belief_token),
                                                             key=knowledge_sequence,
                                                             value=knowledge_sequence,
                                                             need_weights=need_weights,
                                                             average_attn_weights=False)
        belief_token = self.norm2(belief_token + self.dropout2(cross_attn_out))

        ffn_out = self.ffn(belief_token)
        belief_token = self.norm3(belief_token + self.dropout3(ffn_out))

        updated_evidence_sequence = torch.cat([belief_token, evidence_sequence[:, 1:, :]], dim=1)

        return updated_evidence_sequence, self_attn_weights, cross_attn_weights