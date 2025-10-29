# src/models/faec_advanced_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .components.echoprime_encoders import EchoPrimeVideoEncoder, EchoPrimeTextEncoder
from .components.mlps import ClsMLP
from src.utils.prompts import COMMON_KNOWLEDGE, A2C_SPECIFIC_KNOWLEDGE, A4C_SPECIFIC_KNOWLEDGE


# KnowledgeGuidedLayer 是模型的核心推理单元。
# 每一层都负责更新“信念”，模拟一次诊断推理步骤。
class KnowledgeGuidedLayer(nn.Module):
    """
    知识引导层，是FAEC模型中单步推理的核心模块。
    它包含一个自注意力模块用于融合视觉证据，和一个交叉注意力模块用于查询医学知识。
    """
    def __init__(self, embed_dim, n_head, dim_feedforward, dropout):
        super().__init__()
        # 1. 自注意力模块 (Self-Attention)
        # 作用：用于融合和更新“证据序列”内部的信息。
        # 让 belief_token, a2c_feat, a4c_feat 相互交互，聚合最重要的视觉证据信息。
        self.self_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 2. 交叉注意力模块 (Cross-Attention)
        # 作用：让当前的“信念”(belief_token)作为查询(Query)，去“知识序列”(knowledge_sequence)中查找最相关的依据。
        # 这是模型模拟“医生根据当前观察联想相关医学知识”的关键步骤。
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # 3. 前馈网络 (Feed-Forward Network)
        # 作用：对交叉注意力更新后的 belief_token 进行非线性变换，提取更深层次的特征。
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, evidence_sequence, knowledge_sequence, need_weights=False):
        # --- 步骤 1: 证据融合 ---
        # evidence_sequence 作为 Q, K, V 进行自注意力计算，更新序列中的每个token。
        self_attn_out, self_attn_weights = self.self_attn(evidence_sequence, evidence_sequence, evidence_sequence,
                                                          need_weights=need_weights, average_attn_weights=False)
        # 残差连接与层归一化
        evidence_sequence = self.norm1(evidence_sequence + self.dropout1(self_attn_out))

        # 提取更新后的 belief_token，它现在聚合了来自A2C和A4C的最新信息。
        belief_token = evidence_sequence[:, 0:1, :]

        # --- 步骤 2: 知识查询 ---
        # belief_token 作为 Query，去知识序列 knowledge_sequence (作为Key和Value) 中进行交叉注意力计算。
        cross_attn_out, cross_attn_weights = self.cross_attn(query=self.norm2(belief_token),
                                                             key=knowledge_sequence,
                                                             value=knowledge_sequence,
                                                             need_weights=need_weights,
                                                             average_attn_weights=False)
        # 残差连接与层归一化，belief_token 被注入了相关的医学知识。
        belief_token = self.norm2(belief_token + self.dropout2(cross_attn_out))

        # --- 步骤 3: 特征深化 ---
        # 通过FFN深化 belief_token 的特征表示。
        ffn_out = self.ffn(belief_token)
        # 再次进行残差连接与层归一化。
        belief_token = self.norm3(belief_token + self.dropout3(ffn_out))

        # 将最终更新后的 belief_token 与未变的视觉证据token拼接，形成新的证据序列，传递给下一层。
        updated_evidence_sequence = torch.cat([belief_token, evidence_sequence[:, 1:, :]], dim=1)

        return updated_evidence_sequence, self_attn_weights, cross_attn_weights


class FAECAdvancedModel(BaseModel):
    """
    FAEC模型的高级版本V2，实现了两个核心创新：
    1. 精细化证据驱动的知识激活 (Granular ETKA):
       不再使用模糊的知识类别原型，而是让视觉证据与每一条具体的医学知识进行匹配，实现更精准的知识激活。
    2. 带增长激励的置信度引导监督 (CGSC with Growth Incentive):
       在训练时，不仅监督诊断的准确性，还监督模型对其判断的“自信度”，并激励其在推理中解决不确定性。
    """

    def __init__(self,
                 video_encoder_path,
                 text_encoder_path,
                 embed_dim=512,
                 num_layers=6,
                 n_head=8,
                 frozen_video_encoder=True,
                 frozen_text_encoder=True,
                 supervision_indices=(2, 4, 5)):
        super().__init__()

        # --- 1. 基础编码器模块 ---
        # 视频编码器，用于从超声心动图视频中提取高维视觉特征。
        self.video_encoder = EchoPrimeVideoEncoder(
            pretrained_path=video_encoder_path,
            frozen=frozen_video_encoder
        )
        # 文本编码器，用于将医学知识（字符串）转换为高维语义特征向量。
        self.text_encoder = EchoPrimeTextEncoder(
            pretrained_path=text_encoder_path,
            frozen=frozen_text_encoder
        )

        # --- 2. 核心架构参数 ---
        # 定义了在哪些Transformer层级进行监督和输出。这是模型结构和训练动态的关键。
        self.supervision_indices = supervision_indices
        # 从prompts.py加载全部的医学知识文本，形成一个完整的知识库。
        self.prompts_all = COMMON_KNOWLEDGE + A2C_SPECIFIC_KNOWLEDGE + A4C_SPECIFIC_KNOWLEDGE
        self.num_knowledge_prompts = len(self.prompts_all)

        # --- 3. 可学习的Token和位置编码 ---
        # 核心组件：一个可学习的向量，代表了模型在推理过程中的“当前诊断信念”。
        self.belief_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 为证据序列 [belief, a2c, a4c] 提供位置信息。
        self.positional_embedding_evidence = nn.Parameter(torch.randn(1, 3, embed_dim))
        # 为知识序列（36条知识）提供位置信息。
        self.positional_embedding_knowledge = nn.Parameter(torch.randn(1, self.num_knowledge_prompts, embed_dim))

        # --- 4. 深度推理模块 ---
        # 构建一个由多个KnowledgeGuidedLayer组成的堆栈，形成“证据链”。
        self.transformer_layers = nn.ModuleList([
            KnowledgeGuidedLayer(
                embed_dim=embed_dim, n_head=n_head,
                dim_feedforward=embed_dim * 4, dropout=0.1
            ) for _ in range(num_layers)
        ])

        # --- 5. 输出头模块 (CGSC机制) ---
        # 为每一个监督层级创建一对输出头。
        self.classifiers = nn.ModuleDict()
        for i in self.supervision_indices:
            # “诊断头”，用于输出MI诊断的概率（logits）。
            self.classifiers[f'head_{i}'] = ClsMLP(embed_dim, 1)
            # “置信度头”，用于输出模型对当前诊断的自信程度（0-1之间）。
            self.classifiers[f'confidence_head_{i}'] = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Sigmoid()
            )

    @property
    def device(self):
        """一个辅助属性，方便获取模型所在的设备（CPU/GPU）。"""
        return self.belief_token.device

    def forward(self, a2c_video, a4c_video, return_attention=False, return_features=False):
        bs = a2c_video.shape[0]  # 获取批次大小

        # --- 步骤 1: 提取视觉证据 ---
        # 将输入的A2C和A4C视频转换为高维特征向量。
        # [16, 512] <-- [16, 3, 16, 224, 224]
        a2c_feat = self.video_encoder(a2c_video)    # [B, D]
        a4c_feat = self.video_encoder(a4c_video)    # [B, D]

        # --- 步骤 2: (Granular ETKA) 精细化知识激活 ---
        # a. 将所有知识提示编码为特征向量。
        with torch.no_grad():  # 激活过程不参与梯度计算，因为它是一个引导机制。
            text_features_all = self.text_encoder(self.prompts_all).expand(bs, -1, -1) # [B, N, D]

        # b. 逐一计算视觉特征与知识特征的余弦相似度。
        #    由于EchoPrime是CLIP架构，视觉和文本特征在同一语义空间，相似度可以直接衡量其相关性。
        sim_a2c = F.cosine_similarity(a2c_feat.unsqueeze(1), text_features_all, dim=-1)  # -> [B, N]
        sim_a4c = F.cosine_similarity(a4c_feat.unsqueeze(1), text_features_all, dim=-1)  # -> [B, N]

        # c. 为每条知识生成最终的激活权重。取两个视图中更强的激活信号，并确保非负。
        activation_weights = torch.max(sim_a2c, sim_a4c).clamp(min=0)  # -> [B, N]

        # --- 步骤 3: 准备Transformer的输入序列 ---
        # 扩展 belief_token 以匹配批次大小。
        belief_tokens = self.belief_token.expand(bs, -1, -1)
        # 构建证据序列：[信念, A2C证据, A4C证据]。
        evidence_sequence = torch.cat([belief_tokens, a2c_feat.unsqueeze(1), a4c_feat.unsqueeze(1)], dim=1)
        evidence_sequence += self.positional_embedding_evidence

        # 构建知识序列：将激活权重乘到知识特征上，实现“聚焦”。
        knowledge_sequence = text_features_all * activation_weights.unsqueeze(-1)  # [B, N, D] * [B, N, 1]
        knowledge_sequence += self.positional_embedding_knowledge

        # --- 步骤 4: 通过Transformer进行迭代推理 ---
        # 初始化用于存储各层输出的容器。
        logits_dict = {}
        confidence_dict = {}
        all_self_attns = []
        all_cross_attns = []
        final_belief_state = None  # 用于存储最终的特征向量。

        # 逐层通过KnowledgeGuidedLayer进行推理。
        for i, layer in enumerate(self.transformer_layers):
            evidence_sequence, self_attn, cross_attn = layer(evidence_sequence, knowledge_sequence,
                                                             need_weights=return_attention)
            # 如果需要可视化，则保存注意力权重。
            if return_attention:
                all_self_attns.append(self_attn)
                all_cross_attns.append(cross_attn)

            # 在指定的监督层级，进行一次“诊断”和“自信度评估”。
            if (i + 1) in self.supervision_indices:
                layer_idx = i + 1
                key = 'final' if layer_idx == max(self.supervision_indices) else f'aux_{layer_idx}'
                belief_state = evidence_sequence[:, 0, :]

                # 使用对应的头计算logits和置信度。
                logits_dict[key] = self.classifiers[f'head_{layer_idx}'](belief_state)
                confidence_dict[key] = self.classifiers[f'confidence_head_{layer_idx}'](belief_state)

                # 保存最后一层的信念状态作为最终的特征表示。
                if key == 'final':
                    final_belief_state = belief_state

        # --- 步骤 5: 组织并返回输出 ---
        # 将所有输出打包到一个字典中，方便System模块解析。
        return_payload = {
            "logits": logits_dict,
            "confidence": confidence_dict,
            "knowledge_activations": activation_weights  # [B, 36]
        }

        if return_attention:
            return_payload["self_attentions"] = all_self_attns
            return_payload["cross_attentions"] = all_cross_attns

        if return_features:
            return_payload["features"] = final_belief_state

        return return_payload

