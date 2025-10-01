import torch
import torch.nn as nn
from .base_model import BaseModel
from .components.encoders import VideoEncoderForMultiTask, TextEncoder
from .components.mlps import MultiTaskViewClsMLP, MultiTaskMIClsMLP
from .components.adapters import ViewAdapter, TextAdapter
from .components.attention import AttnFusion, VideoTextCrossAttention


class CVFMultiTaskModel(BaseModel):
    """
    用于切面分类和心梗诊断的多任务模型 (CVF: Classification-guided View Fusion)
    """

    def __init__(self, video_encoder_path, frozen_encoder=True, adapter_dim=256, attn_layers=4):
        super().__init__()
        self.video_encoder = VideoEncoderForMultiTask(video_encoder_path, frozen=frozen_encoder)

        self.a2c_adapter = ViewAdapter(512, adapter_dim)
        self.a4c_adapter = ViewAdapter(512, adapter_dim)
        self.attn_fusion = AttnFusion(layers=attn_layers)
        self.view_mlp = MultiTaskViewClsMLP(512, 1)
        self.mi_mlp = MultiTaskMIClsMLP(512, 1)

        self.freeze_mi_task_modules()

    def freeze_mi_task_modules(self):
        """冻结用于MI诊断任务的模块参数"""
        for param in self.mi_mlp.parameters():
            param.requires_grad = False
        for param in self.attn_fusion.parameters():
            param.requires_grad = False
        for param in self.a2c_adapter.parameters():
            param.requires_grad = False
        for param in self.a4c_adapter.parameters():
            param.requires_grad = False

    def unfreeze_mi_task_modules(self):
        """解冻用于MI诊断任务的模块参数"""
        for param in self.mi_mlp.parameters():
            param.requires_grad = True
        for param in self.attn_fusion.parameters():
            param.requires_grad = True
        for param in self.a2c_adapter.parameters():
            param.requires_grad = True
        for param in self.a4c_adapter.parameters():
            param.requires_grad = True

    def forward(self, video1, video2, return_features=False):
        # 1. 特征提取
        view1_feat_view, view1_feat_mi = self.video_encoder(video1)
        view2_feat_view, view2_feat_mi = self.video_encoder(video2)

        # 2. 视图分类
        logit_view1, x1_for_mi = self.view_mlp(view1_feat_view)
        logit_view2, x2_for_mi = self.view_mlp(view2_feat_view)
        # 平均两个视图的中间特征，用于MI任务
        vx_for_mi = (x1_for_mi + x2_for_mi) / 2.0

        # 3. MI诊断 (Classification-guided Fusion)
        with torch.no_grad():  # 推理时分类结果不应有梯度
            view_pred1 = (torch.sigmoid(logit_view1) > 0.5).long()
            view_pred2 = (torch.sigmoid(logit_view2) > 0.5).long()

        # 根据分类结果选择Adapter
        # A2C: 0, A4C: 1
        adapted_feat1 = torch.where(view_pred1 == 0, self.a2c_adapter(view1_feat_mi), self.a4c_adapter(view1_feat_mi))
        adapted_feat2 = torch.where(view_pred2 == 0, self.a2c_adapter(view2_feat_mi), self.a4c_adapter(view2_feat_mi))

        # 4. 特征融合
        stacked_features = torch.stack([adapted_feat1, adapted_feat2], dim=0)
        fused_features = self.attn_fusion(stacked_features)

        # 5. MI分类
        logit_mi = self.mi_mlp(fused_features, vx_for_mi)

        if return_features:
            return logit_mi, logit_view1, logit_view2, fused_features

        return logit_mi, logit_view1, logit_view2