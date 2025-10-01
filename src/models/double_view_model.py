import torch
from .base_model import BaseModel
from .components.encoders import VideoEncoder
from .components.adapters import ViewAdapter
from .components.attention import AttnFusion
from .components.mlps import ClsMLP


class DoubleViewModel(BaseModel):
    """
    使用 Adapter 和 Attention Fusion 的双视图模型.
    """

    def __init__(self, video_encoder_path, frozen_encoder=True, adapter_dim=256, attn_layers=4):
        super().__init__()
        self.encoder = VideoEncoder(video_encoder_path, frozen=frozen_encoder)
        self.a2c_adapter = ViewAdapter(512, adapter_dim)
        self.a4c_adapter = ViewAdapter(512, adapter_dim)
        self.attn_fusion = AttnFusion(embed_dim=512, layers=attn_layers)
        self.classifier = ClsMLP(512, 1)

    def forward(self, a2c_video, a4c_video, return_features=False):
        # 视频特征提取
        a2c_features = self.encoder(a2c_video)
        a4c_features = self.encoder(a4c_video)

        # 通过 Adapter
        a2c_adapted = self.a2c_adapter(a2c_features)
        a4c_adapted = self.a4c_adapter(a4c_features)

        # 特征融合
        # [2, batch_size, 512]
        stacked_features = torch.stack([a2c_adapted, a4c_adapted], dim=0)
        fused_features = self.attn_fusion(stacked_features)

        # 分类
        logits = self.classifier(fused_features)

        if return_features:
            return logits, fused_features
        return logits