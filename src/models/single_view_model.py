from .base_model import BaseModel
from .components.encoders import VideoEncoder
from .components.mlps import ClsMLP


class SingleViewModel(BaseModel):
    """
    单视图模型.
    """

    def __init__(self, video_encoder_path, frozen_encoder=True):
        super().__init__()
        self.encoder = VideoEncoder(video_encoder_path, frozen=frozen_encoder)
        self.classifier = ClsMLP(512, 1)

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits