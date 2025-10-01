import torch.nn as nn


class BaseModel(nn.Module):
    """
    所有模型的基类.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
