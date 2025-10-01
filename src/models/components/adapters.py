import torch.nn as nn

class ViewAdapter(nn.Module):
    """
    用于视频特征的Adapter模块.
    通过一个瓶颈结构对特征进行非线性变换，并使用残差连接。
    """
    def __init__(self, input_dim, adapter_dim):
        super(ViewAdapter, self).__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual

class TextAdapter(nn.Module):
    """
    用于文本特征的Adapter模块.
    """
    def __init__(self, input_dim, adapter_dim):
        super(TextAdapter, self).__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual
