import torch
import torch.nn as nn
import pytorch_lightning as pl

class ClsMLP(pl.LightningModule):
    """一个简单的两层MLP分类器"""
    def __init__(self, input_size, output_size, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiTaskViewClsMLP(pl.LightningModule):
    """用于多任务学习中视图分类的MLP"""
    def __init__(self, input_size, output_size, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_for_mi = self.relu(self.fc1(x))
        x_out = self.dropout1(x_for_mi)
        x_out = self.fc2(x_out)
        return x_out, x_for_mi

class MultiTaskMIClsMLP(pl.LightningModule):
    """用于多任务学习中心梗诊断的MLP，带有残差连接"""
    def __init__(self, input_size, output_size, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        # 输入维度是融合后的特征(256)和来自视图分类分支的特征(256)
        self.fc2 = nn.Linear(256 + 256, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x_fused, x_from_view):
        x = self.relu(self.fc1(x_fused))
        # 特征拼接和残差连接
        x_combined = torch.cat((x, x_from_view), dim=1)
        x = self.dropout1(x_combined)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
