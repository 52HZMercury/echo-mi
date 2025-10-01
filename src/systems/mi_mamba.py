import torch
import torchmetrics
import hydra
from .base_system import BaseSystem


class MIMambaSystem(BaseSystem):
    """单视图模型的训练系统"""

    def __init__(self, model_cfg, learning_rate=1e-5):
        super().__init__(learning_rate=learning_rate)
        # 保存超参数，同时忽略已经实例化的模型对象，避免冗余保存
        self.save_hyperparameters(ignore=['model_cfg'])
        # 直接接收Hydra已经为我们实例化好的模型对象
        self.model = model_cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, targets, _ = batch
        logits = self(x).squeeze()
        loss = self.bce_loss(logits, targets.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 验证步骤现在也需要处理样本ID
        x, targets, _ = batch
        logits = self(x).squeeze()
        loss = self.bce_loss(logits, targets.float())

        preds = torch.sigmoid(logits)
        self.val_metrics.update(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, targets, sample_ids = batch
        # logits, features = self.model(x, return_features=True)
        logits = self.model(x, return_features=True)
        logits = logits.squeeze()

        loss = self.bce_loss(logits, targets.float())
        preds = torch.sigmoid(logits)

        self.test_metrics.update(preds, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # 返回需要在回调中保存的数据
        return {
            'sample_ids': sample_ids,
            'targets': targets,
            'preds': preds,
            # 'features': features
        }