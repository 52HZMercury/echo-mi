import torch
import madgrad
import torchmetrics
import pytorch_lightning as pl


class BaseSystem(pl.LightningModule):
    """
    训练系统的基类.
    包含通用的验证逻辑和优化器配置.
    """

    def __init__(self, learning_rate=1e-5,  weight_decay=1e-2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay=weight_decay
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        # Metrics
        metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="binary"),
            'f1_score': torchmetrics.F1Score(task="binary"),
            'specificity': torchmetrics.Specificity(task="binary"),
            'precision': torchmetrics.Precision(task="binary"),
            'recall': torchmetrics.Recall(task="binary"),
            'auroc': torchmetrics.AUROC(task="binary")
        })
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def configure_optimizers(self):
        optimizer = madgrad.MADGRAD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate
        )

        # optimizer = torch.optim.AdamW(
        #     filter(lambda p: p.requires_grad, self.parameters()),
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay # 应用权重衰减
        # )

        return optimizer

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        # 仅计算和记录指标，保存逻辑已移至ResultsSaver回调
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()