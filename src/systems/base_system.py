import torch
import madgrad
import torchmetrics
import pytorch_lightning as pl


class BaseSystem(pl.LightningModule):
    """
    训练系统的基类.
    包含通用的验证逻辑和优化器配置.
    """

    def __init__(self, learning_rate=1e-5, weight_decay=1e-2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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

        # optimizer = torch.optim.RMSprop(
        #     filter(lambda p: p.requires_grad, self.parameters()),
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay,
        #     alpha=0.99  # 平滑常数
        # )


        # MultiStepLR
        # 默认
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[20, 40],
            gamma=0.1
        )

        # 定义LinearLR调度器，线性增加
        # scheduler = LinearLR(
        #     optimizer,
        #     end_lr=1e-2,# 最终学习率
        #     num_iter=60, # 总迭代次数
        # )

        # 定义ExponentialLR调度器
        # scheduler = ExponentialLR(
        #     optimizer,
        #     end_lr=1e-2,  # 最终学习率
        #     num_iter=60,  # 总迭代次数
        # )

        # 定义WarmupCosineSchedule调度器
        # scheduler = WarmupCosineSchedule(
        #     optimizer,
        #     warmup_steps=5, # 线性warmup步骤
        #     t_total=60,  # 总训练步骤
        #     cycles=0.5,  # 余弦周期
        # )

        # 定义LinearWarmupCosineAnnealingLR调度器
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=5,# 热身epoch数
        #     max_epochs=60,  # 总训练epoch数
        #     warmup_start_lr=1e-6,# 热身开始的学习率
        #     eta_min=1e-6,  # 最小学习率
        # )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer, metric):
        scheduler.step()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        # 仅计算和记录指标，保存逻辑已移至ResultsSaver回调
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()
