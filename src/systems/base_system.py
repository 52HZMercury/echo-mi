import torch
import madgrad
import torchmetrics
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

class BaseSystem(pl.LightningModule):
    """
    训练系统的基类.
    包含通用的验证逻辑和优化器配置.
    """

    # def __init__(self, learning_rate=1e-5, weight_decay=1e-2, **kwargs):
    #     super().__init__()
    #     self.save_hyperparameters()
    #     self.learning_rate = learning_rate
    #     self.weight_decay = weight_decay
    #     self.bce_loss = torch.nn.BCEWithLogitsLoss()
    #
    #     # Metrics
    #     metrics = torchmetrics.MetricCollection({
    #         'accuracy': torchmetrics.Accuracy(task="binary"),
    #         'f1_score': torchmetrics.F1Score(task="binary"),
    #         'specificity': torchmetrics.Specificity(task="binary"),
    #         'precision': torchmetrics.Precision(task="binary"),
    #         'recall': torchmetrics.Recall(task="binary"),
    #         'auroc': torchmetrics.AUROC(task="binary")
    #     })
    #     self.val_metrics = metrics.clone(prefix='val_')
    #     self.test_metrics = metrics.clone(prefix='test_')
    def __init__(self, learning_rate=1e-5, weight_decay=1e-2, num_classes=2, **kwargs):
        super().__init__()

        # 2. 将所有参数手动解析为标准的 Python 类型 (dict/list)
        # resolve=True 会解析所有的引用（例如我们在 YAML 里加的时间戳 ${now:...}）
        full_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_classes": num_classes,
            **kwargs
        }

        # 递归地将配置中的 DictConfig 转换为普通 dict
        clean_hparams = OmegaConf.to_container(
            OmegaConf.create(full_config),
            resolve=True
        )

        # 3. 显式保存处理后的字典
        self.save_hyperparameters(clean_hparams)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes  # 添加类别数属性

        # 根据类别数确定任务类型
        task_type = "binary" if num_classes == 2 else "multiclass"

        # Metrics
        metrics_config = {
            'accuracy': torchmetrics.Accuracy(task=task_type, num_classes=num_classes if task_type == "multiclass" else None),
            'f1_score': torchmetrics.F1Score(task=task_type, num_classes=num_classes if task_type == "multiclass" else None),
            'precision': torchmetrics.Precision(task=task_type, num_classes=num_classes if task_type == "multiclass" else None),
            'recall': torchmetrics.Recall(task=task_type, num_classes=num_classes if task_type == "multiclass" else None),
            'auroc': torchmetrics.AUROC(task=task_type, num_classes=num_classes if task_type == "multiclass" else None)
        }

        # 二分类特有指标
        if task_type == "binary":
            metrics_config['specificity'] = torchmetrics.Specificity(task="binary")

        metrics = torchmetrics.MetricCollection(metrics_config)
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def configure_optimizers(self):
        optimizer = madgrad.MADGRAD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate
        )

        # MultiStepLR
        # 默认
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[20, 40],
            gamma=0.1
        )

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
