import torch
import hydra
import torchmetrics
from .base_system import BaseSystem


class MultiTaskSystem(BaseSystem):
    """多任务学习模型的训练系统"""

    def __init__(self, model_cfg, learning_rate=1e-5, unfreeze_epoch=80):
        super().__init__(learning_rate=learning_rate)
        self.save_hyperparameters(ignore=['model_cfg'])
        self.model = model_cfg

        # 为验证和测试分别创建指标
        self.val_metrics_view = self.val_metrics.clone(prefix='val_view_')
        self.val_metrics_mi = self.val_metrics.clone(prefix='val_mi_')

        # --- 新增：为测试集创建指标 ---
        self.test_metrics_view = self.test_metrics.clone(prefix='test_view_')
        self.test_metrics_mi = self.test_metrics.clone(prefix='test_mi_')

    def on_train_epoch_start(self):
        """在特定epoch解冻模型的一部分"""
        if self.current_epoch == self.hparams.unfreeze_epoch:
            print(f"Epoch {self.current_epoch}: Unfreezing MI task modules.")
            self.model.unfreeze_mi_task_modules()
            self.trainer.strategy.setup_optimizers(self.trainer)

    def forward(self, video1, video2, return_features=False):
        return self.model(video1, video2, return_features)

    def training_step(self, batch, batch_idx):
        (video1, video2), labels, _ = batch  # 忽略 sample_ids
        view_labels = labels["view_label"]
        mi_labels = labels["mi_label"]

        logit_mi, logit_view1, logit_view2 = self.model(video1, video2)

        loss_view1 = self.bce_loss(logit_view1.squeeze(), view_labels[:, 0].float())
        loss_view2 = self.bce_loss(logit_view2.squeeze(), view_labels[:, 1].float())
        loss_view = (loss_view1 + loss_view2) / 2.0

        if self.current_epoch >= self.hparams.unfreeze_epoch:
            loss_mi = self.bce_loss(logit_mi.squeeze(), mi_labels.float())
            loss = loss_view + 2 * loss_mi
            self.log("training_loss_mi", loss_mi, on_step=False, on_epoch=True)
        else:
            loss = loss_view

        self.log("training_loss", loss, on_step=True, on_epoch=True)
        self.log("training_loss_view", loss_view, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (video1, video2), labels, _ = batch  # 忽略 sample_ids
        view_labels = labels["view_label"]
        mi_labels = labels["mi_label"]

        logit_mi, logit_view1, logit_view2 = self.model(video1, video2)

        preds_view1 = torch.sigmoid(logit_view1.squeeze())
        preds_view2 = torch.sigmoid(logit_view2.squeeze())

        if len(torch.unique(view_labels)) > 1:
            self.val_metrics_view.update(torch.cat([preds_view1, preds_view2]),
                                         torch.cat([view_labels[:, 0], view_labels[:, 1]]))

        if self.current_epoch >= self.hparams.unfreeze_epoch:
            preds_mi = torch.sigmoid(logit_mi.squeeze())
            if len(torch.unique(mi_labels)) > 1:
                self.val_metrics_mi.update(preds_mi, mi_labels)

    def on_validation_epoch_end(self):
        view_metrics = self.val_metrics_view.compute()
        self.log_dict(view_metrics, prog_bar=True)
        self.val_metrics_view.reset()

        if self.current_epoch >= self.hparams.unfreeze_epoch:
            mi_metrics = self.val_metrics_mi.compute()
            self.log_dict(mi_metrics, prog_bar=True)
            self.val_metrics_mi.reset()

    def test_step(self, batch, batch_idx):
        (video1, video2), labels, sample_ids = batch
        view_labels = labels["view_label"]
        mi_labels = labels["mi_label"]

        logit_mi, logit_view1, logit_view2, features = self(video1, video2, return_features=True)

        # 更新视图分类指标
        preds_view1 = torch.sigmoid(logit_view1.squeeze())
        preds_view2 = torch.sigmoid(logit_view2.squeeze())
        if len(torch.unique(view_labels)) > 1:
            self.test_metrics_view.update(torch.cat([preds_view1, preds_view2]),
                                          torch.cat([view_labels[:, 0], view_labels[:, 1]]))

        # 更新MI诊断指标 (在测试时，我们总是评估所有任务)
        preds_mi = torch.sigmoid(logit_mi.squeeze())
        if len(torch.unique(mi_labels)) > 1:
            self.test_metrics_mi.update(preds_mi, mi_labels)

        # 返回需要在回调中保存的数据
        return {
            'sample_ids': sample_ids,
            'targets': mi_labels,
            'preds': preds_mi,
            'features': features
        }

    def on_test_epoch_end(self):
        # 计算并记录视图分类的测试指标
        view_metrics = self.test_metrics_view.compute()
        self.log_dict(view_metrics)
        self.test_metrics_view.reset()

        # 计算并记录MI诊断的测试指标
        mi_metrics = self.test_metrics_mi.compute()
        self.log_dict(mi_metrics)
        self.test_metrics_mi.reset()