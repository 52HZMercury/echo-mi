import torch
from .base_system import BaseSystem
from src.utils.visualizations import log_trace_to_wandb
from src.utils.prompts import segment_prompts  # 导入prompts


class GEFSystem(BaseSystem):
    """生成式证据流模型的训练系统."""

    def __init__(self, model_cfg, learning_rate=1e-4, target_logit=3.0,
                 vis_num_samples=8, vis_ode_steps=25, **kwargs):
        # 移除了重复的learning_rate=learning_rate
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.save_hyperparameters(ignore=['model_cfg'])
        self.model = model_cfg

        # 保存一个固定的验证批次用于可视化
        self.vis_batch = None
        # 这里的prompts是固定的，对于所有样本都一样
        self.prompts = segment_prompts["A4C"]

    def training_step(self, batch, batch_idx):
        # --- **核心修正点 1** ---
        # 采用与您原始代码一致的解包方式
        (a2c, a4c), labels, _ = batch  # 忽略 sample_ids
        targets = labels["mi_label"]  # 从标签字典中获取心梗标签
        # --- 修正结束 ---

        context = self.model.get_context(a2c, a4c, self.prompts)

        b_0 = torch.zeros_like(targets, dtype=torch.float, device=self.device)
        b_1 = torch.where(targets == 1, self.hparams.target_logit, -self.hparams.target_logit)

        t = torch.rand_like(targets, dtype=torch.float, device=self.device)
        b_t = t * b_1
        v_target = b_1

        v_pred = self.model(context, b_t, t).squeeze()
        loss = torch.nn.functional.mse_loss(v_pred, v_target)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # --- **核心修正点 2** ---
        (a2c, a4c), labels, _ = batch  # 忽略 sample_ids
        targets = labels["mi_label"]
        # --- 修正结束 ---

        context = self.model.get_context(a2c, a4c, self.prompts)

        # 使用ODE求解器得到最终预测
        b_final, _, _, _ = self.model.solve_ode(context, self.hparams.vis_ode_steps)

        loss = self.bce_loss(b_final.squeeze(), targets.float())
        preds = torch.sigmoid(b_final.squeeze())

        self.val_metrics.update(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        if batch_idx == 0 and self.vis_batch is None:
            # 保存整个batch用于可视化
            self.vis_batch = (
                a2c[:self.hparams.vis_num_samples],
                a4c[:self.hparams.vis_num_samples],
                targets[:self.hparams.vis_num_samples]
            )

    def on_validation_epoch_end(self):
        # 调用基类方法来计算和记录指标
        super().on_validation_epoch_end()

        if self.vis_batch and self.trainer.is_global_zero:
            a2c, a4c, targets = self.vis_batch
            # 确保数据在正确的设备上
            a2c_dev = a2c.to(self.device)
            a4c_dev = a4c.to(self.device)
            context = self.model.get_context(a2c_dev, a4c_dev, self.prompts)

            _, trace, _, _ = self.model.solve_ode(context, self.hparams.vis_ode_steps)

            log_trace_to_wandb(trace, targets, self.trainer)
            self.vis_batch = None  # 清空，以便下一个epoch重新获取

    def test_step(self, batch, batch_idx):
        (a2c, a4c), labels, sample_ids = batch
        targets = labels["mi_label"]

        context = self.model.get_context(a2c, a4c, self.prompts)
        b_final, _, _, features = self.model.solve_ode(context, self.hparams.vis_ode_steps * 2)  # 测试时更精细

        loss = self.bce_loss(b_final.squeeze(), targets.float())
        preds = torch.sigmoid(b_final.squeeze())

        self.test_metrics.update(preds, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # 返回需要在回调中保存的数据
        return {
            'sample_ids': sample_ids,
            'targets': targets,
            'preds': preds,
            'features': features
        }