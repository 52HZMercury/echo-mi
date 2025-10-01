# src/systems/faec_system.py
import torch
import wandb
import matplotlib.pyplot as plt
from .base_system import BaseSystem
from src.utils.prompts import segment_prompts
# ** 核心修正点 1: 导入新的、全套的仪表盘可视化函数 **
from src.utils.visualizations import (
    plot_belief_evolution_simple,
    plot_self_attention_flow,
    plot_knowledge_attribution_heatmap  # 替换旧的函数
)


class FAECSystem(BaseSystem):
    """
    FAECModel的训练系统。
    负责处理多头输出、计算加权总损失，并在验证时调用全新的“知识归因”仪表盘。
    """

    def __init__(self, model_cfg, learning_rate=1e-5, aux_loss_weight=0.4,
                 num_samples_to_viz=1, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.save_hyperparameters(ignore=['model_cfg'])
        self.model = model_cfg
        self.prompts = segment_prompts["A4C"]  # 文本知识列表
        self.vis_batch = None
        self.num_samples_to_viz = num_samples_to_viz

    def training_step(self, batch, batch_idx):
        (a2c, a4c), labels, _ = batch  # 忽略 sample_ids
        targets = labels["mi_label"]
        logits_dict = self.model(a2c, a4c, self.prompts, return_attention=False)

        final_logits = logits_dict["final"].squeeze(-1)
        loss_final = self.bce_loss(final_logits, targets.float())
        total_loss = loss_final
        self.log("train/loss_final", loss_final, on_step=False, on_epoch=True)

        aux_losses = []
        for key, logits in logits_dict.items():
            if "aux" in key:
                aux_logits = logits.squeeze(-1)
                aux_loss = self.bce_loss(aux_logits, targets.float())
                self.log(f"train/{key}_loss", aux_loss, on_step=False, on_epoch=True)
                aux_losses.append(aux_loss)

        if aux_losses:
            total_aux_loss = torch.stack(aux_losses).mean()
            total_loss += self.hparams.aux_loss_weight * total_aux_loss
            self.log("train/loss_aux", total_aux_loss, on_step=False, on_epoch=True)

        self.log("train/loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        (a2c, a4c), labels, _ = batch  # 忽略 sample_ids
        targets = labels["mi_label"]
        logits_dict = self.model(a2c, a4c, self.prompts, return_attention=False)
        logits = logits_dict["final"].squeeze(-1)

        loss = self.bce_loss(logits, targets.float())
        preds = torch.sigmoid(logits)

        self.val_metrics.update(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        if batch_idx == 0 and self.vis_batch is None:
            self.vis_batch = (a2c, a4c, targets)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        if self.vis_batch and self.trainer.is_global_zero:
            a2c, a4c, targets = self.vis_batch
            num_to_log = min(self.num_samples_to_viz, a2c.shape[0])

            with torch.no_grad():
                logits_dict, self_attns, cross_attns = self.model(
                    a2c[:num_to_log],
                    a4c[:num_to_log],
                    self.prompts,
                    return_attention=True
                )

            for i in range(num_to_log):
                fig = plt.figure(figsize=(16, 12))
                gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1.5])
                ax_belief = fig.add_subplot(gs[0, 0])
                ax_self_attn = fig.add_subplot(gs[0, 1])
                ax_cross_attn = fig.add_subplot(gs[1, :])

                sample_logits = {key: val[i] for key, val in logits_dict.items()}
                sample_self_attns = [sa[i] for sa in self_attns]
                sample_cross_attns = [ca[i] for ca in cross_attns]
                target_label = targets[i].item()

                plot_belief_evolution_simple(ax_belief, sample_logits, target_label, self.model.supervision_indices)
                plot_self_attention_flow(ax_self_attn, sample_self_attns)

                # ** 核心修正点 2: 调用新的热力图函数 **
                plot_knowledge_attribution_heatmap(ax_cross_attn, sample_cross_attns, self.prompts)

                fig.suptitle(f"Diagnostic Reasoning Dashboard - Sample {i} (Epoch {self.trainer.current_epoch})",
                             weight='bold')
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])

                final_prob = torch.sigmoid(logits_dict['final'][i]).item()
                self.logger.experiment.log({
                    f"Val_Dashboard/Sample_{i}": wandb.Image(fig,
                                                             caption=f"GT: {target_label}, Final Pred: {final_prob:.2f}")
                })
                plt.close(fig)

            self.vis_batch = None

    def test_step(self, batch, batch_idx):
        (a2c, a4c), labels, sample_ids = batch
        targets = labels["mi_label"]
        logits_dict, features = self.model(a2c, a4c, self.prompts, return_features=True)
        logits = logits_dict["final"].squeeze(-1)

        loss = self.bce_loss(logits, targets.float())
        preds = torch.sigmoid(logits)

        self.test_metrics.update(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        # 返回需要在回调中保存的数据
        return {
            'sample_ids': sample_ids,
            'targets': targets,
            'preds': preds,
            'features': features
        }