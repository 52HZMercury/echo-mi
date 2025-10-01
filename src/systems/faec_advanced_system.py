# src/systems/faec_advanced_system.py

import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from .base_system import BaseSystem
from src.utils.prompts import FAEC_KNOWLEDGE_BASE
from src.utils.visualizations_advanced import (
    plot_knowledge_activation_chart,
    plot_belief_confidence_dual_track,
    plot_knowledge_query_chart,
    plot_knowledge_attribution_heatmap
)

class FAECAdvancedSystem(BaseSystem):
    # __init__, training_step, validation_step, on_validation_epoch_end 保持不变...
    def __init__(self, model_cfg, learning_rate=1e-5, supervision_loss_weight=0.5,
                 correctness_gate_beta=1.0,
                 num_samples_to_viz=1, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.save_hyperparameters(ignore=['model_cfg'])
        self.model = model_cfg
        self.prompts = FAEC_KNOWLEDGE_BASE
        self.vis_batch = None
        self.num_samples_to_viz = num_samples_to_viz
        self.bce_loss_no_reduction = torch.nn.BCEWithLogitsLoss(reduction='none')

    def training_step(self, batch, batch_idx):
        (a2c, a4c), labels, _ = batch
        targets = labels["mi_label"]
        outputs = self.model(a2c, a4c)
        logits_dict = outputs["logits"]
        confidence_dict = outputs["confidence"]
        total_loss = 0.0
        running_unconfidence = 1.0
        sorted_indices = sorted(self.model.supervision_indices)
        previous_confidence = None
        for i, layer_idx in enumerate(sorted_indices):
            is_final_layer = (layer_idx == max(sorted_indices))
            key = 'final' if is_final_layer else f'aux_{layer_idx}'
            logits = logits_dict[key].squeeze(-1)
            confidence = confidence_dict[key].squeeze(-1)
            layer_loss = self.bce_loss_no_reduction(logits, targets.float())
            weighted_bce_loss = running_unconfidence * layer_loss
            total_loss += weighted_bce_loss.mean()
            if previous_confidence is not None:
                correctness_gate = torch.exp(-self.hparams.correctness_gate_beta * layer_loss.detach())
                confidence_gain = confidence - previous_confidence.detach()
                incentive_loss = - (1 - previous_confidence.detach()) * confidence_gain * correctness_gate
                total_loss += 0.1 * incentive_loss.mean()
            previous_confidence = confidence
            running_unconfidence = running_unconfidence * (1 - confidence.detach())
        confidence_regularization = running_unconfidence.mean()
        total_loss += self.hparams.supervision_loss_weight * confidence_regularization
        self.log("train/loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/final_confidence", confidence_dict['final'].mean(), on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        (a2c, a4c), labels, sample_ids = batch
        targets = labels["mi_label"]
        outputs = self.model(a2c, a4c)
        logits = outputs["logits"]['final'].squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        preds = torch.sigmoid(logits)
        self.val_metrics.update(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        if batch_idx == 0 and self.vis_batch is None:
            self.vis_batch = (a2c, a4c, targets, sample_ids)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.vis_batch and self.trainer.is_global_zero:
            a2c, a4c, targets, sample_ids = self.vis_batch
            num_to_log = min(self.num_samples_to_viz, a2c.shape[0])

            with torch.no_grad():
                outputs = self.model(a2c[:num_to_log], a4c[:num_to_log], return_attention=True)

            supervision_indices = self.model.supervision_indices
            num_supervised_layers = len(supervision_indices)
            num_prompts = self.model.num_knowledge_prompts

            if num_supervised_layers != 3:
                print(f"Skipping dashboard visualization as it's designed for 3 supervised layers, but found {num_supervised_layers}.")
                return

            for i in range(num_to_log):
                # --- 与独立脚本完全一致的 1-4-1 布局逻辑 ---
                fig = plt.figure(figsize=(28, 30))  # 调整画布尺寸以获得最佳比例
                # 使用3行8列的网格系统，并微调高度比例
                gs = fig.add_gridspec(3, 9, height_ratios=[0.8, 1, num_prompts * 0.08], hspace=0.4, wspace=0.8)

                # ** Row 1 **
                ax_belief_evo = fig.add_subplot(gs[0, 1:7])

                sample_activations = outputs['knowledge_activations'][i]
                sample_logits = {key: val[i] for key, val in outputs['logits'].items()}
                sample_confidence = {key: val[i] for key, val in outputs['confidence'].items()}
                all_cross_attns = [ca[i] for ca in outputs['cross_attentions']]
                target_label = targets[i].item()
                current_sample_id = sample_ids[i]

                plot_belief_confidence_dual_track(ax_belief_evo, sample_logits, sample_confidence, target_label, supervision_indices)

                # ** Row 2 **
                ax_activation = fig.add_subplot(gs[1, 1:3], polar=True)
                ax_q1 = fig.add_subplot(gs[1, 3:5], polar=True)
                ax_q2 = fig.add_subplot(gs[1, 5:7], polar=True)
                ax_q3 = fig.add_subplot(gs[1, 7:9], polar=True)

                plot_knowledge_activation_chart(ax_activation, sample_activations)

                query_axes = [ax_q1, ax_q2, ax_q3]
                for j, layer_idx in enumerate(supervision_indices):
                    attention_for_layer = all_cross_attns[layer_idx - 1]
                    plot_knowledge_query_chart(query_axes[j], attention_for_layer, layer_idx)

                # ** Row 3 **
                ax_heatmap = fig.add_subplot(gs[2, 1:7])
                supervised_cross_attns = [all_cross_attns[idx - 1] for idx in supervision_indices]
                plot_knowledge_attribution_heatmap(ax_heatmap, supervised_cross_attns, self.model.prompts_all, supervision_indices)

                final_prob = torch.sigmoid(outputs['logits']['final'][i]).item()
                fig.suptitle(f"Diagnostic Reasoning Dashboard - Sample ID: {current_sample_id}\n"
                             f"GT: {'MI' if target_label==1 else 'Normal'}, Pred: {final_prob:.2f}",
                             weight='bold')
                fig.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=4)

                # self.logger.experiment.log({
                #     f"Val_Dashboard/Sample_{current_sample_id}": wandb.Image(fig,
                #         caption=f"ID: {current_sample_id}, GT: {target_label}, Pred: {final_prob:.2f}")
                # })
                # plt.close(fig)

                # 修改日志记录方式，使其兼容不同类型的记录器
                try:
                    # 尝试使用WandbLogger的方式
                    self.logger.experiment.log({
                        f"Val_Dashboard/Sample_{current_sample_id}": wandb.Image(fig,
                                                                                 caption=f"ID: {current_sample_id}, GT: {target_label}, Pred: {final_prob:.2f}")
                    })
                except AttributeError:
                    try:
                        # 如果是TensorBoardLogger，使用add_figure
                        self.logger.experiment.add_figure(
                            f"Val_Dashboard/Sample_{current_sample_id}",
                            fig,
                            global_step=self.current_epoch,
                            close=True
                        )
                    except AttributeError:
                        # 如果是CSVLogger或其他不支持图像的日志记录器，则只保存到文件系统
                        import os
                        save_dir = getattr(self.logger, 'save_dir', '.')
                        version = getattr(self.logger, 'version', 'default')
                        log_dir = os.path.join(save_dir, 'lightning_logs', str(version), 'val_dashboard')
                        os.makedirs(log_dir, exist_ok=True)
                        fig.savefig(os.path.join(log_dir, f"sample_{current_sample_id}_epoch_{self.current_epoch}.png"),
                                    dpi=150, bbox_inches='tight')
                        plt.close(fig)

                plt.close(fig)

            self.vis_batch = None

    def test_step(self, batch, batch_idx):
        (a2c, a4c), labels, sample_ids = batch
        targets = labels["mi_label"]
        outputs = self.model(a2c, a4c, return_attention=False, return_features=True)
        logits = outputs["logits"]["final"].squeeze(-1)
        features = outputs.get("features", None)
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        preds = torch.sigmoid(logits)
        self.test_metrics.update(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {'sample_ids': sample_ids, 'targets': targets, 'preds': preds, 'features': features}