# visualize_faec_advanced.py

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from src.utils.visualizations_advanced import (
    plot_knowledge_activation_chart,
    plot_belief_confidence_dual_track,
    plot_knowledge_query_chart,
    plot_knowledge_attribution_heatmap
)
from src.systems.faec_advanced_system import FAECAdvancedSystem


@hydra.main(config_path="configs", config_name="visualize_faec_advanced", version_base=None)
def visualize(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    OmegaConf.update(cfg, "data.batch_size", 1, merge=True)
    output_dir = "outputs/faec_advanced_dashboards"
    os.makedirs(output_dir, exist_ok=True)
    print(f"--> PDF dashboards will be saved to: {output_dir}")

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    device = 'cuda' if cfg.trainer.accelerator == 'gpu' else 'cpu'
    system = FAECAdvancedSystem.load_from_checkpoint(cfg.checkpoint_path, map_location=device)
    system.eval()
    model = system.model

    supervision_indices = model.supervision_indices
    num_supervised_layers = len(supervision_indices)
    num_prompts = model.num_knowledge_prompts
    assert num_supervised_layers == 3, f"This layout is designed for 3 supervised layers, but found {num_supervised_layers}."

    process_all_samples = (cfg.num_samples_to_viz == -1)
    num_to_process = len(test_loader) if process_all_samples else cfg.num_samples_to_viz
    print(f"--> Generating {num_to_process} reasoning dashboards...")
    progress_bar = tqdm(enumerate(test_loader), total=num_to_process, desc="Generating Dashboards")

    for i, batch in progress_bar:
        if not process_all_samples and i >= num_to_process:
            break

        (a2c, a4c), labels, sample_ids = batch
        a2c, a4c = a2c.to(device), a4c.to(device)
        target = labels["mi_label"][0].item()
        current_sample_id = sample_ids[0]

        with torch.no_grad():
            outputs = model(a2c, a4c, return_attention=True)

        sample_activations = outputs['knowledge_activations'][0]
        sample_logits = {key: val[0] for key, val in outputs['logits'].items()}
        sample_confidence = {key: val[0] for key, val in outputs['confidence'].items()}
        all_cross_attns = [ca[0] for ca in outputs['cross_attentions']]

        # --- 核心修改点: 全新的、为CVPR优化的 1-4-1 最终布局 ---
        fig = plt.figure(figsize=(28, 30))  # 调整画布尺寸以获得最佳比例
        # 使用3行8列的网格系统，并微调高度比例
        gs = fig.add_gridspec(3, 9, height_ratios=[0.8, 1, num_prompts * 0.08], hspace=0.4, wspace=0.8)

        # ** Row 1: Belief & Confidence Evolution (1 plot) **
        ax_belief_evo = fig.add_subplot(gs[0, 3:7])  # 居中，但不占满全宽
        plot_belief_confidence_dual_track(ax_belief_evo, sample_logits, sample_confidence, target, supervision_indices)

        # ** Row 2: All Radar Plots (4 plots) **
        ax_activation = fig.add_subplot(gs[1, 1:3], polar=True)
        ax_q1 = fig.add_subplot(gs[1, 3:5], polar=True)
        ax_q2 = fig.add_subplot(gs[1, 5:7], polar=True)
        ax_q3 = fig.add_subplot(gs[1, 7:9], polar=True)

        plot_knowledge_activation_chart(ax_activation, sample_activations)

        query_axes = [ax_q1, ax_q2, ax_q3]
        for j, layer_idx in enumerate(supervision_indices):
            attention_for_layer = all_cross_attns[layer_idx - 1]
            plot_knowledge_query_chart(query_axes[j], attention_for_layer, layer_idx)

        # ** Row 3: Heatmap Summary (1 plot) **
        ax_heatmap = fig.add_subplot(gs[2, 1:7])
        supervised_cross_attns = [all_cross_attns[idx - 1] for idx in supervision_indices]
        plot_knowledge_attribution_heatmap(ax_heatmap, supervised_cross_attns, model.prompts_all, supervision_indices)

        final_prob = torch.sigmoid(outputs['logits']['final'][0]).item()
        fig.suptitle(f"Diagnostic Reasoning Dashboard - Sample ID: {current_sample_id}\n"
                     f"GT: {'MI' if target == 1 else 'Normal'}, Final Prediction: {final_prob:.2f}",
                     weight='bold')

        pdf_save_path = os.path.join(output_dir, f"dashboard_sample_{current_sample_id}.pdf")
        plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

    print(f"\n--> All {i + 1} PDF dashboards have been successfully saved to '{output_dir}'.")


if __name__ == "__main__":
    visualize()
