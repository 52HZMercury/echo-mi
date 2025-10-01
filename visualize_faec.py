# visualize_faec.py

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb

# 导入新的可视化工具
from src.utils.visualizations import (
    plot_belief_evolution_simple,
    plot_self_attention_flow,
    plot_knowledge_attribution_heatmap  # 替换旧的函数
)
from src.systems.faec_system import FAECSystem
from src.utils.prompts import segment_prompts


@hydra.main(config_path="configs", config_name="visualize_faec", version_base=None)
def visualize(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    wandb_project_name = f"{cfg.wandb.project}_KnowledgeAttribution"
    wandb.init(project=wandb_project_name, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True))

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    system = FAECSystem.load_from_checkpoint(cfg.checkpoint_path)
    device = 'cuda' if cfg.trainer.accelerator == 'gpu' else 'cpu'
    system.to(device).eval()
    model = system.model

    # 这里的prompts将作为Y轴标签传入
    prompts = segment_prompts["A4C"]

    for i, batch in enumerate(tqdm(test_loader, desc="Generating Reasoning Dashboards")):
        if i >= cfg.num_samples_to_viz:
            break

        (a2c, a4c), labels = batch
        a2c, a4c = a2c.to(device), a4c.to(device)
        target = labels["mi_label"].item()

        with torch.no_grad():
            logits_dict, self_attns, cross_attns = model(a2c, a4c, prompts, return_attention=True)

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1.5])

        ax_belief = fig.add_subplot(gs[0, 0])
        ax_self_attn = fig.add_subplot(gs[0, 1])
        ax_cross_attn = fig.add_subplot(gs[1, :])  # 热力图需要更多垂直空间

        plot_belief_evolution_simple(ax_belief, {k: v[0] for k, v in logits_dict.items()}, target,
                                     model.supervision_indices)
        plot_self_attention_flow(ax_self_attn, [sa[0] for sa in self_attns])

        # **核心修改: 调用新的热力图函数**
        plot_knowledge_attribution_heatmap(ax_cross_attn, [ca[0] for ca in cross_attns], prompts)

        fig.suptitle(f"Diagnostic Reasoning Dashboard - Sample {i}", fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        final_prob = torch.sigmoid(logits_dict['final'][0]).item()
        wandb.log({f"Reasoning_Dashboard/Sample_{i}": wandb.Image(fig,
                                                                  caption=f"GT: {target}, Final Pred: {final_prob:.2f}")})
        plt.close(fig)

    wandb.finish()
    print("\nVisualization script finished successfully!")


if __name__ == "__main__":
    visualize()
