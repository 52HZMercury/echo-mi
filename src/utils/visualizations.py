# src/utils/visualizations.py

import matplotlib

matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
import numpy as np
import re

plt.style.use('seaborn-v0_8-whitegrid')
FONT_SIZES = {'title': 16, 'suptitle': 18, 'label': 14, 'tick': 12, 'annotation': 10}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'axes.titlesize': FONT_SIZES['title'], 'axes.labelsize': FONT_SIZES['label'],
    'xtick.labelsize': FONT_SIZES['tick'], 'ytick.labelsize': FONT_SIZES['tick'],
    'figure.titlesize': FONT_SIZES['suptitle']
})


def plot_belief_evolution_simple(ax, logits_dict, target, supervision_indices):
    """在指定的ax上绘制简单的信念演化图。"""
    layer_indices = sorted(list(supervision_indices))
    final_layer_idx = max(layer_indices)
    probabilities = []
    for idx in layer_indices:
        key = 'final' if idx == final_layer_idx else f'aux_{idx}'
        if key in logits_dict:
            probabilities.append(torch.sigmoid(logits_dict[key].squeeze()).item())
        else:
            probabilities.append(float('nan'))

    ax.plot(layer_indices, probabilities, marker='o', linestyle='-', color='royalblue', markersize=7, linewidth=2)
    ax.axhline(y=0.5, color='black', linestyle=':', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(layer_indices)
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('P(MI)')
    ax.set_title(f"A: Belief Evolution (GT: {'MI' if target == 1 else 'Normal'})")
    return ax


def plot_self_attention_flow(ax, self_attns):
    """在指定的ax上绘制证据融合（自注意力）流的热力图。"""
    num_layers = len(self_attns)
    evidence_labels = ['to [BELIEF]', 'to [A2C]', 'to [A4C]']
    flow_matrix = np.zeros((len(evidence_labels), num_layers))

    for i, attn in enumerate(self_attns):
        attn_from_belief = attn[:, 0, :]
        attn_avg = attn_from_belief.mean(dim=0).cpu().numpy()
        if attn_avg.shape[0] == len(evidence_labels):
            flow_matrix[:, i] = attn_avg

    im = ax.imshow(flow_matrix, cmap="Blues", aspect='auto', interpolation='nearest')
    ax.set_yticks(np.arange(len(evidence_labels)), labels=evidence_labels)
    ax.set_xticks(np.arange(num_layers))
    ax.set_xlabel("Transformer Layer")
    ax.set_title("B: Evidence Fusion (From [BELIEF] Token)")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    return ax


def extract_keywords(prompts, max_len=30):
    """从长文本提示中提取关键词用于图表标签。"""
    keywords = []
    for p in prompts:
        # 移除非字母数字字符，并取前几个词
        clean_p = re.sub(r'[^A-Za-z0-9 ]+', '', p).strip()
        short_p = ' '.join(clean_p.split()[:5])
        if len(short_p) > max_len:
            short_p = short_p[:max_len] + '...'
        keywords.append(short_p)
    return keywords


def plot_knowledge_attribution_heatmap(ax, cross_attns, prompts):
    """(新) 在指定的ax上绘制知识归因（交叉注意力）的热力图。"""
    num_layers = len(cross_attns)
    num_prompts = cross_attns[0].shape[-1]

    # [num_prompts, num_layers]
    heatmap_matrix = np.zeros((num_prompts, num_layers))

    for i, attn in enumerate(cross_attns):
        # attn shape: [n_head, 1, num_prompts]
        # 平均所有头，并移除多余的维度
        attn_avg = attn.mean(dim=0).squeeze().cpu().numpy()
        if attn_avg.shape[0] == num_prompts:
            heatmap_matrix[:, i] = attn_avg

    im = ax.imshow(heatmap_matrix, cmap="Greens", aspect='auto', interpolation='nearest')

    # 使用关键词作为Y轴标签
    prompt_keywords = extract_keywords(prompts)
    ax.set_yticks(np.arange(len(prompt_keywords)), labels=prompt_keywords)
    ax.tick_params(axis='y', labelsize=FONT_SIZES['annotation'])

    ax.set_xticks(np.arange(num_layers))
    ax.set_xlabel("Transformer Layer")
    ax.set_title("C: Knowledge Attribution Heatmap (Cross-Attention)")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    return ax
