#####################################################################################################################
# 方案1：不同的节段用不同的颜色
######################################################################################################################

# # src/utils/visualizations_advanced.py (Final Polished Version with Detailed Comments)
#
# import matplotlib
#
# # --- 后端设置 ---
# # 设置Matplotlib后端为'Agg'。这是一个非交互式后端，它只将图形渲染到文件（如PNG）。
# # 这在服务器或没有图形用户界面的环境中运行代码时至关重要，可以防止程序因尝试打开GUI窗口而崩溃。
# matplotlib.use('Agg')
#
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from matplotlib.patches import Wedge  # 导入Wedge补丁，可用于在雷达图中绘制扇形区域
#
# # --- 全局绘图样式 ---
# # 使用'seaborn-v0_8-whitegrid'样式，这提供了一个美观的带有白色网格的绘图背景。
# plt.style.use('seaborn-v0_8-whitegrid')
#
# # --- 1. CVPR风格的颜色和字体配置 ---
# # --- 调色板定义 ---
# # 该区域定义了整个可视化脚本中使用的颜色方案，集中管理便于修改。
#
# # 为12个知识节段（A2C的6个 + A4C的6个）定义一个专业调色板。
# # "Paired" 是一个定性调色板，适合用于区分不同类别，且颜色两两配对。
# # 可调整参数:
# # - "Paired": 可以换成其他seaborn调色板，如 "viridis", "Set2", "husl" 等。
# # - 12: 节段的总数。
# SEGMENT_COLORS = sns.color_palette("Paired", 12)
#
# # 定义高亮颜色，用于标记雷达图中的Top-K分数。
# # 使用字典结构，可以为不同类型的图（如激活图和查询图）定义不同的高亮色。
# # 可调整参数:
# # - "#D90429": 可以修改为任何十六进制颜色码，以改变高亮边框的颜色。
# HIGHLIGHT_COLOR = {
#     "activation": "00F5D4",  # 激活高亮用醒目的红色#D90429
#     "query": "#00F5D4"  # 查询高亮用明亮的青色
# }
#
# # 其他绘图元素的颜色，如折线图和热力图。
# # 将这些颜色集中管理，确保了整个报告或论文中图形风格的一致性。
# # 可调整参数:
# # - sns.color_palette("colorblind")[0]: 使用对色盲友好的调色板，并选择其中的颜色。
# # - "viridis": 热力图的颜色映射方案，可以换成 "plasma", "magma", "coolwarm" 等。
# COLOR_PALETTE = {
#     "belief_line": sns.color_palette("colorblind")[0],
#     "confidence_line": sns.color_palette("colorblind")[2],
#     "heatmap_cmap": "viridis",
# }
#
# # --- 字体大小定义 ---
# # 以字典形式统一定义不同文本元素的字体大小，便于全局调整。
# FONT_SIZES = {
#     'title': 22,  # 图表标题
#     'suptitle': 28,  # (未使用) Figure的总标题
#     'label': 20,  # 坐标轴标签
#     'tick': 18,  # 坐标轴刻度
#     'annotation': 16,  # 注释文本
#     'segment_label': 11  # 雷达图节段标签
# }
#
# # --- Matplotlib全局参数更新 ---
# # 使用rcParams.update()将上述字体设置应用到所有后续生成的图中。
# plt.rcParams.update({
#     'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],  # 设置首选无衬线字体
#     'axes.titlesize': FONT_SIZES['title'],  # 应用坐标轴标题字体大小
#     'axes.labelsize': FONT_SIZES['label'],  # 应用坐标轴标签字体大小
#     'xtick.labelsize': FONT_SIZES['tick'],  # 应用x轴刻度字体大小
#     'ytick.labelsize': FONT_SIZES['tick'],  # 应用y轴刻度字体大小
#     'figure.titlesize': FONT_SIZES['suptitle'],  # 应用Figure标题字体大小
#     'legend.fontsize': FONT_SIZES['tick']  # 应用图例字体大小
# })
#
#
# def format_prompts_for_heatmap(prompts):
#     """
#     一个简单的辅助函数，为热力图的y轴标签格式化prompt文本。
#     它会在每个prompt前加上 "序号: " 的前缀。
#     """
#     return [f"{i + 1}: {p}" for i, p in enumerate(prompts)]
#
#
# def _plot_radar_chart(ax, scores, title, palette_key, top_k=1):
#     """
#     (内部核心函数) 绘制雷达图。此函数被其他上层函数调用。
#     它处理雷达图的所有核心逻辑，包括数据准备、颜色分配、绘制条形图、高亮和添加标签。
#
#     Args:
#         ax (matplotlib.axes.Axes): 用于绘图的matplotlib坐标轴对象 (应为极坐标)。
#         scores (np.ndarray): 包含所有prompt得分的一维numpy数组。
#         title (str): 图表的标题。
#         palette_key (str): 用于从HIGHLIGHT_COLOR字典中选择高亮颜色的键 ('activation' 或 'query')。
#         top_k (int): 需要高亮的得分数量。
#     """
#     # --- 数据结构定义 (硬编码) ---
#     # 这些值基于知识库的特定结构。如果知识库结构改变，需要修改这里。
#     num_a2c_segments = 6  # A2C知识区的节段数量
#     num_a4c_segments = 6  # A4C知识区的节段数量
#     num_a2c_prompts = num_a2c_segments * 2  # A2C区域的prompt总数 (每个节段2个)
#
#     # --- 数据准备 ---
#     # 从完整的scores数组中，根据结构截取出用于可视化的部分。
#     scores_a2c = scores[:num_a2c_prompts]
#     scores_a4c = scores[num_a2c_prompts: num_a2c_prompts + (num_a4c_segments * 2)]
#     scores_viz = np.concatenate([scores_a2c, scores_a4c])  # 将两部分拼接成最终的可视化数组
#
#     num_prompts_viz = len(scores_viz)
#     if num_prompts_viz == 0: return ax  # 如果没有数据，直接返回
#
#     # --- 角度和宽度计算 ---
#     # 计算每个柱状图在极坐标下的角度位置。
#     # np.linspace在0到2*pi之间生成等间距的角度。endpoint=False确保终点(2*pi)不包含在内，避免与起点(0)重叠。
#     angles = np.linspace(0, 2 * np.pi, num_prompts_viz, endpoint=False)
#     # 计算每个柱状图的宽度，确保它们均匀地填满整个圆形。
#     width = (2 * np.pi) / num_prompts_viz
#
#     # --- 1. 颜色分配 ---
#     # 为每个柱状图根据其所属的节段分配颜色。
#     bar_colors = []
#     # 为A2C的6个节段分配颜色，每个节段的2个prompt使用相同的颜色。
#     for i in range(num_a2c_segments):
#         bar_colors.extend([SEGMENT_COLORS[i]] * 2)
#     # 为A4C的6个节段分配颜色，同样每个节段2个prompt颜色相同。
#     for i in range(num_a4c_segments):
#         bar_colors.extend([SEGMENT_COLORS[i + num_a2c_segments]] * 2)
#
#     # --- 2. 绘制背景柱状图 ---
#     # 绘制所有prompt得分的基础柱状图。
#     # 可调整参数:
#     # - alpha=0.6: 条形的透明度，范围0-1。增加此值使颜色更深。
#     # - edgecolor='white': 条形之间的边界线颜色。
#     # - linewidth=0.5: 边界线宽度。
#     # - zorder=3: 绘图层次。值越高，越显示在顶层。确保它在高亮的下方，网格的上方。
#     ax.bar(angles, scores_viz, width=width, color=bar_colors, alpha=0.6, edgecolor='white', linewidth=0.5, zorder=3)
#
#     # --- 3. 分区高亮 Top-K ---
#     # 这一部分找到A2C和A4C区域中得分最高的k个prompt，并用更突出的样式重新绘制它们。
#
#     # 高亮A2C区域的Top-K
#     k_a2c = min(top_k, len(scores_a2c))
#     if k_a2c > 0:
#         top_k_indices_a2c = np.argsort(scores_a2c)[-k_a2c:]  # 找到得分最高的k个prompt的索引
#         highlight_colors_a2c = [bar_colors[i] for i in top_k_indices_a2c]  # 高亮条的颜色与原条一致
#         # 在原位置重新绘制高亮条
#         # 可调整参数:
#         # - alpha=0.9: 高亮条的不透明度，通常比背景条更实。
#         # - edgecolor: 高亮条的边框颜色，从HIGHLIGHT_COLOR字典中获取。
#         # - linewidth=2.0: 高亮条的边框宽度，比背景条更粗以示强调。
#         # - zorder=4: 确保高亮条显示在背景条之上。
#         ax.bar(angles[top_k_indices_a2c], scores_a2c[top_k_indices_a2c],
#                width=width, color=highlight_colors_a2c, alpha=0.9,
#                edgecolor=HIGHLIGHT_COLOR[palette_key], linewidth=2.0, zorder=4)
#
#     # 高亮A4C区域的Top-K (逻辑与A2C类似，但需要处理索引的偏移)
#     k_a4c = min(top_k, len(scores_a4c))
#     if k_a4c > 0:
#         top_k_indices_a4c_local = np.argsort(scores_a4c)[-k_a4c:]  # 在A4C分数内部找Top-K
#         top_k_indices_a4c_global = top_k_indices_a4c_local + num_a2c_prompts  # 转换为全局索引
#         highlight_colors_a4c = [bar_colors[i] for i in top_k_indices_a4c_global]
#         ax.bar(angles[top_k_indices_a4c_global], scores_a4c[top_k_indices_a4c_local],
#                width=width, color=highlight_colors_a4c, alpha=0.9,
#                edgecolor=HIGHLIGHT_COLOR[palette_key], linewidth=2.0, zorder=4)
#
#     # --- 4. 美化图表 ---
#     # 可调整参数:
#     # - scores_viz.max() * 1.4: Y轴的最大值。乘以一个大于1的系数（如1.4）是为了在图表顶部留出足够的空间来放置标签，避免重叠。
#     #   如果标签被截断，可以增大这个系数。
#     ylim_max = max(0.01, scores_viz.max() * 1.4)
#     ax.set_ylim(0, ylim_max)
#     ax.set_yticklabels([])  # 隐藏Y轴（半径轴）的刻度标签
#     ax.set_xticks(angles)  # 设置X轴（角度轴）的刻度位置为每个柱状图的中心
#     # 设置X轴的刻度标签为1到N的数字。
#     ax.set_xticklabels(range(1, num_prompts_viz + 1), fontsize=FONT_SIZES['annotation'] - 6, weight='bold')
#     # 设置图表标题。
#     # - pad=45: 调整标题与图表顶部的垂直距离。如果标题与标签重叠，增加此值。
#     ax.set_title(title, fontsize=FONT_SIZES['title'], weight='normal', pad=45)
#     # 显示网格线。
#     # - zorder=0: 确保网格线在最底层，不遮挡任何数据。
#     ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)
#
#     # --- 5. 【核心】绘制切向环绕的节段标签 ---
#     # 这部分是实现将文本标签水平（切向）放置在雷达图周围的关键。
#     # 可调整参数:
#     # - ylim_max * 1.25: 决定了标签距离圆心的半径。增大系数使标签离图表更远，反之则更近。
#     segment_label_radius = ylim_max * 0.85
#     a2c_segment_names = ["B-Ant", "M-Ant", "A-Ant", "B-Inf", "M-Inf", "A-Inf"]
#     a4c_segment_names = ["B-Sep", "M-Sep", "A-Sep", "B-Lat", "M-Lat", "A-Lat"]
#     all_segment_names = a2c_segment_names + a4c_segment_names
#
#     for i in range(len(all_segment_names)):
#         # 计算每个节段标签的中心角度。一个节段包含两个prompt，标签应位于两者之间。
#         # angles[i * 2 + 1] 是第二个prompt的角度，减去半个宽度(width / 2)就得到两个prompt中间的角度。
#         angle_rad = angles[i * 2 + 1] - (width / 2)
#         angle_deg = np.rad2deg(angle_rad)  # 将角度从弧度转换为度
#
#         # 计算文本的旋转角度。
#         # 目标是让文本的基线与半径垂直，即与圆周的切线平行。
#         # 默认情况下，0度的文本是水平的。因此，我们将角度减去90度来实现切向效果。
#         rotation = angle_deg - 90
#
#         # 为了避免在图表左侧（90到270度之间）的文本上下颠倒，可以取消下面代码块的注释。
#         # 这会使左侧的文本再次旋转180度，使其正向朝外，但可能会导致阅读方向不一致。
#         # 当前版本为了保持所有标签朝向一致（例如，顶部朝外），没有启用此逻辑。
#         # if 90 < angle_deg < 270:
#         #     rotation += 180
#
#         # 获取当前节段的颜色，用于标签的背景。
#         segment_color = SEGMENT_COLORS[i]
#
#         # 放置文本标签
#         # ax.text(x, y, text, ...), 在极坐标中，x是角度（弧度），y是半径。
#         ax.text(angle_rad, segment_label_radius, all_segment_names[i],
#                 ha='center', va='center',  # 水平和垂直对齐方式都设置为居中。
#                 rotation=rotation,  # 应用计算好的旋转角度。
#                 fontsize=FONT_SIZES['segment_label'], weight='bold',
#                 # bbox添加一个带背景色的边框。
#                 # - boxstyle="round,pad=0.5": 圆角矩形，pad控制文本与边框的间距。
#                 # - fc=segment_color: 背景填充色。
#                 # - alpha=0.4: 背景的透明度。
#                 bbox=dict(boxstyle="round,pad=0.7", fc=segment_color, ec='none', alpha=0.4))
#
#     return ax
#
#
# def plot_knowledge_activation_chart(ax, activation_scores, top_k=1):
#     """
#     (上层函数) 绘制初始知识激活的雷达图。
#     这是一个包装函数，它负责准备数据并调用核心的_plot_radar_chart函数。
#     """
#     scores = activation_scores.cpu().numpy()  # 将PyTorch Tensor转换为Numpy数组
#     return _plot_radar_chart(ax, scores, "Initial Knowledge Activation", "activation", top_k)
#
#
# def plot_knowledge_query_chart(ax, attention_weights, layer_idx, top_k=1):
#     """
#     (上层函数) 绘制知识查询的雷达图。
#     这也是一个包装函数，处理注意力权重的平均，并调用核心的_plot_radar_chart函数。
#     """
#     # 在head维度上取平均值，得到每个prompt的平均注意力权重
#     scores = attention_weights.mean(dim=0).squeeze().cpu().numpy()
#     return _plot_radar_chart(ax, scores, f"Knowledge Query at Layer {layer_idx}", "query", top_k)
#
#
# def plot_belief_confidence_dual_track(ax, logits_dict, confidence_dict, target, supervision_indices):
#     """
#     绘制信念与自信度双轨演化图 (折线图)。
#     """
#     layer_indices = sorted(list(supervision_indices))
#     probabilities, confidences = [], []
#     # 从字典中提取每个监督层的概率和置信度
#     for idx in layer_indices:
#         key = 'final' if idx == max(layer_indices) else f'aux_{idx}'
#         if key in logits_dict:
#             probabilities.append(torch.sigmoid(logits_dict[key].squeeze()).item())
#             confidences.append(confidence_dict[key].squeeze().item())
#         else:
#             probabilities.append(float('nan'))  # 如果数据缺失，则添加nan
#             confidences.append(float('nan'))
#
#     # --- 绘制折线 ---
#     # 可调整参数:
#     # - marker='o': 数据点的标记样式 ('o', '^', 's', 'x', ...)。
#     # - linestyle='-': 线的样式 ('-', '--', ':', '-.').
#     # - color: 线的颜色。
#     # - markersize=10: 标记的大小。
#     # - linewidth=3: 线的宽度。
#     ax.plot(layer_indices, probabilities, marker='o', linestyle='-', color=COLOR_PALETTE["belief_line"], markersize=10,
#             linewidth=3, label='P(MI)')
#     ax.plot(layer_indices, confidences, marker='^', linestyle='--', color=COLOR_PALETTE["confidence_line"],
#             markersize=10, linewidth=3, label='Confidence')
#
#     # 绘制决策阈值参考线
#     ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Decision Threshold')
#
#     # --- 美化图表 ---
#     ax.set_ylim(-0.05, 1.05)  # 设置Y轴范围，留出一点边距
#     ax.set_xticks(layer_indices)  # 确保每个监督层都有一个刻度
#     ax.set_xlabel('Transformer Layer')
#     ax.set_ylabel('Score')
#     ax.set_title(f"Belief & Confidence Evolution (GT: {'MI' if target == 1 else 'Normal'})",
#                  fontsize=FONT_SIZES['title'])
#     ax.legend()  # 显示图例
#     ax.grid(True, which='both', linestyle='--', linewidth=0.7)  # 显示网格
#     return ax
#
#
# def plot_knowledge_attribution_heatmap(ax, cross_attns, prompts, supervision_indices):
#     """
#     绘制知识归因热力图。
#     """
#     num_supervised_layers = len(cross_attns)
#     num_prompts = cross_attns[0].shape[-1]
#     heatmap_matrix = np.zeros((num_prompts, num_supervised_layers))
#
#     # 构建热力图矩阵：行是prompt，列是层，值是平均注意力
#     for i, attn in enumerate(cross_attns):
#         attn_avg = attn.mean(dim=0).squeeze().cpu().numpy()
#         if attn_avg.shape[0] == num_prompts:
#             heatmap_matrix[:, i] = attn_avg
#
#     # --- 绘制热力图 ---
#     # 可调整参数:
#     # - cmap: 颜色映射方案，从COLOR_PALETTE中获取。
#     # - aspect='auto': 调整单元格的宽高比以填充坐标轴区域。
#     # - interpolation='nearest': 像素插值方法，'nearest'会产生清晰的像素块。
#     im = ax.imshow(heatmap_matrix, cmap=COLOR_PALETTE["heatmap_cmap"], aspect='auto', interpolation='nearest')
#
#     # --- 设置坐标轴和标签 ---
#     prompt_labels = format_prompts_for_heatmap(prompts)
#     ax.set_yticks(np.arange(len(prompt_labels)), labels=prompt_labels)  # 设置Y轴刻度和标签
#     ax.tick_params(axis='y', labelsize=FONT_SIZES['tick'])  # 单独调整Y轴标签字体大小
#
#     ax.set_xticks(np.arange(num_supervised_layers))  # 设置X轴刻度和标签
#     ax.set_xticklabels([f"Layer {idx}" for idx in supervision_indices])
#
#     ax.set_xlabel("Supervised Transformer Layer")
#     ax.set_title("Knowledge Attribution Summary", fontsize=FONT_SIZES['title'])
#
#     # --- 添加颜色条 (Colorbar) ---
#     # pad=0.01: 颜色条与热力图之间的间距。
#     cbar = plt.colorbar(im, ax=ax, pad=0.01)
#
#     return ax


#####################################################################################################################
# 方案2：不同的节段用同一种颜色的不同渐变色
######################################################################################################################
# src/utils/visualizations_advanced.py (Final Polished Version with Full Comments)

import matplotlib

# --- 后端设置 ---
# 设置Matplotlib后端为'Agg'。这是一个非交互式后端，它只将图形渲染到文件（如PNG）。
# 这在服务器或没有图形用户界面的环境中运行代码时至关重要，可以防止程序因尝试打开GUI窗口而崩溃。
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Wedge

# --- 全局绘图样式 ---
# 使用'seaborn-v0_8-whitegrid'样式，这提供了一个美观的带有白色网格的绘图背景。
# 您可以尝试其他样式，如 'ggplot', 'dark_background', 'default'。
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. CVPR风格的颜色和字体配置 ---
# --- 调色板定义 ---
# 该区域集中定义了整个可视化脚本中使用的颜色方案，便于统一修改和管理。

# 为A2C和A4C知识区定义不同的基准色调，这是实现双色渐变的基础。
# 可调整参数: 可以将这里的十六进制颜色码替换为您希望的任何颜色。
A2C_BASE_COLOR = "#0077B6"  # 为A2C知识区选择一种沉稳的蓝色
A4C_BASE_COLOR = "#F77F00"  # 为A4C知识区选择一种活力的橙色

# 创建一个结构化的调色板字典，用以管理不同模式下（激活/查询）的颜色。
# - sns.light_palette: 基于一个基准色生成一个从浅到深的颜色列表（渐变色）。
#   - n_colors=7: 我们特意多生成一种颜色，再通过切片 [1:] 或 [:-1] 去掉最极端的一个，
#                 这样可以让渐变色带的视觉效果更和谐，避免纯白或纯黑。
#   - reverse=True/False: 控制渐变方向。这里我们让 activation 和 query 的渐变方向相反，
#                         从而在视觉上区分这两种不同的分析模式。
#   - a2c_highlight/a4c_highlight: 为每个区域定义一个醒目的、通常是较深的颜色，用于内部文字标签。
COLOR_PALETTE_CVPR = {
    "activation": {
        "a2c_segments": sns.light_palette(A2C_BASE_COLOR, n_colors=7, reverse=False)[1:],
        "a4c_segments": sns.light_palette(A4C_BASE_COLOR, n_colors=7, reverse=False)[1:],
        "a2c_highlight": "#0077B6",  # 为A2C选择一个非常深的蓝色作为高亮/标签色
        "a4c_highlight": "#F77F00"  # 为A4C选择一个非常深的橙色作为高亮/标签色
    },
    "query": {
        "a2c_segments": sns.light_palette(A2C_BASE_COLOR, n_colors=7, reverse=True)[:-1],
        "a4c_segments": sns.light_palette(A4C_BASE_COLOR, n_colors=7, reverse=True)[:-1],
        "a2c_highlight": "#0077B6",
        "a4c_highlight": "#F77F00"
    }
}

# 其他绘图元素的颜色，如折线图和热力图。
COLOR_PALETTE = {
    "belief_line": sns.color_palette("colorblind")[0],
    "confidence_line": sns.color_palette("colorblind")[2],
    "heatmap_cmap": "viridis",
}

# --- 字体大小定义 ---
# 以字典形式统一定义不同文本元素的字体大小，便于全局调整。
# 可调整参数: 修改字典中的数值即可改变所有图中对应元素的字体大小。
FONT_SIZES = {
    'title': 22,  # 图表标题
    'suptitle': 28,  # (未使用) Figure的总标题
    'label': 20,  # 坐标轴标签 (X/Y Label)
    'tick': 18,  # 坐标轴刻度 (Tick Label)
    'annotation': 16,  # 注释文本
    'segment_label': 11,  # 雷达图外部的节段标签
    'area_label': 12  # 雷达图内部的区域说明文字 (A2C/A4C Knowledge)
}
# --- Matplotlib全局参数更新 ---
# 使用rcParams.update()将上述字体设置应用到所有后续生成的图中，确保风格统一。
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],  # 设置首选无衬线字体
    'axes.titlesize': FONT_SIZES['title'],  # 应用坐标轴标题字体大小
    'axes.labelsize': FONT_SIZES['label'],  # 应用坐标轴标签字体大小
    'xtick.labelsize': FONT_SIZES['tick'],  # 应用x轴刻度字体大小
    'ytick.labelsize': FONT_SIZES['tick'],  # 应用y轴刻度字体大小
    'figure.titlesize': FONT_SIZES['suptitle'],  # 应用Figure标题字体大小
    'legend.fontsize': FONT_SIZES['tick']  # 应用图例字体大小
})


def format_prompts_for_heatmap(prompts):
    """
    一个简单的辅助函数，为热力图的y轴标签格式化prompt文本。
    它会在每个prompt前加上 "序号: " 的前缀，使其更易读。
    """
    return [f"{i + 1}: {p}" for i, p in enumerate(prompts)]


def _plot_radar_chart(ax, scores, title, palette_key, top_k=1):
    """
    (内部核心函数) 绘制带节段渐变色和内部区域标识的雷达图。

    Args:
        ax (matplotlib.axes.Axes): 用于绘图的matplotlib坐标轴对象 (应为极坐标)。
        scores (np.ndarray): 包含所有prompt得分的一维numpy数组。
        title (str): 图表的标题。
        palette_key (str): 用于从COLOR_PALETTE_CVPR字典中选择颜色方案的键 ('activation' 或 'query')。
        top_k (int): 需要用边框高亮的得分数量。
    """
    # --- 数据结构定义 (硬编码) ---
    # 这些值基于知识库的特定结构。如果您的知识库prompt数量或分段方式改变，需要修改这里。
    num_a2c_segments = 6
    num_a4c_segments = 6
    num_a2c_prompts = num_a2c_segments * 2

    # --- 数据准备 ---
    # 从完整的scores数组中，根据结构截取出用于可视化的部分。
    scores_a2c = scores[:num_a2c_prompts]
    scores_a4c = scores[num_a2c_prompts: num_a2c_prompts + (num_a4c_segments * 2)]
    scores_viz = np.concatenate([scores_a2c, scores_a4c])  # 拼接成最终的可视化数组

    num_prompts_viz = len(scores_viz)
    if num_prompts_viz == 0: return ax  # 如果没有数据，直接返回

    # --- 角度和宽度计算 ---
    # 计算每个柱状图在极坐标下的角度位置和宽度。
    angles = np.linspace(0, 2 * np.pi, num_prompts_viz, endpoint=False)
    width = (2 * np.pi) / num_prompts_viz

    # --- 1. 颜色分配 ---
    # 根据传入的palette_key（"activation"或"query"）获取对应的颜色方案。
    palette = COLOR_PALETTE_CVPR[palette_key]
    bar_colors = []
    # 为A2C的6个节段分配渐变色 (每个节段的2个prompt使用同一种颜色)。
    for i in range(num_a2c_segments):
        bar_colors.extend([palette["a2c_segments"][i]] * 2)
    # 为A4C的6个节段分配渐变色。
    for i in range(num_a4c_segments):
        bar_colors.extend([palette["a4c_segments"][i]] * 2)

    # --- 2. 绘制背景柱状图 ---
    # 绘制所有prompt得分的基础柱状图。
    # 可调整参数:
    # - alpha=0.7: 条形的透明度 (0-1)。增加此值使颜色更深。
    # - edgecolor='white': 条形之间的边界线颜色。
    # - linewidth=0.5: 边界线宽度。
    # - zorder=3: 绘图层次。值越高，越显示在顶层。确保它在高亮的下方，网格的上方。
    ax.bar(angles, scores_viz, width=width, color=bar_colors, alpha=0.7, edgecolor='white', linewidth=0.5, zorder=3)

    # --- 3. 【核心修改】用红色边框高亮 Top-K ---
    # 这一部分找到A2C和A4C区域中得分最高的k个prompt，并用红色粗边框重新绘制它们。

    # 高亮A2C区域的Top-K
    k_a2c = min(top_k, len(scores_a2c))
    if k_a2c > 0:
        top_k_indices_a2c = np.argsort(scores_a2c)[-k_a2c:]  # 找到得分最高的k个prompt的索引
        highlight_colors_a2c = [bar_colors[i] for i in top_k_indices_a2c]  # 获取这些bar的原始颜色
        # 在原位置重新绘制高亮条，但这次只改变边框。
        # 可调整参数:
        # - color: 保持为原始颜色，不再使用单独的高亮填充色。
        # - alpha=0.9: 透明度比背景稍高，使其在视觉上更突出一点。
        # - edgecolor='red': 高亮条的边框颜色，按要求设为红色。
        # - linewidth=2.5: 高亮条的边框宽度，设为较粗的值以示强调。
        # - zorder=4: 确保高亮条显示在背景条之上。
        ax.bar(angles[top_k_indices_a2c], scores_a2c[top_k_indices_a2c],
               width=width, color=highlight_colors_a2c,
               alpha=0.9, edgecolor='red', linewidth=2.5, zorder=4)

    # 高亮A4C区域的Top-K (逻辑与A2C完全相同，仅索引和数据源不同)
    k_a4c = min(top_k, len(scores_a4c))
    if k_a4c > 0:
        top_k_indices_a4c_local = np.argsort(scores_a4c)[-k_a4c:]  # 在A4C分数内部找Top-K
        top_k_indices_a4c_global = top_k_indices_a4c_local + num_a2c_prompts  # 转换为全局索引
        highlight_colors_a4c = [bar_colors[i] for i in top_k_indices_a4c_global]
        ax.bar(angles[top_k_indices_a4c_global], scores_a4c[top_k_indices_a4c_local],
               width=width, color=highlight_colors_a4c,
               alpha=0.9, edgecolor='red', linewidth=2.5, zorder=4)

    # --- 4. 美化图表 ---
    # 可调整参数:
    # - scores_viz.max() * 1.6: Y轴（半径）的最大值。乘以一个大于1的系数是为了在图表顶部留出足够的空间
    #   来放置外部标签和标题，避免重叠。如果标签被截断或离得太近，可以增大这个系数(如1.7)。
    ylim_max = max(0.01, scores_viz.max() * 1.6)
    ax.set_ylim(0, ylim_max)
    ax.set_yticklabels([])  # 隐藏Y轴（半径轴）的刻度标签，保持图表简洁
    ax.set_xticks(angles)  # 设置X轴（角度轴）的刻度位置为每个柱状图的中心
    ax.set_xticklabels(range(1, num_prompts_viz + 1), fontsize=FONT_SIZES['annotation'] - 6, weight='bold')
    # 设置图表标题。
    # - pad=45: 调整标题与图表顶部的垂直距离。如果标题与外部标签重叠，增加此值。
    ax.set_title(title, fontsize=FONT_SIZES['title'], weight='normal', pad=45)
    # 显示网格线。
    # - zorder=0: 确保网格线在最底层，不遮挡任何数据。
    ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)

    # --- 5. 绘制外部节段标签 ---
    # 可调整参数:
    # - ylim_max * 1.25: 决定了标签距离圆心的半径。增大系数使标签离图表更远，反之则更近。
    segment_label_radius = ylim_max * 0.85
    a2c_segment_names = ["B-Ant", "M-Ant", "A-Ant", "B-Inf", "M-Inf", "A-Inf"]
    a4c_segment_names = ["B-Sep", "M-Sep", "A-Sep", "B-Lat", "M-Lat", "A-Lat"]

    # 绘制A2C的6个节段标签
    for i in range(num_a2c_segments):
        angle_rad = angles[i * 2 + 1] - (width / 2)  # 计算标签中心角度
        rotation = np.rad2deg(angle_rad) - 90  # 计算旋转角度，使其与圆周切线平行
        # 标签背景色使用对应的渐变色，增强关联性。
        ax.text(angle_rad, segment_label_radius, a2c_segment_names[i],
                ha='center', va='center', rotation=rotation,
                fontsize=FONT_SIZES['segment_label'], weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", fc=palette["a2c_segments"][i], ec='none', alpha=0.4))

    # 绘制A4C的6个节段标签
    for i in range(num_a4c_segments):
        angle_rad = angles[num_a2c_prompts + i * 2 + 1] - (width / 2)
        rotation = np.rad2deg(angle_rad) - 90
        ax.text(angle_rad, segment_label_radius, a4c_segment_names[i],
                ha='center', va='center', rotation=rotation,
                fontsize=FONT_SIZES['segment_label'], weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", fc=palette["a4c_segments"][i], ec='none', alpha=0.4))

    # --- 6. 在图内标识A2C和A4C区域 ---
    # 可调整参数:
    # - ylim_max * 0.5: 内部文字标签的放置半径。0.5大约在半径一半的位置。
    #   如果内部空间拥挤或空旷，可以调整此系数 (0.1 - 0.8)。
    area_label_radius = ylim_max * 0.7
    # 计算A2C区域和A4C区域的中心角度，以确定文字放置的位置。
    a2c_label_angle = np.mean(angles[:num_a2c_prompts])
    a4c_label_angle = np.mean(angles[num_a2c_prompts:])

    # 添加A2C知识区标签
    ax.text(a2c_label_angle, area_label_radius, "A2C\nKnowledge",
            rotation=np.rad2deg(a2c_label_angle) - 90,  # 旋转文字使其与所在角度对齐
            ha='center', va='center',
            fontsize=FONT_SIZES['area_label'],  # 使用为区域标签定义的字体大小
            color=palette["a2c_highlight"],  # 文字颜色与区域主题色一致
            weight='bold', alpha=0.8)

    # 添加A4C知识区标签
    ax.text(a4c_label_angle, area_label_radius, "A4C\nKnowledge",
            rotation=np.rad2deg(a4c_label_angle) - 90,
            ha='center', va='center',
            fontsize=FONT_SIZES['area_label'],
            color=palette["a4c_highlight"],
            weight='bold', alpha=0.8)

    return ax


def plot_knowledge_activation_chart(ax, activation_scores, top_k=1):
    """
    (上层函数) 绘制初始知识激活的雷达图。
    这是一个包装函数，它负责准备数据并调用核心的_plot_radar_chart函数。
    """
    scores = activation_scores.cpu().numpy()  # 将PyTorch Tensor转换为Numpy数组
    return _plot_radar_chart(ax, scores, "Initial Knowledge Activation", "activation", top_k)


def plot_knowledge_query_chart(ax, attention_weights, layer_idx, top_k=1):
    """
    (上层函数) 绘制知识查询的雷达图。
    这也是一个包装函数，处理注意力权重的平均，并调用核心的_plot_radar_chart函数。
    """
    # 在head维度上取平均值，得到每个prompt的平均注意力权重
    scores = attention_weights.mean(dim=0).squeeze().cpu().numpy()
    return _plot_radar_chart(ax, scores, f"Knowledge Query at Layer {layer_idx}", "query", top_k)


def plot_belief_confidence_dual_track(ax, logits_dict, confidence_dict, target, supervision_indices):
    """
    绘制信念与自信度双轨演化图 (折线图)。
    """
    layer_indices = sorted(list(supervision_indices))
    probabilities, confidences = [], []
    # 从字典中提取每个监督层的概率和置信度
    for idx in layer_indices:
        key = 'final' if idx == max(layer_indices) else f'aux_{idx}'
        if key in logits_dict:
            probabilities.append(torch.sigmoid(logits_dict[key].squeeze()).item())
            confidences.append(confidence_dict[key].squeeze().item())
        else:
            probabilities.append(float('nan'))  # 如果数据缺失，则添加nan，防止绘图中断
            confidences.append(float('nan'))

    # --- 绘制折线 ---
    # 可调整参数:
    # - marker='o': 数据点的标记样式 ('o', '^', 's', 'x', ...)。
    # - linestyle='-': 线的样式 ('-', '--', ':', '-.').
    # - markersize=10: 标记的大小。
    # - linewidth=3: 线的宽度。
    ax.plot(layer_indices, probabilities, marker='o', linestyle='-', color=COLOR_PALETTE["belief_line"], markersize=10,
            linewidth=3, label='P(MI)')  # P(MI) : 心肌梗死的概率
    ax.plot(layer_indices, confidences, marker='^', linestyle='--', color=COLOR_PALETTE["confidence_line"],
            markersize=10, linewidth=3, label='Confidence')

    # 绘制一条y=0.5的水平虚线，作为决策阈值的参考线。
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Decision Threshold')

    # --- 美化图表 ---
    ax.set_ylim(-0.05, 1.05)  # 设置Y轴范围，比[0,1]稍大以避免标记点被边缘截断
    ax.set_xticks(layer_indices)  # 确保每个监督层都有一个刻度
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Score')
    ax.set_title(f"Belief & Confidence Evolution (GT: {'MI' if target == 1 else 'Normal'})",
                 fontsize=FONT_SIZES['title'])
    ax.legend()  # 显示图例
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)  # 显示网格
    return ax


def plot_knowledge_attribution_heatmap(ax, cross_attns, prompts, supervision_indices):
    """
    绘制知识归因热力图，展示不同层对不同知识prompt的关注度。
    """
    num_supervised_layers = len(cross_attns)
    num_prompts = cross_attns[0].shape[-1]
    # 初始化一个空矩阵来存储热力图数据
    heatmap_matrix = np.zeros((num_prompts, num_supervised_layers))

    # 填充热力图矩阵：行是prompt，列是层，值是该层对该prompt的平均注意力权重
    for i, attn in enumerate(cross_attns):
        attn_avg = attn.mean(dim=0).squeeze().cpu().numpy()
        if attn_avg.shape[0] == num_prompts:
            heatmap_matrix[:, i] = attn_avg

    # --- 绘制热力图 ---
    # 可调整参数:
    # - cmap: 颜色映射方案，例如 "viridis", "plasma", "magma", "coolwarm"。
    # - aspect='auto': 调整单元格的宽高比以自动填充坐标轴区域。
    # - interpolation='nearest': 像素插值方法，'nearest'会产生清晰的像素块，适合展示矩阵。
    im = ax.imshow(heatmap_matrix, cmap=COLOR_PALETTE["heatmap_cmap"], aspect='auto', interpolation='nearest')

    # --- 设置坐标轴和标签 ---
    prompt_labels = format_prompts_for_heatmap(prompts)
    ax.set_yticks(np.arange(len(prompt_labels)), labels=prompt_labels)  # 设置Y轴刻度和标签为prompt文本
    ax.tick_params(axis='y', labelsize=FONT_SIZES['tick'])  # 单独调整Y轴标签字体大小

    ax.set_xticks(np.arange(num_supervised_layers))  # 设置X轴刻度和标签为层编号
    ax.set_xticklabels([f"Layer {idx}" for idx in supervision_indices])

    ax.set_xlabel("Supervised Transformer Layer")
    ax.set_title("Knowledge Attribution Summary", fontsize=FONT_SIZES['title'])

    # --- 添加颜色条 (Colorbar) ---
    # pad=0.01: 控制颜色条与热力图之间的间距。
    cbar = plt.colorbar(im, ax=ax, pad=0.01)

    return ax
